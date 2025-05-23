from typing import Dict, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from state import AgentState
from langgraph.graph import END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize router LLM with consistent settings
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0
)

def route_message(state: AgentState) -> AgentState:
    """Router agent that determines which agent should handle the message."""
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing agent that analyzes user messages and determines the best agent to handle them.
        
        Available agents and their responsibilities:
        
        1. Knowledge Agent (knowledge):
           - Provides general information about InfinitePay and its services
           - Answers questions about product features and capabilities
           - Handles general inquiries not related to specific issues or problems
           - Uses company website content and web search for accurate information
           
        2. Customer Support Agent (support):
           - Handles technical issues and error reports
           - Manages payment-related problems
           - Processes refund requests
           - Creates and manages support tickets
           - Schedules support calls
           - Provides FAQ information for common issues
           - Assists with account-specific problems
         
        Instructions:
        1. Analyze the user's message carefully
        2. Return ONLY ONE of these exact strings: knowledge or support or end
        3. Do not provide any additional information or explanations
        4. If the message is unclear, ask clarifying questions to determine the best agent
        5. If you do not know where to route, choose support
        6. Check agent outcome and if successful, move to next task in list or end
        7. If get anything that is non meaning full, return end
         
        Examples:
        - "What services does InfinitePay offer?" → knowledge
        - "My payment isn't going through" → support
        - "How do I integrate the API?" → knowledge
        - "I need help with an error" → support
        - "How much does a card terminal cost?" → knowledge
        - "I want to schedule a support call" → support
        - "Goodbye" → end
        - "That's all I needed" → end"""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])
    
    try:
        # Initialize and update state
        state["current_agent"] = "router"
        if "agent_stack" not in state:
            state["agent_stack"] = []
        state["agent_stack"].append("router")
        
        # Initialize conversation state if not present
        if "conversation_active" not in state:
            state["conversation_active"] = True
        if "needs_followup" not in state:
            state["needs_followup"] = True
        if "task_list" not in state:
            state["task_list"] = []
        if "workflow_history" not in state:
            state["workflow_history"] = []
        if "tool_outputs" not in state:
            state["tool_outputs"] = {}            # Track start of routing in workflow history
        state["workflow_history"].append({
            "agent_name": "router",
            "action": "start_routing",
            "input": state.get("input", ""),
            "tool_calls": {
                "router_llm": {
                    "calls": [],
                    "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "status": "initialized"
                }
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
            
        # Check for explicit end command
        if state.get("input", "").lower() in ["goodbye", "bye", "exit", "quit", "end"]:
            state["conversation_active"] = False
            state["next"] = END  # Use END constant instead of string
            return state

        # Get routing decision from LLM
        chain = router_prompt | llm
        response = chain.invoke({
            "messages": state["messages"],
            "input": state["input"]
        })

        # Clean and validate response
        next_agent = response.content.lower().strip()
        
        # Update tool usage with LLM call
        if "router_llm" not in state["tool_outputs"]:
            state["tool_outputs"]["router_llm"] = {
                "calls": [],
                "total_uses": 0,
                "last_used": None
            }
            
        # Add new call to history
        state["tool_outputs"]["router_llm"]["calls"].append({
            "input": state["input"],
            "output": next_agent,
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Update tool usage stats
        state["tool_outputs"]["router_llm"]["total_uses"] += 1
        state["tool_outputs"]["router_llm"]["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert string responses to proper states
        if next_agent == "end":
            next_agent = END
            state["conversation_active"] = False
            state["needs_followup"] = False
        elif next_agent not in ["knowledge", "support"]:
            print(f"Unexpected response from router: {next_agent}, defaulting to support")
            next_agent = "support"
            
        # Add completion to workflow history
        state["workflow_history"].append({
            "agent_name": "router",
            "action": "complete_routing",
            "input": state["input"],
            "output": next_agent,
            "tool_calls": {
                "router_llm": {
                    "calls": state["tool_outputs"]["router_llm"]["calls"],
                    "total_uses": state["tool_outputs"]["router_llm"]["total_uses"],
                    "last_used": state["tool_outputs"]["router_llm"]["last_used"],
                    "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "status": "success"
                }
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Handle conversation state and agent outcomes
        if state.get("agent_outcome"):
            if state.get("needs_followup"):
                next_agent = state.get("next", "support")
            elif state["task_list"]:
                next_task = state["task_list"].pop(0)
                state["input"] = next_task
                next_agent = "support"  # Default to support for new tasks
            else:
                next_agent = END  # No more tasks, end the conversation
                state["conversation_active"] = False
                state["needs_followup"] = False
        elif not state.get("conversation_active"):
            next_agent = END  # Use END constant instead of string

        # Validate agent assignment and handle end states
        if isinstance(next_agent, str):
            next_agent = next_agent.strip().lower()
            if next_agent == "end":
                next_agent = END
                
        if next_agent not in ["knowledge", "support", END]:
            print(f"Invalid router response '{next_agent}', defaulting to support")
            next_agent = "support"

        # Prevent automatic ending unless explicitly requested
        if next_agent == END and state.get("conversation_active", True) and state.get("needs_followup", True):
            print("Preventing automatic end, routing to support for follow-up")
            next_agent = "support"

        state["next"] = next_agent
        return state

    except Exception as e:
        print(f"Error in route_message: {str(e)}")
        state["next"] = "support"  # Default to support on error
        state["error"] = str(e)
        return state
