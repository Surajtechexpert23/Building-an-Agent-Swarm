from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from state import AgentState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentFinish
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from tools import rag_search
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0
)

def knowledge_agent(state: AgentState) -> AgentState:
    """Knowledge agent that provides information about InfinitePay services."""
    # Initialize and update state
    state["current_agent"] = "knowledge"
    if "agent_stack" not in state:
        state["agent_stack"] = []
    state["agent_stack"].append("knowledge")
    
    # Initialize tool tracking state
    if "tool_outputs" not in state:
        state["tool_outputs"] = {}
    if "workflow_history" not in state:
        state["workflow_history"] = []
    if "knowledge_context" not in state:
        state["knowledge_context"] = {}
    state["workflow_history"].append({
        "agent_name": "knowledge",
        "action": "start_query",
        "input": state["input"],
        "tool_calls": {
            "status": "started",
            "tools_available": ["rag_search", "web_search"]
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Initialize tools
    web_search = TavilySearchResults(max_results=5)
    rag_tool = rag_search
    tools = [web_search, rag_tool]
    
    # Define the agent's prompt
    knowledge_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable agent specializing in InfinitePay's services and products.
        
        YOUR TOOLS:
        1. rag_search: Find information from InfinitePay's official documentation
        2. web_search: Find general information from the web
        
        INSTRUCTIONS:
        1. Always use rag_search FIRST for InfinitePay-specific information
        2. Use web_search for complementary or general information
        3. Combine information from both sources when relevant
        4. Be specific about features, pricing, and requirements
        5. Always validate web search information against official docs
        6. If information conflicts, trust rag_search over web_search
        
        RESPONSE GUIDELINES:
        1. Be concise but comprehensive
        2. Structure information clearly with categories or bullet points
        3. Include specific details about:
           - Features and benefits
           - Pricing if available
           - Technical requirements
           - Integration capabilities
        4. Note when information is from web search vs official docs
        5. Always ask if user needs clarification
        
        Keep responses professional and accurate."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])
    
    # Create and configure the agent
    prompt = hub.pull("hwchase17/react")
    agent_with_tools = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Create the executor
    agent_executor = AgentExecutor(
        agent=agent_with_tools, 
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    try:
        # Execute the agent
        response = agent_executor.invoke({
            "input": state["input"]
        })
        
        # Process response
        response_content = response["output"]
        
        # Track tool usage from intermediate steps
        tool_calls = {}
        for step in response.get("intermediate_steps", []):
            action, output = step
            tool_name = action.tool
            if tool_name not in tool_calls:
                tool_calls[tool_name] = {
                    "calls": [],
                    "total_uses": 0,
                    "last_used": None
                }
            tool_calls[tool_name]["calls"].append({
                "input": action.tool_input,
                "output": output,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            tool_calls[tool_name]["total_uses"] += 1
            tool_calls[tool_name]["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Update state with tool usage
        state["tool_outputs"].update(tool_calls)
        
        # Add completion to workflow history
        state["workflow_history"].append({
            "agent_name": "knowledge",
            "action": "complete_query",
            "input": state["input"],
            "output": response_content,
            "tool_calls": tool_calls,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Update messages
        state["messages"].append(AIMessage(content=response_content))
        
        # Set agent outcome using AgentFinish
        state["agent_outcome"] = AgentFinish(
            return_values={"output": response_content},
            log=str(tool_calls)
        )
        
        # Update state flags
        state["error"] = None
        state["needs_followup"] = True
        state["is_complete"] = True
        state["conversation_active"] = True
        
        return state
        
    except Exception as e:
        error_message = f"Error in knowledge agent: {str(e)}"
        print(error_message)
        state["error"] = error_message
        state["messages"].append(AIMessage(content=
            "I apologize, but I encountered an error while retrieving the information. "
            "Could you please rephrase your question?"
        ))
        state["needs_followup"] = True
        return state
