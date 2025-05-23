from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from state import AgentState
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0
)

def personality_agent(state: AgentState) -> AgentState:
    """Personality agent that formats and enhances responses."""
    try:
        # Define personality prompt
        personality_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a personality enhancement agent for InfinitePay's customer service.
            
            PERSONALITY TRAITS:
            1. Professional yet approachable
            2. Confident but humble
            3. Clear and concise
            4. Empathetic and understanding
            5. Solution-focused
            6. Tech-savvy but accessible
            
            COMMUNICATION GUIDELINES:
            1. Use positive language
            2. Show empathy for concerns
            3. Maintain professional tone
            4. Be clear about next steps
            5. Keep technical accuracy
            6. Preserve important details
            
            RESPONSE STRUCTURE:
            1. Acknowledge the query/concern
            2. Provide clear information/solution
            3. Add empathetic touch
            4. Include next steps if any
            5. Invite further questions
            
            Transform the input while maintaining all factual information."""),
            ("human", "{input}")
        ])

        # Initialize personality config
        if "personality_config" not in state:
            state["personality_config"] = {
                "style": "professional",
                "tone": "friendly",
                "language_level": "clear"
            }
        
        # Get original response
        original_response = ""
        if state.get("agent_outcome") and "output" in state["agent_outcome"]:
            original_response = state["agent_outcome"]["output"]
        elif state.get("messages") and state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage):
                original_response = last_message.content

        # Transform response
        personality_response = original_response
        if original_response:
            chain = personality_prompt | llm
            result = chain.invoke({"input": original_response})
            personality_response = result.content

        # Update tool usage with personality LLM call
        if original_response:
            state["tool_outputs"]["personality_llm"] = [{
                "input": original_response,
                "output": personality_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }]
        
        # Track this transformation in workflow history
        state["workflow_history"].append({
            "agent_name": "personality",
            "action": "enhance_response",
            "input": original_response,
            "output": personality_response,
            "tool_calls": state.get("tool_outputs", {}),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Use the complete workflow history
        workflow = state.get("workflow_history", [])
        
        # Construct final response format
        final_output = {
            "response": personality_response,
            "source_agent_response": original_response,
            "agent_workflow": workflow,
            "conversation_active": state.get("conversation_active", True),
            "needs_followup": state.get("needs_followup", True),
            "error": state.get("error")
        }
        
        # Update state
        state["messages"].append(AIMessage(content=personality_response))
        state["personality_output"] = final_output
        
        # Update agent stack
        state["current_agent"] = "personality"
        if "agent_stack" not in state:
            state["agent_stack"] = []
        state["agent_stack"].append("personality")
        
        # Update workflow history
        state["workflow_history"].append({
            "agent_name": "personality",
            "action": "enhance_response",
            "input": original_response,
            "output": personality_response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return final_output

    except Exception as e:
        error_message = f"Error in personality agent: {str(e)}"
        state["error"] = error_message
        return {
            "response": "I apologize, but I'm having trouble formatting the response. Let me provide the direct answer:",
            "source_agent_response": original_response if 'original_response' in locals() else "Error retrieving original response",
            "agent_workflow": [{
                "agent_name": "error",
                "tool_calls": {}
            }],
            "conversation_active": True,
            "needs_followup": True
        }

