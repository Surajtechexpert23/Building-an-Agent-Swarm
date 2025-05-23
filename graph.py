from typing import Dict
from langgraph.graph import Graph, END
from state import AgentState
from agents import route_message, knowledge_agent, customer_support_agent, personality_agent

def should_continue(state: AgentState) -> str:
    """
    Determine if we should continue the conversation or end it.
    Returns:
        str: Either 'router' to continue conversation or 'end' to terminate
    """
    # Check for error condition
    if state.get("error"):
        print("Ending due to error condition")
        return END
    # Get the last message
    last_message = state["messages"][-1].content if state["messages"] else ""

    # End indicators for natural conversation endings
    end_indicators = [
        "goodbye",
        "thank you",
        "thanks",
        "have a good day",
        "that's all",
        "ticket has been created",
        "appointment has been scheduled"
    ]

    # Check if this is the end of the conversation
    if any(indicator in last_message.lower() for indicator in end_indicators):
        print("Ending due to natural conversation end")
        return END

    # Check if we should continue with follow-up
    if state.get("needs_followup", True):
        # Reset a completion flag for the next round
        state["is_complete"] = False
        # Add a follow-up prompt to the input
        state["input"] = "Do you have any other questions or is there anything else I can help you with?"
        return "router"

    return END

def create_graph() -> Graph:
    """Create the workflow graph connecting the agents."""
    workflow = Graph()
    # Add nodes
    workflow.add_node("router", route_message)
    workflow.add_node("knowledge", knowledge_agent)
    workflow.add_node("support", customer_support_agent)
    workflow.add_node("personality", personality_agent)

    # Router edge function (unchanged)
    def route_edge(state: AgentState) -> str:
        """Route to the next agent based on the router's decision."""
        if "messages" not in state:
            state["messages"] = []

        if state.get("error"):
            return END
        next_agent = state.get("next", "")
        if not next_agent:
            print("No next agent specified, routing to support agent by default.")
            state["next"] = "support"
            return "support"

        next_agent = next_agent.strip().lower()
        if next_agent in ["end", "__end__", END]:
            print("Routing to END state")
            return END

        print(f"Routing to: {next_agent}")
        if next_agent in ["knowledge", "support", "router"]:
            return next_agent

        print(f"Invalid agent '{next_agent}', routing to support agent by default.")
        state["next"] = "support"
        return "support"

    # Add edges for router with updated mapping
    workflow.add_conditional_edges(
        "router",
        route_edge,
        {
            "knowledge": "knowledge",
            "support": "support",
            "router": "router",
            END: END  # Handle END directly
        }
    )

    # Connect knowledge and support agents to personality
    workflow.add_edge("knowledge", "personality")
    workflow.add_edge("support", "personality")
    workflow.add_edge("personality", END)

    # Define personality-edge function (unchanged)
    def personality_edge(state: AgentState) -> str:
        """Route after personality transformation."""
        if "messages" not in state:
            state["messages"] = []

        if state.get("error"):
            print("Ending due to error in personality edge")
            return END

        result = should_continue(state)

        if result == "router":
            state["tool_result"] = None
            state["last_tool"] = None
            state["agent_outcome"] = None
            state["current_agent"] = "router"
        elif result == END:
            state["knowledge_context"] = {}
            current_history = state.get("support_context", {}).get("interaction_history", [])
            state["support_context"] = {
                "current_ticket": None,
                "current_appointment": None,
                "interaction_history": current_history
            }
            print("Routing back to router for follow-up")
        else:
            print("Ending conversation naturally")

        print(f"Personality edge decision: {result}")
        return result

    # Add conditional edge from personality with updated mapping
    workflow.add_conditional_edges(
        "personality",
        personality_edge,
        {
            "router": "router",
            END: END  # Handle END directly
        }
    )

    # Set the entry point
    workflow.set_entry_point("router")

    return workflow

def cleanup_state(state: AgentState) -> None:
    """Clean up the state after agent execution."""
    # Remove completed agent from stack
    if state.get("agent_stack"):
        state["agent_stack"].pop()
        if state["agent_stack"]:
            state["current_agent"] = state["agent_stack"][-1]
        else:
            state["current_agent"] = None

    # Clear temporary data
    if state.get("is_complete"):
        state["knowledge_context"] = {}
        state["support_context"] = {}

    # Update conversation state
    if state.get("error"):
        state["conversation_active"] = False
        state["needs_followup"] = False
    elif not state.get("task_list") and not state.get("needs_followup"):
        state["conversation_active"] = False

def invoke_graph(message: str) -> Dict:
    """Invoke the agent workflow with a message."""

    # Create the initial state
    state = {
        # Core conversation state
        "input": message,
        "messages": [],
        "next": "",
        "error": None,
        "agent_outcome": None,

        # Tool tracking
        "tool_outputs": {},  # Store outputs by tool name
        "tool_usage": [],    # Track all tool calls
        "last_tool": None,   # Last tool used
        "tool_result": None, # Last tool result

        # Workflow state
        "workflow_history": [],  # Track agent interactions
        "current_agent": None,   # Current active agent
        "agent_stack": [],       # Track agent call stack

        # Conversation control
        "conversation_active": True,
        "needs_followup": True,
        "is_complete": False,
        "task_list": [],         # Pending tasks

        # Agent-specific states
        "personality_output": None,
        "knowledge_context": {},  # Store RAG context
        "support_context": {}     # Store support interaction context
    }

    # Create and run the graph
    workflow = create_graph()
    app = workflow.compile()
    result = app.invoke(state)

    # Clean up state after execution
    cleanup_state(state)

    # Ensure a result has required fields
    if not isinstance(result, dict):
        return {
            "response": "An error occurred processing your request.",
            "source_agent_response": "",
            "messages": [],
            "conversation_active": False,
            "needs_followup": False,
            "agent_workflow": state.get("workflow_history", []),
            "error": "Invalid result type",
            "is_complete": True
        }

    # Add any missing required fields
    result.setdefault("response", "")
    result.setdefault("source_agent_response", "")
    result.setdefault("messages", [])
    result.setdefault("conversation_active", True)
    result.setdefault("needs_followup", True)
    result.setdefault("agent_workflow", state.get("workflow_history", []))
    result.setdefault("error", None)
    result.setdefault("is_complete", False)

    return result

if __name__ == "__main__":
    # Example usage
    message = "What services does InfinitePay offer?"
    result = invoke_graph(message)

    # Print the conversation result
    print("\nConversation:")
    if isinstance(result, dict):
        print(f"Assistant: {result.get('response', 'No response available')}\n")