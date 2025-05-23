from langchain_core.tools import tool
from state import AgentState
from datetime import datetime
import uuid
from rag import RAGManager


@tool
def rag_search(query: str, state: AgentState = None) -> str:
    """
    search for information using RAG (Retrieval-Augmented Generation).

    Args:
        query (str): The search query.
        state (AgentState, optional): The agent's current state.

    Returns:
        str: The search result or error message.
    """
    rag_manager = RAGManager()
    rag_manager.load_or_create_vectorstore()
    try:
        result = rag_manager.query(query)

        if state is not None:
            tool_output = {
                "input": {"query": query},
                "output": result,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            state.setdefault("tool_outputs", {}).setdefault("rag_search", []).append(tool_output)
            state.setdefault("tool_usage", []).append({
                "tool": "rag_search",
                "input": {"query": query},
                "output": result,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            state["last_tool"] = "rag_search"
            state["tool_result"] = result

        return result

    except Exception as e:
        error_message = f"RAG search error: {str(e)}"
        if state is not None:
            state.setdefault("tool_outputs", {}).setdefault("rag_search", []).append({
                "input": {"query": query},
                "output": None,
                "error": error_message,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            state["error"] = error_message
            state["last_tool"] = "rag_search"
        return error_message


@tool
def create_support_ticket(
    issue_description: str,
    priority: str = "normal",
    category: str = "general",
    state: AgentState = None
) -> str:
    """
    create a support ticket for customer issues.

    Args:
        issue_description (str): Description of the issue.
        priority (str): Ticket priority. Options: low, normal, high, urgent.
        category (str): Issue category. Options: billing, technical, account, general, refund.
        state (AgentState, optional): The agent's current state.

    Returns:
        str: Ticket confirmation message.
    """
    ticket_id = f"TICK-{str(uuid.uuid4())[:8].upper()}"

    if priority.lower() not in ["low", "normal", "high", "urgent"]:
        priority = "normal"
    if category.lower() not in ["billing", "technical", "account", "general", "refund"]:
        category = "general"

    ticket_data = {
        "ticket_id": ticket_id,
        "issue_description": issue_description
    }
    print(f"[SYSTEM] Ticket created in database: {ticket_data}")

    response = f"""
    ✅ Support Ticket Created Successfully!
    Ticket ID: {ticket_id}
    Issue Description: {issue_description}
    Expected Response Time:
    - Low: 24-48 hrs | Normal: 12-24 hrs | High: 4-8 hrs | Urgent: 1-2 hrs
    """

    if state is not None:
        state.setdefault("tool_outputs", {}).setdefault("create_support_ticket", []).append({
            "input": {
                "issue_description": issue_description,
                "priority": priority,
                "category": category
            },
            "output": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        state.setdefault("tool_usage", []).append({
            "tool": "create_support_ticket",
            "input": ticket_data,
            "output": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        state["last_tool"] = "create_support_ticket"
        state["tool_result"] = response

    return response


@tool
def schedule_support_call(
    
    issue_summary: str,
    call_type: str = "general",
    state: AgentState = None
) -> str:
    """
    schedule a support call with the customer.

    Args:
        preferred_date (str): Preferred call date (YYYY-MM-DD).
        preferred_time (str): Preferred call time (HH:MM).
        issue_summary (str): Brief issue summary.
        call_type (str): Type of call (technical, billing, consultation, general).
        state (AgentState, optional): The agent's current state.

    Returns:
        str: Appointment confirmation or error message.
    """

    preferred_date = "2025-05-26"
    preferred_time= "14:30"
    appointment_id = f"APT-{str(uuid.uuid4())[:8].upper()}"
    if call_type.lower() not in ["technical", "billing", "consultation", "general"]:
        call_type = "general"

    try:
        appointment_date = datetime.strptime(preferred_date, "%Y-%m-%d")
        appointment_time = datetime.strptime(preferred_time, "%H:%M").time()
        full_dt = datetime.combine(appointment_date.date(), appointment_time)

        
        if appointment_time.hour < 9 or appointment_time.hour >= 17:
            return "❌ Error: Time must be between 9:00 AM and 5:00 PM."
        if appointment_date.weekday() >= 5:
            return "❌ Error: Appointments only on weekdays."

    except ValueError:
        return "❌ Error: Date format (YYYY-MM-DD) and time format (HH:MM) required."

    formatted_date = appointment_date.strftime("%A, %B %d, %Y")
    formatted_time = appointment_time.strftime("%I:%M %p")

    appointment_data = {
        "appointment_id": appointment_id,
        "date": preferred_date,
        "time": preferred_time,
        "formatted_datetime": f"{formatted_date} at {formatted_time}",
        "issue_summary": issue_summary,
        "call_type": call_type.lower(),
        "status": "scheduled",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    print(f"[SYSTEM] Appointment scheduled in system: {appointment_data}")

    response = f"""
    📞 Support Call Scheduled Successfully!
    Appointment ID: {appointment_id}
    Scheduled: {formatted_date} at {formatted_time}
    Call Type: {call_type.title()}
    Issue: {issue_summary}
    Note: Calls occur only during business hours (9 AM - 5 PM, Mon-Fri).
    """

    if state is not None:
        state.setdefault("tool_outputs", {}).setdefault("schedule_support_call", []).append({
            "input": {
                "preferred_date": preferred_date,
                "preferred_time": preferred_time,
                "issue_summary": issue_summary,
                "call_type": call_type
            },
            "output": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        state.setdefault("tool_usage", []).append({
            "tool": "schedule_support_call",
            "input": appointment_data,
            "output": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        state["last_tool"] = "schedule_support_call"
        state["tool_result"] = response

    return response
