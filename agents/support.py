from typing import Dict, Optional, List, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from state import AgentState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from tools import create_support_ticket, schedule_support_call
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0
)

def process_customer_data(user_input: str) -> Dict[str, Any]:
    """Process and validate customer request data."""
    # Determine intent from user input
    call_keywords = [
        "call", "schedule", "appointment", "meeting", "talk", 
        "discuss", "phone", "speak", "consultation", "demo",
        "training", "walkthrough", "setup"
    ]
    
    intent = "schedule_call" if any(keyword in user_input.lower() for keyword in call_keywords) else "create_ticket"
    
    # Load appropriate template based on intent
    json_file = "two.json" if intent == "schedule_call" else "one.json"
    try:
        with open(json_file, 'r') as file:
            customer_data = json.load(file)
            return {
                "intent": intent,
                "data": customer_data,
                "error": None
            }
    except Exception as e:
        return {
            "intent": intent,
            "data": None,
            "error": f"Error loading customer data: {str(e)}"
        }

def customer_support_agent(state: AgentState) -> AgentState:
    """Customer support agent that handles tickets and appointments."""
    # Initialize and update state
    state["current_agent"] = "support"
    if "agent_stack" not in state:
        state["agent_stack"] = []
    state["agent_stack"].append("support")
    
    # Initialize tool tracking state
    if "tool_outputs" not in state:
        state["tool_outputs"] = {}
    if "workflow_history" not in state:
        state["workflow_history"] = []
    if "support_context" not in state:
        state["support_context"] = {
            "current_ticket": None,
            "current_appointment": None,
            "interaction_history": []
        }
        
    # Track support agent activation in workflow
    state["workflow_history"].append({
        "agent_name": "support",
        "action": "handle_request",
        "input": state["input"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Process customer data
    process_result = process_customer_data(state["input"])
    
    if process_result["error"]:
        state["error"] = process_result["error"]
        state["messages"].append(AIMessage(content=
            f"I apologize, but I encountered an issue: {process_result['error']}. "
            "Could you please provide your request details again?"
        ))
        return state

    # Set up tools
    tools = [create_support_ticket, schedule_support_call]

    # Create agent prompt
    support_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a professional customer support agent for InfinitePay.

        AVAILABLE TOOLS:
        1. create_support_ticket: Create support tickets
           Required: issue description
           Optional: priority (low/normal/high/urgent), category
           
        2. schedule_support_call: Schedule support calls
           Required: issue summary, preferred date (YYYY-MM-DD), preferred time (HH:MM) in format in string only
           
        BUSINESS RULES:
        - Support calls available Monday-Friday, 9 AM - 5 PM only
        - Verify all required fields are present before scheduling
        - Always validate date and time formats
        
        GUIDELINES:
        1. Be professional and empathetic
        2. Gather all required information
        3. Confirm understanding before taking action
        4. Provide clear next steps
        5. Follow up after actions
         6. date and time formats: YYYY-MM-DD for date, HH:MM for time and str format
        
        Current Customer Data: {process_result['data']}
        Current Intent: {process_result['intent']}"""),
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
        # Prepare input based on intent
        if process_result["intent"] == "create_ticket":
            tool_input = """
                Create a support ticket with:
                - issue_description: {data}
                - priority: normal
                - category: general
            """.format(data=process_result["data"]["request_data"]["issue_description"])
        else:
            # Validate required fields for support call
            request_data = process_result["data"]["request_data"]
            missing_fields = []
            
            if not request_data.get("preferred_time"):
                missing_fields.append("preferred time")
            if not request_data.get("issue_summary"):
                missing_fields.append("issue summary")
            if not request_data.get("preferred_date"):
                missing_fields.append("preferred date")
                
            if missing_fields:
                state["error"] = f"Missing required fields: {', '.join(missing_fields)}"
                state["messages"].append(AIMessage(content=
                    "I need some additional information to schedule your support call. "
                    f"Please provide: {', '.join(missing_fields)}."
                ))
                state["needs_followup"] = True
                return state
                
            # Format the input as a structured command
            tool_input = (
                f'schedule_support_call("{request_data["preferred_date"]}", '
                f'"{request_data["preferred_time"]}", '
                f'"{request_data["issue_summary"]}", '
                f'"general")'
            )

        # Execute agent
        response = agent_executor.invoke({
            "input": tool_input,
            "chat_history": [system_message for system_message in state["messages"] 
                           if isinstance(system_message, SystemMessage)]
        })
        
        # Track tool usage
        state["workflow_history"].append({
            "agent_name": "support",
            "tool_calls": state.get("tool_usage", [])
        })
        
        # Format response
        response_content = response["output"]
        if not any(phrase in response_content.lower() for phrase in [
            "anything else", "other questions", "can i help",
            "need clarification", "is there anything"
        ]):
            response_content += "\n\nIs there anything else I can help you with?"
        
        # Update state
        state["messages"].append(AIMessage(content=response_content))
        state["agent_outcome"] = response
        state["error"] = None
        state["needs_followup"] = True
        state["is_complete"] = True
        
    except Exception as e:
        error_message = f"Error in customer support agent: {str(e)}"
        print(error_message)
        state["error"] = error_message
        state["messages"].append(AIMessage(content=
            "I apologize, but I need some additional information. "
            "Could you please provide:\n"
            "1. A detailed description of your issue\n"
            "2. Your preferred priority level (if applicable)\n"
            "3. The best time to contact you (if you'd like a call)"
        ))
        
        # Update error state
        state["needs_followup"] = True
        state["workflow_history"].append({
            "agent_name": "support",
            "action": "error_handling",
            "error": error_message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Clear any partial tool outputs
        if "tool_outputs" in state:
            state["tool_outputs"].pop("create_support_ticket", None)
            state["tool_outputs"].pop("schedule_support_call", None)
            
        # Clear support context on error
        if "support_context" in state:
            state["support_context"] = {
                "current_ticket": None,
                "current_appointment": None,
                "interaction_history": state["support_context"].get("interaction_history", [])
            }
    
    return state
