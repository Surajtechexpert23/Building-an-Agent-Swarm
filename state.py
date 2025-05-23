from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage

class ToolCall(TypedDict):
    """Represents a single tool call."""
    tool: str
    input: Dict[str, Any]
    output: str
    timestamp: str

class ToolUsage(TypedDict):
    """Represents tool usage statistics."""
    tools_called: List[ToolCall]
    total_calls: int

class ToolOutput(TypedDict):
    """Represents a tool's output."""
    input: Dict[str, Any]
    output: str
    timestamp: str

class AgentState(TypedDict):
    """Represents the complete state of the agent during task execution."""
    # Core state
    input: str
    messages: List[BaseMessage]
    agent_outcome: Optional[AgentFinish]
    next: str
    
    # Tool state
    tool_outputs: Dict[str, List[ToolOutput]]  # Store tool outputs by tool name
    tool_usage: List[ToolCall]  # Track individual tool calls
    last_tool: Optional[str]  # Last tool used
    tool_result: Optional[Any]  # Current tool result
    
    # Workflow state
    workflow_history: List[Dict[str, Any]]  # Track the agent workflow
    personality_output: Optional[Dict[str, Any]]  # Personality agent output
    
    # Additional metadata
    error: Optional[str]  # Any error message
    is_complete: Optional[bool]  # Whether the task is complete


