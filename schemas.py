from pydantic import BaseModel
from typing import Dict, Any, List


class ChatRequest(BaseModel):
    message: str
    user_id: str

class ChatResponse(BaseModel):
    response: str
    source_agent_response: str
    agent_workflow: List[Dict[str, Any]]
    conversation_active: bool = True
    needs_followup: bool = True
    error: str | None = None
