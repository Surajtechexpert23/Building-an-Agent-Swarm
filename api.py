from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List
from graph import invoke_graph
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Agent Swarm API",
    description="API for interacting with the agent swarm implementation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="The message to process. Must contain either 'dev' or '123456' and cannot be only numbers")
    user_id: str = Field(..., min_length=1, max_length=100, description="Unique identifier for the user")

    @validator('message')
    def validate_message(cls, v):
        if v.strip() == "":
            raise ValueError("Message cannot be empty or just whitespace")
        
        # Check if message contains only numbers
        if v.strip().replace(" ", "").isdigit():
            raise ValueError("Message cannot contain only numbers")
        
      
            
        return v.strip()

    @validator('user_id')
    def validate_user_id(cls, v):
        if not v.strip():
            raise ValueError("User ID cannot be empty or just whitespace")
        return v.strip()

class ChatResponse(BaseModel):
    response: str = Field(..., description="The main response from the agent system")
    source_agent_response: str = Field(..., description="Response from the source agent")
    agent_workflow: List[Dict[str, Any]] = Field(default_factory=list, description="List of agent workflow steps")
    conversation_active: bool = Field(default=True, description="Whether the conversation is still active")
    needs_followup: bool = Field(default=True, description="Whether the conversation needs follow-up")
    error: str | None = Field(default=None, description="Error message if any")

    @validator('response', 'source_agent_response')
    def validate_responses(cls, v):
        if not v or not v.strip():
            raise ValueError("Response cannot be empty")
        return v

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    """Process a chat message through the agent system.
    
    Args:
        request: ChatRequest containing message and user_id
        
    Returns:
        Dict containing the response from the agent system
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        # Process message through agent workflow
        result = invoke_graph(request.message)
        
        # Validate and format response
        if not isinstance(result, dict):
            raise ValueError("Invalid response format from agent system")
        
        if not result.get("response"):
            raise ValueError("Empty response from agent system")
            
        # Return formatted response
        return ChatResponse(
            response=result.get("response", ""),
            source_agent_response=result.get("source_agent_response", ""),
            agent_workflow=result.get("agent_workflow", []),
            conversation_active=result.get("conversation_active", True),
            needs_followup=result.get("needs_followup", True),
            error=result.get("error")
        )
        
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)