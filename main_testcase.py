import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api import app
from schemas import ChatResponse

client = TestClient(app)

# ---------- Test Health Check ----------

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# ---------- Test Chat Endpoint ----------

@patch("main.invoke_graph")  # Replace 'main' with the name of your FastAPI file if not 'main.py'
def test_chat_endpoint_success(mock_invoke):
    mock_invoke.return_value = {
        "response": "Hello there!",
        "source_agent_response": "LLM_Agent",
        "agent_workflow": ["Step1", "Step2"],
        "conversation_active": True,
        "needs_followup": False,
        "error": None
    }

    payload = {
        "message": "Hello!",
        "user_id": "user123"
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["response"] == "Hello there!"
    assert data["source_agent_response"] == "LLM_Agent"
    assert isinstance(data["agent_workflow"], list)
    assert data["conversation_active"] is True
    assert data["needs_followup"] is False
    assert data["error"] is None


@patch("main.invoke_graph")
def test_chat_endpoint_invalid_response_format(mock_invoke):
    mock_invoke.return_value = "Not a dict"

    payload = {
        "message": "Test message",
        "user_id": "user123"
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 500
    assert "Invalid response format" in response.json()["detail"]


@patch("main.invoke_graph")
def test_chat_endpoint_raises_exception(mock_invoke):
    mock_invoke.side_effect = RuntimeError("Internal error")

    payload = {
        "message": "Another test",
        "user_id": "user456"
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 500
    assert "Internal error" in response.json()["detail"]
