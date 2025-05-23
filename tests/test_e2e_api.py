from fastapi.testclient import TestClient
from api import app  # Update this import if your app is in a different location
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

# List of test messages
testcases = [
    { "message": "What are the fees of the Maquininha Smart?", "user_id": "client789" },
    { "message": "Are there any fees for creating boletos?", "user_id": "client789" },
    { "message": "Can I generate boletos on my mobile device?", "user_id": "client789" },
    { "message": "Quando foi o último jogo do Palmeiras?", "user_id": "client789" }

]

# Parameterized style to test each case
@pytest.mark.parametrize("testcase", testcases)
def test_chat_endpoint_multiple(testcase):
    response = client.post("/chat", json=testcase)
    assert response.status_code == 200
    # json_data = response.json()
    # assert "response" in json_data
    # assert isinstance(json_data["response"], str)

testcases_error  = [
    { "message": "", "user_id": "client789" },
    { "message": "Are there any fees for creating boletos?", "user_id": "" },
    { "message": "12345", "user_id": "client789" },
    { "message": "", "user_id": "" }
]

@pytest.mark.parametrize("testcase", testcases_error)
def test_chat_endpoint_error(testcase):
    response = client.post("/chat", json=testcase)
    assert response.status_code == 422
    # json_data = response.json()
    # assert "response" in json_data
    # assert isinstance(json_data["response"], str)

