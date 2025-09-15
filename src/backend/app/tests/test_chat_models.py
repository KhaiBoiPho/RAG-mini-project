import pytest
from datetime import datetime
from src.backend.app.models.chat_models import Message, MessageRole, ChatRequest, ChatChoice, ChatResponse, ChatUsage, ErrorResponse

# ----------------------
# Test Message model
# ----------------------
def test_message_valid():
    msg = Message(role=MessageRole.USER, content="Hello world")
    assert msg.role == "user"
    assert msg.content == "Hello world"
    assert isinstance(msg.timestamp, datetime)

def test_message_whitespace_content():
    with pytest.raises(ValueError):
        Message(role=MessageRole.USER, content="   ")

# ----------------------
# Test ChatRequest model
# ----------------------
def test_chatrequest_valid():
    msg1 = Message(role=MessageRole.USER, content="Hi")
    request = ChatRequest(messages=[msg1], model="gpt-5-nano")
    assert request.model == "gpt-5-nano"
    assert request.messages[0].content == "Hi"

def test_chatrequest_last_message_not_user():
    msg1 = Message(role=MessageRole.ASSISTANT, content="Hi")
    with pytest.raises(ValueError):
        ChatRequest(messages=[msg1])

def test_chatrequest_invalid_model():
    msg1 = Message(role=MessageRole.USER, content="Hi")
    with pytest.raises(ValueError):
        ChatRequest(messages=[msg1], model="unknown-model")

# ----------------------
# Test ChatChoice and ChatResponse
# ----------------------
def test_chatchoice_and_response():
    msg1 = Message(role=MessageRole.ASSISTANT, content="Hello")
    choice = ChatChoice(index=0, message=msg1, finish_reason="stop")
    usage = ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    
    response = ChatResponse(
        conversation_id="conv123",
        model="gpt-5-nano",
        choices=[choice],
        usage=usage
    )
    
    assert response.conversation_id == "conv123"
    assert response.model == "gpt-5-nano"
    assert response.choices[0].message.content == "Hello"
    assert response.usage.total_tokens == 15

# ----------------------
# Test ErrorResponse
# ----------------------
def test_error_response_defaults():
    error = ErrorResponse(error="ValidationError", message="Invalid")
    assert error.error == "ValidationError"
    assert error.message == "Invalid"
    assert isinstance(error.timestamp, int)
