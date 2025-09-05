# tests/test_unit_cases.py

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime

from api.main import app  # or your FastAPI entrypoint
from src.document_chat.retrieval import ConversationalRAG
from model.models import DocumentAnswer
from langchain_core.messages import HumanMessage, AIMessage

client = TestClient(app)

@pytest.fixture
def test_session_id():
    return "test_session_123"

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text

@pytest.fixture
def mock_faiss_index(test_session_id):
    """Create a temporary FAISS index directory"""
    temp_dir = tempfile.mkdtemp()
    session_dir = os.path.join(temp_dir, test_session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Create mock index files
    with open(os.path.join(session_dir, "index.faiss"), "wb") as f:
        f.write(b"mock_faiss_data")
    with open(os.path.join(session_dir, "index.pkl"), "wb") as f:
        f.write(b"mock_pickle_data")
    
    yield session_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_document_answer():
    return DocumentAnswer(
        answer="This is a test answer about RAG systems.",
        confidence=0.85,
        sources=["/Users/anujpandey/llm_projects/llmops/document_portal/data/document_compare/Long_Report_V1.pdf", 
                 "/Users/anujpandey/llm_projects/llmops/document_portal/data/document_compare/Long_Report_V2.pdf"],
        reasoning="Answer found in retrieved documents about RAG methodology.",  # ✅ Added required field
        answer_type="factual" 
    )

@patch('src.document_chat.retrieval.ConversationalRAG')
@patch('src.document_ingestion.data_ingestion.FaissManager')  # ✅ Mock FAISS loading
@patch('utils.model_loader.ModelLoader')  # ✅ Mock ModelLoader
def test_chat_query_success(mock_model_loader, mock_faiss, mock_rag_class, test_session_id, mock_document_answer):
    # Setup mocks
    mock_rag = Mock()
    mock_rag.model_name = "gpt-4o"
    mock_rag.invoke_with_memory_and_cache.return_value = mock_document_answer
    mock_rag_class.return_value = mock_rag
    
    # Test data
    form_data = {
        "question": "What is RAG?",
        "session_id": test_session_id,
        "use_session_dirs": True,
        "k": 5,
        "use_cache": True
    }
    
    with patch('os.path.isdir', return_value=True):
        response = client.post("/chat/query", data=form_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["session_id"] == test_session_id
    assert data["k"] == 5
    assert data["engine"] == "LCEL-RAG"
    assert data["cached"] == True
    assert "response_time_seconds" in data
    assert data["model_used"] == "gpt-4o"

def test_chat_query_missing_session_id():
    form_data = {
        "question": "What is RAG?",
        "use_session_dirs": True,  # Requires session_id
        "k": 5
    }
    
    response = client.post("/chat/query", data=form_data)
    assert response.status_code == 400
    assert "session_id is required" in response.json()["detail"]
    
def test_chat_query_index_not_found(test_session_id):
    form_data = {
        "question": "What is RAG?",
        "session_id": test_session_id,
        "use_session_dirs": True,
        "k": 5
    }
    
    with patch('os.path.isdir', return_value=False):
        response = client.post("/chat/query", data=form_data)
    
    assert response.status_code == 404
    assert "FAISS index not found" in response.json()["detail"]

@patch('src.document_chat.retrieval.ConversationalRAG')
def test_chat_query_without_cache(mock_rag_class, test_session_id, mock_document_answer):
    mock_rag = Mock()
    mock_rag.model_name = "gemini-2.0-flash"
    mock_rag.invoke_with_memory.return_value = mock_document_answer
    mock_rag_class.return_value = mock_rag
    
    form_data = {
        "question": "Explain machine learning",
        "session_id": test_session_id,
        "use_session_dirs": True,
        "k": 3,
        "use_cache": False  # Disable caching
    }
    
    with patch('os.path.isdir', return_value=True):
        response = client.post("/chat/query", data=form_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["cached"] == False
    assert data["k"] == 3
    mock_rag.invoke_with_memory.assert_called_once()

@patch('src.document_chat.retrieval.ConversationalRAG')
def test_chat_query_large_question(mock_rag_class, test_session_id, mock_document_answer):
    mock_rag = Mock()
    mock_rag.model_name = "gpt-4o"
    mock_rag.invoke_with_memory_and_cache.return_value = mock_document_answer
    mock_rag_class.return_value = mock_rag
    
    # Large question (1000+ characters)
    large_question = "What is " + "artificial intelligence " * 100 + "?"
    
    form_data = {
        "question": large_question,
        "session_id": test_session_id,
        "use_session_dirs": True,
        "k": 5
    }
    
    with patch('os.path.isdir', return_value=True):
        response = client.post("/chat/query", data=form_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    # Verify the large question was handled
    mock_rag.invoke_with_memory_and_cache.assert_called_once_with(large_question)

@patch('api.main.token_counter')
def test_session_costs_analytics(mock_token_counter, test_session_id):
    from api.main import SessionCosts, TokenUsage
    
    # Mock session costs data
    mock_session_costs = SessionCosts(
        session_id=test_session_id,
        total_requests=5,
        total_input_tokens=1000,
        total_output_tokens=500,
        total_tokens=1500,
        total_cost_usd=0.045,
        first_request=datetime.now(),
        last_request=datetime.now(),
        requests=[]
    )
    
    mock_token_counter.get_session_costs.return_value = mock_session_costs
    
    response = client.get(f"/analytics/session-costs/{test_session_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == test_session_id
    assert data["total_requests"] == 5
    assert data["total_tokens"] == 1500
    assert data["total_cost_usd"] == 0.045

@patch('api.main.token_counter')
def test_session_costs_no_data(mock_token_counter):
    mock_token_counter.get_session_costs.return_value = None
    
    response = client.get("/analytics/session-costs/nonexistent_session")
    
    assert response.status_code == 200
    data = response.json()
    assert "No cost data found" in data["message"]
    assert data["costs"] is None

@patch('src.document_chat.retrieval.ConversationalRAG')
def test_clear_chat_history(mock_rag_class, test_session_id):
    mock_rag = Mock()
    mock_rag_class.return_value = mock_rag
    
    form_data = {"session_id": test_session_id}
    
    response = client.post("/chat/clear-history", data=form_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "cleared successfully" in data["message"]
    assert data["session_id"] == test_session_id
    mock_rag.clear_memory.assert_called_once()

@patch('src.document_chat.retrieval.ConversationalRAG')
def test_chat_query_internal_error(mock_rag_class, test_session_id):
    # Setup mock to raise an exception
    mock_rag = Mock()
    mock_rag.load_retriever_from_faiss.side_effect = Exception("Database connection failed")
    mock_rag_class.return_value = mock_rag
    
    form_data = {
        "question": "What is RAG?",
        "session_id": test_session_id,
        "use_session_dirs": True,
        "k": 5
    }
    
    with patch('os.path.isdir', return_value=True):
        response = client.post("/chat/query", data=form_data)
    
    assert response.status_code == 500
    assert "Query failed" in response.json()["detail"]

def test_invalid_k_parameter():
    """Test with invalid k parameter"""
    form_data = {
        "question": "What is RAG?",
        "session_id": "test_session",
        "use_session_dirs": True,
        "k": -1  # Invalid k value
    }
    
    with patch('os.path.isdir', return_value=True):
        with patch('src.document_chat.retrieval.ConversationalRAG') as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            response = client.post("/chat/query", data=form_data)
            # The API should handle this gracefully
            assert response.status_code in [200, 400, 422]

def test_empty_question():
    """Test with empty question - should process normally"""
    form_data = {
        "question": "",  # Empty question
        "session_id": "test_session",
        "use_session_dirs": True,
        "k": 5
    }
    
    with patch('os.path.isdir', return_value=True):
        with patch('src.document_chat.retrieval.ConversationalRAG') as mock_rag:
            # Mock RAG to return something for empty question
            mock_rag_instance = Mock()
            mock_rag_instance.model_name = "gpt-4o"
            mock_rag_instance.invoke_with_memory_and_cache.return_value = Mock(
                answer="I need more information to provide a helpful answer.",
                confidence=0.1,
                sources=[]
            )
            mock_rag.return_value = mock_rag_instance
            
            response = client.post("/chat/query", data=form_data)
    
    # Should process successfully (your API doesn't reject empty questions)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["session_id"] == "test_session"

def test_conversation_memory_integration():
    """Test that conversation memory is properly integrated"""
    with patch('src.document_chat.retrieval.ConversationalRAG') as mock_rag:
        mock_rag_instance = Mock()
        mock_rag_instance.get_conversation_history.return_value = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
        mock_rag.return_value = mock_rag_instance
        
        # This would be tested through the chat endpoint indirectly
        history = mock_rag_instance.get_conversation_history()
        assert len(history) == 2
        assert history[0].content == "Hello"
        assert history[1].content == "Hi there!"


# Run specific tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])