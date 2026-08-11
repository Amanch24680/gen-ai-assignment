from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest

from app.api.dependencies import get_rag_service
from app.generation.exceptions import (
    GenerationError,
    OllamaConnectionError,
)
from app.main import app
from app.rag.service import RAGService
from app.retrieval.exceptions import EmptyQueryError, InvalidRetrievalConfigError
from app.schemas.query import Citation, RAGQueryResponse

client = TestClient(app)


def test_health_check_endpoint():
    """Verify GET /api/v1/health status endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "cost-efficient-rag"}


def test_query_endpoint_success():
    """Test 6A: Successful POST /api/v1/query request using mocked RAGService."""
    mock_rag_service = MagicMock(spec=RAGService)
    expected_response = RAGQueryResponse(
        query="What is vector search?",
        answer="Vector search finds nearest neighbor embeddings.",
        citations=[
            Citation(doc_id="d1", chunk_id="c1", snippet="Vector search text", score=0.88)
        ],
        retrieved_chunk_count=1,
        has_relevant_context=True,
        latency_ms=45.2,
    )
    mock_rag_service.query.return_value = expected_response

    # Apply dependency override
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

    try:
        payload = {"query": "What is vector search?"}
        response = client.post("/api/v1/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is vector search?"
        assert data["answer"] == "Vector search finds nearest neighbor embeddings."
        assert data["retrieved_chunk_count"] == 1
        assert data["has_relevant_context"] is True
        assert len(data["citations"]) == 1
        assert data["citations"][0]["chunk_id"] == "c1"
    finally:
        app.dependency_overrides.clear()


def test_query_endpoint_forwards_retrieval_parameters():
    """Test 6C: Retrieval parameters (top_k, relevance_threshold, metadata_filter) are forwarded to RAGService."""
    mock_rag_service = MagicMock(spec=RAGService)
    mock_rag_service.query.return_value = RAGQueryResponse(
        query="Filtered Query",
        answer="Answer text",
        citations=[],
        retrieved_chunk_count=0,
        has_relevant_context=False,
        latency_ms=10.0,
    )

    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

    try:
        payload = {
            "query": "Filtered Query",
            "top_k": 8,
            "relevance_threshold": 0.60,
            "metadata_filter": {"file_type": "pdf"},
        }
        response = client.post("/api/v1/query", json=payload)

        assert response.status_code == 200
        mock_rag_service.query.assert_called_once_with(
            query="Filtered Query",
            top_k=8,
            relevance_threshold=0.60,
            metadata_filter={"file_type": "pdf"},
        )
    finally:
        app.dependency_overrides.clear()


def test_query_endpoint_error_mapping_400():
    """Test 6D: Empty query or invalid config maps to HTTP 400 Bad Request."""
    mock_rag_service = MagicMock(spec=RAGService)
    mock_rag_service.query.side_effect = EmptyQueryError("Query string cannot be empty.")

    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

    try:
        payload = {"query": ""}
        response = client.post("/api/v1/query", json=payload)

        assert response.status_code == 400
        assert "Query string cannot be empty" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_query_endpoint_error_mapping_503():
    """Test 6D: Connection error to Ollama maps to HTTP 503 Service Unavailable."""
    mock_rag_service = MagicMock(spec=RAGService)
    mock_rag_service.query.side_effect = OllamaConnectionError("Failed to connect to Ollama")

    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

    try:
        payload = {"query": "Valid query"}
        response = client.post("/api/v1/query", json=payload)

        assert response.status_code == 503
        assert "LLM Generation Service unavailable" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_query_endpoint_error_mapping_500():
    """Test 6D: Internal generation/retrieval error maps to HTTP 500 Internal Server Error."""
    mock_rag_service = MagicMock(spec=RAGService)
    mock_rag_service.query.side_effect = GenerationError("Internal LLM generation failure")

    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

    try:
        payload = {"query": "Valid query"}
        response = client.post("/api/v1/query", json=payload)

        assert response.status_code == 500
        assert "RAG processing error" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_api_architectural_boundary():
    """Test 6F: Architectural boundary check - router and endpoints do not import low-level client tools directly."""
    import app.api.router as router_module
    module_dict = router_module.__dict__

    assert "QdrantClient" not in module_dict
    assert "SentenceTransformer" not in module_dict
    assert "httpx" not in module_dict
