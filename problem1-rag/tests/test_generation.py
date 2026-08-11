import json
from unittest.mock import MagicMock
import httpx
import pytest

from app.config.settings import get_settings
from app.generation.exceptions import (
    EmptyGenerationError,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from app.generation.service import NO_CONTEXT_RESPONSE_TEXT, OllamaGenerator
from app.schemas.document import DocumentChunk


def test_generator_instantiation_from_settings():
    """Test A & B: Generator instantiates correctly and reads defaults from settings."""
    settings = get_settings()
    generator = OllamaGenerator()
    assert generator.ollama_base_url == settings.ollama_base_url.rstrip("/")
    assert generator.generator_model == settings.generator_model


def test_generator_empty_context_does_not_call_ollama():
    """Test H & I: Empty context returns no-context response without making HTTP requests."""
    mock_client = MagicMock(spec=httpx.Client)
    generator = OllamaGenerator(http_client=mock_client)

    response = generator.generate("What is RAG?", context_chunks=[])

    # HTTP client must NOT be called
    mock_client.post.assert_not_called()
    assert response.has_relevant_context is False
    assert response.retrieved_chunk_count == 0
    assert response.answer == NO_CONTEXT_RESPONSE_TEXT
    assert response.citations == []


def test_generator_request_payload_and_endpoint():
    """Test C, D, E, F, G, J, P: Successful Ollama request formats payload, prompt, and parses response."""
    mock_client = MagicMock(spec=httpx.Client)

    # Mock Ollama HTTP 200 response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "RAG stands for Retrieval-Augmented Generation."}
    mock_client.post.return_value = mock_response

    generator = OllamaGenerator(
        ollama_base_url="http://localhost:11434",
        generator_model="gemma:2b",
        http_client=mock_client,
    )

    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            text="Retrieval-Augmented Generation (RAG) grounds LLM responses.",
            chunk_index=0,
            metadata={"filename": "rag.pdf", "page_number": 1, "score": 0.90},
            score=0.90,
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            doc_id="doc_1",
            text="It combines vector retrieval with local LLMs.",
            chunk_index=1,
            metadata={"filename": "rag.pdf", "page_number": 2, "score": 0.85},
            score=0.85,
        ),
    ]

    response = generator.generate("What is RAG?", context_chunks=chunks)

    # Exactly one HTTP request made
    assert mock_client.post.call_count == 1
    call_args = mock_client.post.call_args
    endpoint = call_args[0][0]
    json_payload = call_args[1]["json"]

    assert endpoint == "http://localhost:11434/api/generate"
    assert json_payload["model"] == "gemma:2b"
    assert json_payload["stream"] is False

    # Check prompt contents
    prompt = json_payload["prompt"]
    assert "What is RAG?" in prompt
    assert "Retrieval-Augmented Generation (RAG) grounds LLM responses." in prompt
    assert "It combines vector retrieval with local LLMs." in prompt
    assert "[Source 1]" in prompt
    assert "Filename: rag.pdf" in prompt
    assert "Page: 1" in prompt
    assert "[Source 2]" in prompt
    assert "Page: 2" in prompt

    # Response schema checks
    assert response.query == "What is RAG?"
    assert response.answer == "RAG stands for Retrieval-Augmented Generation."
    assert response.has_relevant_context is True
    assert response.retrieved_chunk_count == 2
    assert len(response.citations) == 2
    assert response.citations[0].chunk_id == "chunk_1"
    assert response.citations[0].score == 0.90


def test_generator_http_error_handling():
    """Test K: HTTP status code != 200 raises OllamaResponseError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_client.post.return_value = mock_response

    generator = OllamaGenerator(http_client=mock_client)
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", text="Sample text", chunk_index=0, metadata={}
    )

    with pytest.raises(OllamaResponseError) as exc_info:
        generator.generate("Query", context_chunks=[chunk])
    assert "500" in str(exc_info.value)


def test_generator_connection_failure_handling():
    """Test L: Connection error raises OllamaConnectionError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")

    generator = OllamaGenerator(http_client=mock_client)
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", text="Sample text", chunk_index=0, metadata={}
    )

    with pytest.raises(OllamaConnectionError):
        generator.generate("Query", context_chunks=[chunk])


def test_generator_timeout_handling():
    """Test M: Timeout exception raises OllamaTimeoutError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.TimeoutException("Timed out")

    generator = OllamaGenerator(http_client=mock_client)
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", text="Sample text", chunk_index=0, metadata={}
    )

    with pytest.raises(OllamaTimeoutError):
        generator.generate("Query", context_chunks=[chunk])


def test_generator_malformed_response_handling():
    """Test N: Missing or malformed JSON payload raises OllamaResponseError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"invalid_field": "no response key"}
    mock_client.post.return_value = mock_response

    generator = OllamaGenerator(http_client=mock_client)
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", text="Sample text", chunk_index=0, metadata={}
    )

    with pytest.raises(OllamaResponseError) as exc_info:
        generator.generate("Query", context_chunks=[chunk])
    assert "missing 'response' field" in str(exc_info.value)


def test_generator_empty_response_handling():
    """Test O: Empty/whitespace answer response raises EmptyGenerationError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "   "}
    mock_client.post.return_value = mock_response

    generator = OllamaGenerator(http_client=mock_client)
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", text="Sample text", chunk_index=0, metadata={}
    )

    with pytest.raises(EmptyGenerationError):
        generator.generate("Query", context_chunks=[chunk])


def test_prompt_grounding_instructions():
    """Test 14: Verify prompt contains strict anti-hallucination grounding instructions."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Grounded answer"}
    mock_client.post.return_value = mock_response

    generator = OllamaGenerator(http_client=mock_client)
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", text="Target Fact 42", chunk_index=0, metadata={}
    )

    generator.generate("Target Query", context_chunks=[chunk])

    prompt = mock_client.post.call_args[1]["json"]["prompt"]
    assert "Answer the query based ONLY on the context below" in prompt
    assert "Do not use external knowledge or invent facts" in prompt
    assert "Target Fact 42" in prompt
    assert "Target Query" in prompt


# --- Real Ollama Live Integration Test ---

@pytest.mark.integration
def test_live_ollama_generator_integration():
    """
    Test 13: Live integration test against local Ollama service (if available).
    Verifies Python -> Ollama HTTP API -> gemma:2b -> RAGQueryResponse pipeline.
    Skipped automatically if Ollama service or model is unreachable.
    """
    settings = get_settings()
    base_url = settings.ollama_base_url.rstrip("/")

    # Check if Ollama is running
    try:
        res = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        if res.status_code != 200:
            pytest.skip("Local Ollama service is unreachable.")
        models_data = res.json().get("models", [])
        model_names = [m.get("name") for m in models_data]
        if not any(settings.generator_model in m for m in model_names):
            pytest.skip(f"Model '{settings.generator_model}' not found in Ollama tags.")
    except Exception as exc:
        pytest.skip(f"Local Ollama service unavailable: {exc}")

    # Live generation execution
    generator = OllamaGenerator()
    chunk = DocumentChunk(
        chunk_id="live_chunk_01",
        doc_id="doc_live",
        text="The capital of France is Paris.",
        chunk_index=0,
        metadata={"filename": "geography.md", "source": "/docs/geography.md"},
        score=0.95,
    )

    response = generator.generate("What is the capital of France?", context_chunks=[chunk])

    assert response.has_relevant_context is True
    assert response.retrieved_chunk_count == 1
    assert "Paris" in response.answer or "paris" in response.answer.lower()
    assert response.latency_ms > 0.0
    assert len(response.citations) == 1
