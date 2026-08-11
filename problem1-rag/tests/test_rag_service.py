from unittest.mock import MagicMock
import pytest

from app.generation.base import BaseGenerator
from app.generation.exceptions import GenerationError
from app.generation.service import NO_CONTEXT_RESPONSE_TEXT
from app.logging.logger import BaseQueryLogger
from app.rag.service import RAGService
from app.retrieval.base import BaseRetriever
from app.retrieval.exceptions import RetrievalError
from app.schemas.document import DocumentChunk
from app.schemas.metrics import QueryExecutionMetrics
from app.schemas.query import Citation, RAGQueryResponse


def test_rag_service_retrieves_then_generates():
    """Test 1: Verify RAGService calls retriever first, then passes exact query and chunks to generator."""
    mock_retriever = MagicMock(spec=BaseRetriever)
    mock_generator = MagicMock(spec=BaseGenerator)

    chunk_1 = DocumentChunk(
        chunk_id="c1", doc_id="d1", text="Chunk 1 text", chunk_index=0, metadata={}
    )
    chunk_2 = DocumentChunk(
        chunk_id="c2", doc_id="d1", text="Chunk 2 text", chunk_index=1, metadata={}
    )

    mock_retriever.retrieve.return_value = [chunk_1, chunk_2]

    expected_response = RAGQueryResponse(
        query="What is RAG?",
        answer="RAG is Retrieval-Augmented Generation.",
        citations=[
            Citation(doc_id="d1", chunk_id="c1", snippet="Chunk 1 text", score=0.90)
        ],
        retrieved_chunk_count=2,
        has_relevant_context=True,
        latency_ms=120.0,
    )
    mock_generator.generate.return_value = expected_response

    service = RAGService(retriever=mock_retriever, generator=mock_generator)
    result = service.query("What is RAG?")

    # 1. Verify retriever was called with query
    mock_retriever.retrieve.assert_called_once_with(
        query="What is RAG?",
        top_k=None,
        relevance_threshold=None,
        metadata_filter=None,
    )

    # 2. Verify generator was called with same query and returned chunks
    mock_generator.generate.assert_called_once_with(
        query="What is RAG?",
        context_chunks=[chunk_1, chunk_2],
    )

    # 3. Verify returned result contains expected answer and metrics
    assert result.answer == expected_response.answer
    assert result.metrics is not None
    assert result.metrics.retrieved_chunk_count == 2


def test_rag_service_passes_retrieval_parameters():
    """Test 2: Verify top_k, relevance_threshold, and metadata_filter are correctly passed to retriever."""
    mock_retriever = MagicMock(spec=BaseRetriever)
    mock_generator = MagicMock(spec=BaseGenerator)

    mock_retriever.retrieve.return_value = []
    mock_generator.generate.return_value = RAGQueryResponse(
        query="Custom params query",
        answer=NO_CONTEXT_RESPONSE_TEXT,
        citations=[],
        retrieved_chunk_count=0,
        has_relevant_context=False,
        latency_ms=1.0,
    )

    service = RAGService(retriever=mock_retriever, generator=mock_generator)

    filter_dict = {"file_type": "pdf"}
    service.query(
        "Custom params query",
        top_k=10,
        relevance_threshold=0.65,
        metadata_filter=filter_dict,
    )

    mock_retriever.retrieve.assert_called_once_with(
        query="Custom params query",
        top_k=10,
        relevance_threshold=0.65,
        metadata_filter=filter_dict,
    )


def test_rag_service_empty_retrieval_passes_empty_context_to_generator():
    """Test 3: Empty retrieval returns [] to generator, which returns no-context response unchanged."""
    mock_retriever = MagicMock(spec=BaseRetriever)
    mock_generator = MagicMock(spec=BaseGenerator)

    mock_retriever.retrieve.return_value = []
    no_context_resp = RAGQueryResponse(
        query="Unknown Query",
        answer=NO_CONTEXT_RESPONSE_TEXT,
        citations=[],
        retrieved_chunk_count=0,
        has_relevant_context=False,
        latency_ms=0.5,
    )
    mock_generator.generate.return_value = no_context_resp

    service = RAGService(retriever=mock_retriever, generator=mock_generator)
    result = service.query("Unknown Query")

    mock_retriever.retrieve.assert_called_once()
    mock_generator.generate.assert_called_once_with(
        query="Unknown Query",
        context_chunks=[],
    )
    assert result.answer == NO_CONTEXT_RESPONSE_TEXT
    assert result.has_relevant_context is False


def test_rag_service_retrieval_error_propagates():
    """Test 4: Verify exceptions from retriever propagate without being swallowed."""
    mock_retriever = MagicMock(spec=BaseRetriever)
    mock_generator = MagicMock(spec=BaseGenerator)

    mock_retriever.retrieve.side_effect = RetrievalError("Retrieval system error")

    service = RAGService(retriever=mock_retriever, generator=mock_generator)

    with pytest.raises(RetrievalError) as exc_info:
        service.query("Query")

    assert "Retrieval system error" in str(exc_info.value)
    mock_generator.generate.assert_not_called()


def test_rag_service_generation_error_propagates():
    """Test 5: Verify exceptions from generator propagate without being swallowed."""
    mock_retriever = MagicMock(spec=BaseRetriever)
    mock_generator = MagicMock(spec=BaseGenerator)

    mock_retriever.retrieve.return_value = [
        DocumentChunk(chunk_id="c1", doc_id="d1", text="Text", chunk_index=0, metadata={})
    ]
    mock_generator.generate.side_effect = GenerationError("Ollama service failure")

    service = RAGService(retriever=mock_retriever, generator=mock_generator)

    with pytest.raises(GenerationError) as exc_info:
        service.query("Query")

    assert "Ollama service failure" in str(exc_info.value)


def test_rag_service_logs_and_attaches_metrics():
    """Test 6: Verify QueryObservabilityLogger is called and response.metrics is populated."""
    mock_retriever = MagicMock(spec=BaseRetriever)
    mock_generator = MagicMock(spec=BaseGenerator)
    mock_logger = MagicMock(spec=BaseQueryLogger)

    chunk = DocumentChunk(chunk_id="c1", doc_id="d1", text="Sample", chunk_index=0, metadata={})
    mock_retriever.retrieve.return_value = [chunk]

    mock_generator.generate.return_value = RAGQueryResponse(
        query="Test query",
        answer="Test answer",
        citations=[],
        retrieved_chunk_count=1,
        has_relevant_context=True,
        latency_ms=50.0,
        prompt_tokens=45,
        completion_tokens=15,
        total_tokens=60,
    )

    service = RAGService(retriever=mock_retriever, generator=mock_generator, query_logger=mock_logger)
    result = service.query("Test query")

    # Verify logger was called once
    mock_logger.log_query_metrics.assert_called_once()
    logged_query, logged_metrics = mock_logger.log_query_metrics.call_args[0]
    assert logged_query == "Test query"
    assert isinstance(logged_metrics, QueryExecutionMetrics)
    assert logged_metrics.retrieved_chunk_count == 1
    assert logged_metrics.prompt_tokens == 45
    assert logged_metrics.completion_tokens == 15
    assert logged_metrics.total_tokens == 60
    assert logged_metrics.retrieval_latency_ms >= 0.0
    assert logged_metrics.generation_latency_ms >= 0.0
    assert logged_metrics.total_latency_ms >= logged_metrics.retrieval_latency_ms

    # Verify attached metrics on response
    assert result.metrics == logged_metrics


def test_rag_service_does_not_directly_call_qdrant_or_ollama():
    """Test 7: Architectural boundary check - RAGService only relies on injected abstractions."""
    mock_retriever = MagicMock(spec=BaseRetriever)
    mock_generator = MagicMock(spec=BaseGenerator)

    service = RAGService(retriever=mock_retriever, generator=mock_generator)

    assert hasattr(service, "retriever")
    assert hasattr(service, "generator")
    assert service.retriever is mock_retriever
    assert service.generator is mock_generator

    # Check module imports of service.py to confirm no Qdrant or HTTP clients imported
    import app.rag.service as service_module

    module_dict = service_module.__dict__
    assert "QdrantClient" not in module_dict
    assert "QdrantVectorStore" not in module_dict
    assert "SentenceTransformer" not in module_dict
    assert "httpx" not in module_dict
    assert "FastAPI" not in module_dict
