import pytest
from app.api.router import router
from app.embeddings.base import BaseEmbeddingService
from app.generation.base import BaseGenerator
from app.ingestion.base import BaseChunker, BaseDocumentParser
from app.logging.logger import BaseQueryLogger, QueryObservabilityLogger
from app.retrieval.base import BaseRetriever
from app.schemas.document import Document, DocumentChunk
from app.schemas.metrics import QueryExecutionMetrics
from app.schemas.query import Citation, RAGQueryRequest, RAGQueryResponse
from app.vector_store.base import BaseVectorStore


def test_schemas_instantiation():
    """Verify that all data schemas instantiate cleanly."""
    doc = Document(
        doc_id="doc_1",
        content="Sample text",
        file_type="pdf",
        metadata={"author": "Test"},
    )
    assert doc.doc_id == "doc_1"

    chunk = DocumentChunk(
        chunk_id="chunk_1",
        doc_id="doc_1",
        text="Sample chunk text",
        chunk_index=0,
        metadata={},
    )
    assert chunk.chunk_id == "chunk_1"

    citation = Citation(
        doc_id="doc_1",
        chunk_id="chunk_1",
        snippet="Sample",
        score=0.95,
    )
    assert citation.score == 0.95

    request = RAGQueryRequest(query="What is RAG?")
    assert request.query == "What is RAG?"

    response = RAGQueryResponse(
        query="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation.",
        citations=[citation],
        retrieved_chunk_count=1,
        has_relevant_context=True,
        latency_ms=120.5,
    )
    assert response.retrieved_chunk_count == 1

    metrics = QueryExecutionMetrics(
        retrieval_latency_ms=20.0,
        generation_latency_ms=100.0,
        total_latency_ms=120.0,
        retrieved_chunk_count=1,
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
    )
    assert metrics.total_tokens == 70


def test_abstract_interfaces_cannot_be_instantiated():
    """Verify that abstract base classes enforce implementation of abstract methods."""
    with pytest.raises(TypeError):
        BaseDocumentParser()

    with pytest.raises(TypeError):
        BaseChunker()

    with pytest.raises(TypeError):
        BaseEmbeddingService()

    with pytest.raises(TypeError):
        BaseVectorStore()

    with pytest.raises(TypeError):
        BaseRetriever()

    with pytest.raises(TypeError):
        BaseGenerator()

    with pytest.raises(TypeError):
        BaseQueryLogger()


def test_router_configuration():
    """Verify FastAPI router prefix and health endpoint exist."""
    assert router.prefix == "/api/v1"
    route_paths = [route.path for route in router.routes]
    assert "/api/v1/health" in route_paths
    assert "/api/v1/query" in route_paths


def test_query_observability_logger():
    """Verify QueryObservabilityLogger logs metrics without raising errors."""
    logger_instance = QueryObservabilityLogger()
    metrics = QueryExecutionMetrics(
        retrieval_latency_ms=15.0,
        generation_latency_ms=85.0,
        total_latency_ms=100.0,
        retrieved_chunk_count=2,
        total_tokens=150,
    )
    # Logging should execute cleanly
    logger_instance.log_query_metrics("Test query", metrics)
