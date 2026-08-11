from unittest.mock import MagicMock
import pytest

from app.rag.service import RAGService
from app.schemas.query import Citation, RAGQueryResponse
from evaluation.rag_evaluator import RAGEvaluator


def test_rag_evaluator_workflow_mocked():
    """Verify RAGEvaluator executes RAGService queries and calculates end-to-end answer metrics."""
    mock_rag_service = MagicMock(spec=RAGService)

    mock_rag_service.query.side_effect = lambda query, top_k=None, relevance_threshold=None: RAGQueryResponse(
        query=query,
        answer="The vector stores are pgvector, Qdrant, ChromaDB, LanceDB, FAISS, sqlite-vec." if "vector" in query else "Information not available.",
        citations=[
            Citation(doc_id="doc1", chunk_id="60e906b285514b553b944db54bd9790e", snippet="vector stores pgvector Qdrant ChromaDB", score=0.9)
        ] if "vector" in query else [],
        retrieved_chunk_count=1 if "vector" in query else 0,
        has_relevant_context=True if "vector" in query else False,
        latency_ms=120.0,
    )

    evaluator = RAGEvaluator(rag_service=mock_rag_service)
    summary = evaluator.evaluate(top_k=5)

    assert summary.total_cases == 20
    assert summary.answerable_cases == 18
    assert summary.unanswerable_cases == 2
    assert summary.successful_generations == 20
    assert summary.failed_generations == 0

    assert mock_rag_service.query.call_count == 20
    assert summary.mean_answer_f1 > 0.0
    assert summary.average_latency_ms == 120.0


def test_rag_evaluator_limit_option():
    """Verify RAGEvaluator respects limit argument for development runs."""
    mock_rag_service = MagicMock(spec=RAGService)
    mock_rag_service.query.return_value = RAGQueryResponse(
        query="q",
        answer="ans",
        citations=[],
        retrieved_chunk_count=0,
        has_relevant_context=False,
        latency_ms=50.0,
    )

    evaluator = RAGEvaluator(rag_service=mock_rag_service)
    summary = evaluator.evaluate(limit=3)

    assert summary.total_cases == 3
    assert mock_rag_service.query.call_count == 3


def test_rag_evaluator_handles_generation_failure():
    """Verify RAGEvaluator handles exception thrown by RAGService without crashing."""
    mock_rag_service = MagicMock(spec=RAGService)
    mock_rag_service.query.side_effect = RuntimeError("Ollama connection failed")

    evaluator = RAGEvaluator(rag_service=mock_rag_service)
    summary = evaluator.evaluate(limit=2)

    assert summary.total_cases == 2
    assert summary.failed_generations == 2
    assert summary.successful_generations == 0
    assert summary.per_question_results[0].success is False
    assert "Ollama connection failed" in summary.per_question_results[0].error_message
