from unittest.mock import MagicMock
import pytest

from app.retrieval.base import BaseRetriever
from app.schemas.document import DocumentChunk
from evaluation.retrieval_evaluator import RetrievalEvaluator


def test_retrieval_evaluator_workflow_mocked():
    """Verify RetrievalEvaluator executes retrieval for dataset without calling generator."""
    mock_retriever = MagicMock(spec=BaseRetriever)

    # Return chunks for queries
    mock_retriever.retrieve.side_effect = lambda query, top_k=5, relevance_threshold=None: [
        DocumentChunk(
            chunk_id="60e906b285514b553b944db54bd9790e" if "low-cost" in query else "dummy_chunk",
            doc_id="doc_1",
            text="Text excerpt",
            chunk_index=0,
            score=0.9,
            metadata={}
        )
    ]

    evaluator = RetrievalEvaluator(retriever=mock_retriever)
    summary = evaluator.evaluate(top_k=5)

    # Verify counts
    assert summary.total_cases == 20
    assert summary.answerable_cases == 18
    assert summary.unanswerable_cases == 2

    # Verify retriever call count
    assert mock_retriever.retrieve.call_count == 20

    # Verify per-question results schema & metrics
    assert len(summary.per_question_results) == 20
    q001_res = next(r for r in summary.per_question_results if r.question_id == "q001")
    assert q001_res.recall_at_1 == 1.0
    assert q001_res.precision_at_1 == 1.0
    assert q001_res.hit_rate_at_1 == 1.0
    assert q001_res.ndcg_at_1 == 1.0
    assert q001_res.reciprocal_rank == 1.0
    assert q001_res.retrieval_latency_ms >= 0.0

    # Verify summary fields
    assert summary.hit_rate_at_1 >= 0.0
    assert summary.ndcg_at_1 >= 0.0
    assert summary.p50_retrieval_latency_ms >= 0.0
    assert summary.p95_retrieval_latency_ms >= 0.0

    # Verify unanswerable metrics
    assert summary.unanswerable_empty_retrieval_count >= 0
    assert summary.unanswerable_non_empty_retrieval_count >= 0
    assert "direct_fact" in summary.category_metrics
    assert "unanswerable" in summary.category_metrics


def test_retrieval_evaluator_unanswerable_empty_behavior():
    """Verify unanswerable cases record empty retrieval rate correctly."""
    mock_retriever = MagicMock(spec=BaseRetriever)
    mock_retriever.retrieve.return_value = []  # Returns empty results

    evaluator = RetrievalEvaluator(retriever=mock_retriever)
    summary = evaluator.evaluate(top_k=5)

    assert summary.unanswerable_cases == 2
    assert summary.unanswerable_empty_retrieval_count == 2
    assert summary.unanswerable_empty_retrieval_rate == 1.0
