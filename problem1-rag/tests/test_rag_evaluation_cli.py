import json
from unittest.mock import MagicMock
import pytest

from evaluation.cli import main
from evaluation.rag_evaluator import RAGEvaluator
from evaluation.schemas import RAGAnswerEvaluationSummary


def test_cli_rag_mode_execution_mocked(tmp_path, capsys):
    mock_evaluator = MagicMock(spec=RAGEvaluator)
    mock_summary = RAGAnswerEvaluationSummary(
        total_cases=20,
        answerable_cases=18,
        unanswerable_cases=2,
        successful_generations=20,
        failed_generations=0,
        mean_answer_f1=0.75,
        mean_context_support=0.82,
        mean_citation_coverage=0.90,
        unanswerable_safe_handling_rate=1.0,
        average_latency_ms=250.0,
        median_latency_ms=240.0,
        min_latency_ms=180.0,
        max_latency_ms=350.0,
        category_metrics={},
        per_question_results=[],
    )
    mock_evaluator.evaluate.return_value = mock_summary

    output_json = tmp_path / "rag_results.json"
    result = main(
        ["--mode", "rag", "--k", "5", "--limit", "5", "--output", str(output_json)],
        rag_evaluator=mock_evaluator,
    )

    assert result == 0
    mock_evaluator.evaluate.assert_called_once_with(top_k=5, relevance_threshold=None, limit=5)

    captured = capsys.readouterr()
    assert "RAG ANSWER EVALUATION" in captured.out
    assert "Answer F1: 0.7500" in captured.out
    assert "Average Latency: 250.00 ms" in captured.out

    # Verify JSON output file
    assert output_json.exists()
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_cases"] == 20
        assert data["mean_answer_f1"] == 0.75


def test_cli_invalid_limit_rejected():
    result = main(["--limit", "0"])
    assert result != 0
