import json
from unittest.mock import MagicMock
import pytest

from evaluation.cli import build_parser, main
from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.schemas import RetrievalEvaluationSummary


def test_cli_parser_options():
    parser = build_parser()
    args = parser.parse_args(["--k", "3", "--output", "out.json"])
    assert args.k == 3
    assert args.output == "out.json"


def test_cli_execution_mocked(tmp_path, capsys):
    mock_evaluator = MagicMock(spec=RetrievalEvaluator)
    mock_summary = RetrievalEvaluationSummary(
        total_cases=20,
        answerable_cases=18,
        unanswerable_cases=2,
        recall_at_1=0.88,
        recall_at_3=0.95,
        recall_at_5=1.0,
        precision_at_1=0.88,
        precision_at_3=0.44,
        precision_at_5=0.26,
        mrr=0.92,
        unanswerable_empty_retrieval_count=2,
        unanswerable_non_empty_retrieval_count=0,
        unanswerable_empty_retrieval_rate=1.0,
        category_metrics={},
        per_question_results=[],
    )
    mock_evaluator.evaluate.return_value = mock_summary

    output_json = tmp_path / "results.json"
    result = main(["--k", "5", "--output", str(output_json)], evaluator=mock_evaluator)

    assert result == 0
    mock_evaluator.evaluate.assert_called_once_with(top_k=5, relevance_threshold=None)

    captured = capsys.readouterr()
    assert "RAG RETRIEVAL EVALUATION" in captured.out
    assert "Recall@1: 0.8800" in captured.out
    assert "MRR: 0.9200" in captured.out

    # Verify JSON output file created
    assert output_json.exists()
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_cases"] == 20
        assert data["mrr"] == 0.92


def test_cli_invalid_k_rejected():
    result = main(["--k", "0"])
    assert result != 0
