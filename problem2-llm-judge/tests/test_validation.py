from unittest.mock import MagicMock
import pytest

from app.evaluator import Evaluator
from app.judge import JudgeClient
from app.schemas import CriterionScore, EvaluationItem, JudgeVerdict, SuiteReport
from app.validation import run_test_retest_validation


def test_run_test_retest_validation():
    mock_evaluator = MagicMock(spec=Evaluator)
    mock_client = MagicMock(spec=JudgeClient)
    mock_client.judge_model = "qwen2.5:1.5b-instruct"
    mock_evaluator.judge_client = mock_client

    v1 = JudgeVerdict(
        question_id="s1",
        criteria_scores={"correctness": CriterionScore(criterion="correctness", score=4.0, rationale="")},
        overall_score=4.0,
        passed=True,
        summary_rationale="",
    )
    v2 = JudgeVerdict(
        question_id="s1",
        criteria_scores={"correctness": CriterionScore(criterion="correctness", score=4.0, rationale="")},
        overall_score=4.0,
        passed=True,
        summary_rationale="",
    )

    report1 = SuiteReport(
        total_cases=1, passed_cases=1, failed_cases=0, pass_rate=1.0, mean_overall_score=4.0, verdicts=[v1]
    )
    report2 = SuiteReport(
        total_cases=1, passed_cases=1, failed_cases=0, pass_rate=1.0, mean_overall_score=4.0, verdicts=[v2]
    )

    mock_evaluator.evaluate_suite.side_effect = [report1, report2]
    items = [EvaluationItem(id="s1", input="q", model_output="o")]

    val_report = run_test_retest_validation(mock_evaluator, items, temperature=0.0)

    assert val_report.total_cases == 1
    assert val_report.unchanged_cases == 1
    assert val_report.changed_cases == 0
    assert val_report.consistency_rate == 1.0
    assert val_report.mean_score_delta == 0.0
    assert val_report.judge_model == "qwen2.5:1.5b-instruct"


def test_run_test_retest_validation_empty():
    mock_evaluator = MagicMock(spec=Evaluator)
    mock_client = MagicMock(spec=JudgeClient)
    mock_client.judge_model = "qwen2.5:1.5b-instruct"
    mock_evaluator.judge_client = mock_client

    val_report = run_test_retest_validation(mock_evaluator, [])

    assert val_report.total_cases == 0
    assert val_report.consistency_rate == 1.0
