from unittest.mock import MagicMock
import pytest

from app.bias import (
    measure_position_bias,
    measure_score_clustering,
    measure_sycophancy,
    measure_verbosity_bias,
)
from app.evaluator import Evaluator
from app.schemas import ABComparisonItem, ABSuiteReport, ABVerdict, EvaluationItem, JudgeVerdict, CriterionScore


def test_measure_position_bias():
    mock_evaluator = MagicMock(spec=Evaluator)

    orig_report = ABSuiteReport(
        total_cases=2, wins_a=2, wins_b=0, ties=0, win_rate_a=1.0, win_rate_b=0.0, overall_winner="A",
        verdicts=[
            ABVerdict(question_id="1", winner="A", score_a=5.0, score_b=3.0, rationale=""),
            ABVerdict(question_id="2", winner="A", score_a=5.0, score_b=3.0, rationale=""),
        ]
    )
    swap_report = ABSuiteReport(
        total_cases=2, wins_a=1, wins_b=1, ties=0, win_rate_a=0.5, win_rate_b=0.5, overall_winner="Tie",
        verdicts=[
            ABVerdict(question_id="1", winner="A", score_a=5.0, score_b=3.0, rationale=""),
            ABVerdict(question_id="2", winner="B", score_a=3.0, score_b=5.0, rationale=""),
        ]
    )

    mock_evaluator.evaluate_ab_suite.side_effect = [orig_report, swap_report]
    ab_items = [
        ABComparisonItem(id="1", input="q1", candidate_a_output="a1", candidate_b_output="b1"),
        ABComparisonItem(id="2", input="q2", candidate_a_output="a2", candidate_b_output="b2"),
    ]

    res = measure_position_bias(mock_evaluator, ab_items)

    assert res.total_pairs == 2
    assert res.flips == 1
    assert res.flip_rate == 0.5


def test_measure_verbosity_bias():
    mock_evaluator = MagicMock(spec=Evaluator)

    normal_verdict = JudgeVerdict(
        question_id="1", criteria_scores={}, overall_score=3.0, passed=False, summary_rationale=""
    )
    padded_verdict = JudgeVerdict(
        question_id="1_pad", criteria_scores={}, overall_score=4.5, passed=True, summary_rationale=""
    )
    mock_evaluator.evaluate_item.side_effect = [normal_verdict, padded_verdict]

    n_item = EvaluationItem(id="1", input="q", model_output="concise")
    p_item = EvaluationItem(id="1_pad", input="q", model_output="padded")

    res = measure_verbosity_bias(mock_evaluator, n_item, p_item)

    assert res.normal_mean_score == 3.0
    assert res.verbose_mean_score == 4.5
    assert res.score_delta == 1.5
    assert res.verbosity_biased is True


def test_measure_sycophancy():
    mock_evaluator = MagicMock(spec=Evaluator)

    v1 = JudgeVerdict(question_id="1", criteria_scores={}, overall_score=1.5, passed=False, summary_rationale="")
    v2 = JudgeVerdict(question_id="2", criteria_scores={}, overall_score=4.5, passed=True, summary_rationale="")

    mock_evaluator.evaluate_item.side_effect = [v1, v2]
    items = [
        EvaluationItem(id="1", input="q1", model_output="wrong 1"),
        EvaluationItem(id="2", input="q2", model_output="wrong 2"),
    ]

    res = measure_sycophancy(mock_evaluator, items)

    assert res.total_cases == 2
    assert res.detected_correctly == 1
    assert res.sycophancy_rate == 0.5


def test_measure_score_clustering():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = measure_score_clustering(scores)

    assert res.min_score == 1.0
    assert res.max_score == 5.0
    assert res.score_std_dev > 1.0
    assert res.is_clustered is False
