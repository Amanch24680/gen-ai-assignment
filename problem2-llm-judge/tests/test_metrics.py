import pytest
from app.aggregation import aggregate_ab_results, aggregate_suite_results
from app.schemas import ABVerdict, CriterionScore, JudgeVerdict


def test_aggregate_suite_results_calculations():
    verdicts = [
        JudgeVerdict(
            question_id="q1",
            criteria_scores={
                "correctness": CriterionScore(criterion="correctness", score=5.0, rationale=""),
                "faithfulness": CriterionScore(criterion="faithfulness", score=5.0, rationale=""),
            },
            overall_score=5.0,
            passed=True,
            summary_rationale="Pass",
        ),
        JudgeVerdict(
            question_id="q2",
            criteria_scores={
                "correctness": CriterionScore(criterion="correctness", score=2.0, rationale=""),
                "faithfulness": CriterionScore(criterion="faithfulness", score=2.0, rationale=""),
            },
            overall_score=2.0,
            passed=False,
            summary_rationale="Fail",
        ),
    ]

    report = aggregate_suite_results(verdicts, pass_threshold=3.5)

    assert report.total_cases == 2
    assert report.passed_cases == 1
    assert report.failed_cases == 1
    assert report.pass_rate == 0.5
    assert report.mean_overall_score == 3.5
    assert report.mean_criterion_scores["correctness"] == 3.5


def test_aggregate_ab_results_calculations():
    verdicts = [
        ABVerdict(question_id="ab1", winner="A", score_a=5.0, score_b=3.0, rationale="A won"),
        ABVerdict(question_id="ab2", winner="A", score_a=4.5, score_b=2.0, rationale="A won"),
        ABVerdict(question_id="ab3", winner="B", score_a=2.0, score_b=4.0, rationale="B won"),
        ABVerdict(question_id="ab4", winner="Tie", score_a=3.0, score_b=3.0, rationale="Tie"),
    ]

    report = aggregate_ab_results(verdicts)

    assert report.total_cases == 4
    assert report.wins_a == 2
    assert report.wins_b == 1
    assert report.ties == 1
    assert report.overall_winner == "Candidate A"
    assert report.win_rate_a == round(2 / 3, 4)


def test_aggregate_suite_results_category_breakdown():
    verdicts = [
        JudgeVerdict(
            question_id="c1",
            category="cat_a",
            criteria_scores={
                "correctness": CriterionScore(criterion="correctness", score=5.0, rationale=""),
            },
            overall_score=5.0,
            passed=True,
            summary_rationale="",
        ),
        JudgeVerdict(
            question_id="c2",
            category="cat_a",
            criteria_scores={
                "correctness": CriterionScore(criterion="correctness", score=4.0, rationale=""),
            },
            overall_score=4.0,
            passed=True,
            summary_rationale="",
        ),
        JudgeVerdict(
            question_id="c3",
            category="cat_b",
            criteria_scores={
                "correctness": CriterionScore(criterion="correctness", score=2.0, rationale=""),
            },
            overall_score=2.0,
            passed=False,
            summary_rationale="",
        ),
    ]

    report = aggregate_suite_results(verdicts, pass_threshold=3.5)

    assert "cat_a" in report.category_scores
    assert "cat_b" in report.category_scores

    cat_a = report.category_scores["cat_a"]
    assert cat_a["total_cases"] == 2
    assert cat_a["passed_cases"] == 2
    assert cat_a["pass_rate"] == 1.0
    assert cat_a["mean_overall_score"] == 4.5

    cat_b = report.category_scores["cat_b"]
    assert cat_b["total_cases"] == 1
    assert cat_b["passed_cases"] == 0
    assert cat_b["pass_rate"] == 0.0
    assert cat_b["mean_overall_score"] == 2.0


def test_aggregate_empty_suite_reports():
    empty_suite = aggregate_suite_results([])
    assert empty_suite.total_cases == 0
    assert empty_suite.pass_rate == 0.0
    assert empty_suite.category_scores == {}

    empty_ab = aggregate_ab_results([])
    assert empty_ab.total_cases == 0
    assert empty_ab.overall_winner == "Tie"
