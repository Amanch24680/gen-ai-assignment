from unittest.mock import MagicMock, patch
import pytest

from app.cli import main
from app.schemas import (
    ABSuiteReport,
    ABVerdict,
    BiasReport,
    CriterionScore,
    JudgeVerdict,
    PositionBiasResult,
    ScoreClusteringResult,
    SuiteReport,
    SycophancyResult,
    VerbosityBiasResult,
)


@patch("app.cli.run_complete_bias_suite")
@patch("app.evaluator.Evaluator.evaluate_suite")
@patch("app.evaluator.Evaluator.evaluate_ab_suite")
def test_cli_all_execution(mock_ab, mock_suite, mock_bias, tmp_path, capsys):
    mock_suite_report = SuiteReport(
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        pass_rate=1.0,
        mean_overall_score=4.5,
        mean_criterion_scores={"correctness": 4.5},
        verdicts=[
            JudgeVerdict(
                question_id="s001",
                criteria_scores={"correctness": CriterionScore(criterion="correctness", score=4.5, rationale="")},
                overall_score=4.5,
                passed=True,
                summary_rationale="Pass",
            )
        ],
    )
    mock_ab_report = ABSuiteReport(
        total_cases=1,
        wins_a=1,
        wins_b=0,
        ties=0,
        win_rate_a=1.0,
        win_rate_b=0.0,
        overall_winner="Candidate A",
        verdicts=[ABVerdict(question_id="ab001", winner="A", score_a=5.0, score_b=3.0, rationale="")],
    )
    mock_bias_report = BiasReport(
        position_bias=PositionBiasResult(total_pairs=1, original_winners=["A"], swapped_winners=["A"], flips=0, flip_rate=0.0),
        verbosity_bias=VerbosityBiasResult(normal_mean_score=3.0, verbose_mean_score=4.5, score_delta=1.5, verbosity_biased=True),
        sycophancy_bias=SycophancyResult(total_cases=1, detected_correctly=1, sycophancy_rate=0.0),
        score_clustering=ScoreClusteringResult(min_score=1.0, max_score=5.0, score_std_dev=1.2, score_counts={"1": 1}, is_clustered=False),
    )

    mock_suite.return_value = mock_suite_report
    mock_ab.return_value = mock_ab_report
    mock_bias.return_value = mock_bias_report

    result = main(["--all"])
    assert result == 0

    captured = capsys.readouterr()
    assert "RUNNING LLM-AS-JUDGE SUITE EVALUATION" in captured.out
    assert "RUNNING PAIRWISE A/B COMPARISON EVALUATION" in captured.out
    assert "RUNNING LLM-AS-JUDGE BIAS MEASUREMENT PROBES" in captured.out


@patch("app.evaluator.Evaluator.evaluate_suite")
def test_cli_suite_only(mock_suite, tmp_path, capsys):
    mock_suite_report = SuiteReport(
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        pass_rate=1.0,
        mean_overall_score=4.0,
        mean_criterion_scores={"correctness": 4.0},
        verdicts=[],
    )
    mock_suite.return_value = mock_suite_report

    out_json = tmp_path / "suite_out.json"
    suite_json = tmp_path / "test_suite.json"
    suite_json.write_text('[{"id":"s1","input":"q","model_output":"o"}]', encoding="utf-8")

    result = main(["--suite", str(suite_json), "--output", str(out_json)])
    assert result == 0
    assert out_json.exists()
    captured = capsys.readouterr()
    assert "RUNNING LLM-AS-JUDGE SUITE EVALUATION" in captured.out


@patch("app.evaluator.Evaluator.evaluate_ab_suite")
def test_cli_ab_only(mock_ab, tmp_path, capsys):
    mock_ab_report = ABSuiteReport(
        total_cases=1,
        wins_a=1,
        wins_b=0,
        ties=0,
        win_rate_a=1.0,
        win_rate_b=0.0,
        overall_winner="Candidate A",
        verdicts=[],
    )
    mock_ab.return_value = mock_ab_report

    out_json = tmp_path / "ab_out.json"
    ab_json = tmp_path / "test_ab.json"
    ab_json.write_text('[{"id":"ab1","input":"q","candidate_a_output":"a","candidate_b_output":"b"}]', encoding="utf-8")

    result = main(["--ab", str(ab_json), "--output", str(out_json)])
    assert result == 0
    assert out_json.exists()
    captured = capsys.readouterr()
    assert "RUNNING PAIRWISE A/B COMPARISON EVALUATION" in captured.out
