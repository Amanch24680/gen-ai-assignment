import statistics
from typing import Dict, List, Tuple

from app.schemas import ABSuiteReport, ABVerdict, JudgeVerdict, SuiteReport


def aggregate_suite_results(verdicts: List[JudgeVerdict], pass_threshold: float = 3.5) -> SuiteReport:
    """Aggregate individual item verdicts into a complete suite report including per-category metrics."""
    if not verdicts:
        return SuiteReport(
            total_cases=0,
            passed_cases=0,
            failed_cases=0,
            pass_rate=0.0,
            mean_overall_score=0.0,
            mean_criterion_scores={},
            category_scores={},
            verdicts=[],
        )

    total_cases = len(verdicts)
    passed_cases = sum(1 for v in verdicts if v.passed)
    failed_cases = total_cases - passed_cases
    pass_rate = round(passed_cases / total_cases, 4)

    mean_overall = round(sum(v.overall_score for v in verdicts) / total_cases, 2)

    # Compute mean per-criterion score
    all_criteria = set()
    for v in verdicts:
        all_criteria.update(v.criteria_scores.keys())

    mean_criterion_scores: Dict[str, float] = {}
    for crit in sorted(all_criteria):
        scores = [v.criteria_scores[crit].score for v in verdicts if crit in v.criteria_scores]
        if scores:
            mean_criterion_scores[crit] = round(sum(scores) / len(scores), 2)

    # Compute per-category breakdown
    category_map: Dict[str, List[JudgeVerdict]] = {}
    for v in verdicts:
        cat = v.category or "general"
        category_map.setdefault(cat, []).append(v)

    category_scores: Dict[str, Dict[str, Any]] = {}
    for cat, cat_verdicts in sorted(category_map.items()):
        cat_total = len(cat_verdicts)
        cat_passed = sum(1 for v in cat_verdicts if v.passed)
        cat_pass_rate = round(cat_passed / cat_total, 4) if cat_total > 0 else 0.0
        cat_mean_overall = round(sum(v.overall_score for v in cat_verdicts) / cat_total, 2)

        cat_crit_scores: Dict[str, float] = {}
        for crit in sorted(all_criteria):
            scores = [v.criteria_scores[crit].score for v in cat_verdicts if crit in v.criteria_scores]
            if scores:
                cat_crit_scores[crit] = round(sum(scores) / len(scores), 2)

        category_scores[cat] = {
            "total_cases": cat_total,
            "passed_cases": cat_passed,
            "pass_rate": cat_pass_rate,
            "mean_overall_score": cat_mean_overall,
            "mean_criterion_scores": cat_crit_scores,
        }

    return SuiteReport(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=pass_rate,
        mean_overall_score=mean_overall,
        mean_criterion_scores=mean_criterion_scores,
        category_scores=category_scores,
        verdicts=verdicts,
    )


def aggregate_ab_results(verdicts: List[ABVerdict]) -> ABSuiteReport:
    """Aggregate pairwise A/B comparison verdicts into an A/B suite report."""
    if not verdicts:
        return ABSuiteReport(
            total_cases=0,
            wins_a=0,
            wins_b=0,
            ties=0,
            win_rate_a=0.0,
            win_rate_b=0.0,
            overall_winner="Tie",
            verdicts=[],
        )

    total_cases = len(verdicts)
    wins_a = sum(1 for v in verdicts if v.winner.upper() == "A")
    wins_b = sum(1 for v in verdicts if v.winner.upper() == "B")
    ties = sum(1 for v in verdicts if v.winner.upper() == "TIE")

    non_ties = wins_a + wins_b
    win_rate_a = round(wins_a / non_ties, 4) if non_ties > 0 else 0.0
    win_rate_b = round(wins_b / non_ties, 4) if non_ties > 0 else 0.0

    if wins_a > wins_b:
        overall_winner = "Candidate A"
    elif wins_b > wins_a:
        overall_winner = "Candidate B"
    else:
        overall_winner = "Tie"

    return ABSuiteReport(
        total_cases=total_cases,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        win_rate_a=win_rate_a,
        win_rate_b=win_rate_b,
        overall_winner=overall_winner,
        verdicts=verdicts,
    )
