import logging
from typing import List, Optional

from app.config import settings
from app.evaluator import Evaluator
from app.schemas import CaseValidationComparison, EvaluationItem, ValidationReport

logger = logging.getLogger(__name__)


def run_test_retest_validation(
    evaluator: Evaluator,
    items: List[EvaluationItem],
    temperature: float = 0.0,
) -> ValidationReport:
    """
    Perform test-retest consistency validation by running the suite twice at temperature=0.0.
    Calculates consistency_rate, changed_cases, and per-case score deltas.
    """
    if not items:
        return ValidationReport(
            total_cases=0,
            unchanged_cases=0,
            changed_cases=0,
            consistency_rate=1.0,
            mean_score_delta=0.0,
            judge_model=evaluator.judge_client.judge_model,
            temperature=temperature,
            case_comparisons=[],
        )

    logger.info("Executing Run 1 of Test-Retest Validation...")
    report1 = evaluator.evaluate_suite(items)

    logger.info("Executing Run 2 of Test-Retest Validation...")
    report2 = evaluator.evaluate_suite(items)

    verdicts1_map = {v.question_id: v for v in report1.verdicts}
    verdicts2_map = {v.question_id: v for v in report2.verdicts}

    comparisons: List[CaseValidationComparison] = []
    total_delta = 0.0
    unchanged_count = 0

    for item in items:
        v1 = verdicts1_map.get(item.id)
        v2 = verdicts2_map.get(item.id)

        if v1 and v2:
            s1 = v1.overall_score
            s2 = v2.overall_score
            p1 = v1.passed
            p2 = v2.passed
            delta = round(abs(s1 - s2), 2)
            total_delta += delta

            # Unchanged if pass/fail status is identical and overall score delta <= 0.1
            unchanged = (p1 == p2) and (delta <= 0.1)
            if unchanged:
                unchanged_count += 1

            comparisons.append(
                CaseValidationComparison(
                    question_id=item.id,
                    run1_score=s1,
                    run2_score=s2,
                    run1_passed=p1,
                    run2_passed=p2,
                    score_delta=delta,
                    unchanged=unchanged,
                )
            )

    total_cases = len(comparisons)
    changed_count = total_cases - unchanged_count
    consistency_rate = round(unchanged_count / total_cases, 4) if total_cases > 0 else 1.0
    mean_delta = round(total_delta / total_cases, 2) if total_cases > 0 else 0.0

    return ValidationReport(
        total_cases=total_cases,
        unchanged_cases=unchanged_count,
        changed_cases=changed_count,
        consistency_rate=consistency_rate,
        mean_score_delta=mean_delta,
        judge_model=evaluator.judge_client.judge_model,
        temperature=temperature,
        case_comparisons=comparisons,
    )
