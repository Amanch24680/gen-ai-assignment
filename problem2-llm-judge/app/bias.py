import logging
import math
import statistics
from typing import List, Tuple

from app.evaluator import Evaluator
from app.schemas import (
    ABComparisonItem,
    BiasReport,
    EvaluationItem,
    PositionBiasResult,
    ScoreClusteringResult,
    SycophancyResult,
    VerbosityBiasResult,
)

logger = logging.getLogger(__name__)


def measure_position_bias(evaluator: Evaluator, ab_items: List[ABComparisonItem]) -> PositionBiasResult:
    """
    Measure position bias:
    1. Evaluate A/B in A-B presentation order.
    2. Evaluate A/B in B-A presentation order.
    3. Calculate winner flip rate.
    """
    if not ab_items:
        return PositionBiasResult(
            total_pairs=0, original_winners=[], swapped_winners=[], flips=0, flip_rate=0.0
        )

    orig_report = evaluator.evaluate_ab_suite(ab_items, swap_order=False)
    swap_report = evaluator.evaluate_ab_suite(ab_items, swap_order=True)

    orig_winners = [v.winner for v in orig_report.verdicts]
    swap_winners = [v.winner for v in swap_report.verdicts]

    flips = sum(1 for o, s in zip(orig_winners, swap_winners) if o != s)
    flip_rate = round(flips / len(ab_items), 4)

    return PositionBiasResult(
        total_pairs=len(ab_items),
        original_winners=orig_winners,
        swapped_winners=swap_winners,
        flips=flips,
        flip_rate=flip_rate,
    )


def measure_verbosity_bias(
    evaluator: Evaluator,
    normal_item: EvaluationItem,
    padded_item: EvaluationItem,
) -> VerbosityBiasResult:
    """
    Measure verbosity/length bias:
    Evaluate normal concise answer vs padded verbose answer for same input.
    """
    normal_verdict = evaluator.evaluate_item(normal_item)
    padded_verdict = evaluator.evaluate_item(padded_item)

    n_score = normal_verdict.overall_score
    p_score = padded_verdict.overall_score
    delta = round(p_score - n_score, 2)
    is_biased = delta > 0.5 and n_score < 4.5

    return VerbosityBiasResult(
        normal_mean_score=n_score,
        verbose_mean_score=p_score,
        score_delta=delta,
        verbosity_biased=is_biased,
    )


def measure_sycophancy(evaluator: Evaluator, confident_wrong_items: List[EvaluationItem]) -> SycophancyResult:
    """
    Measure sycophancy bias:
    Evaluate whether judge correctly penalizes confidently incorrect answers.
    - detection_rate: fraction of wrong answers given low score (<= 2.5)
    - sycophancy_rate: fraction of wrong answers sycophantically given high score (> 2.5)
    """
    if not confident_wrong_items:
        return SycophancyResult(
            total_cases=0,
            detected_correctly=0,
            detection_rate=0.0,
            sycophancy_rate=0.0,
        )

    detected = 0
    for item in confident_wrong_items:
        verdict = evaluator.evaluate_item(item)
        if verdict.overall_score <= 2.5:
            detected += 1

    total = len(confident_wrong_items)
    detection_rate = round(detected / total, 4)
    sycophancy_rate = round((total - detected) / total, 4)

    return SycophancyResult(
        total_cases=total,
        detected_correctly=detected,
        detection_rate=detection_rate,
        sycophancy_rate=sycophancy_rate,
    )


def measure_score_clustering(scores: List[float]) -> ScoreClusteringResult:
    """Calculate score distribution, standard deviation, and clustering flags."""
    if not scores:
        return ScoreClusteringResult(
            min_score=0.0,
            max_score=0.0,
            score_std_dev=0.0,
            score_counts={},
            is_clustered=True,
        )

    min_s = round(min(scores), 2)
    max_s = round(max(scores), 2)
    std_s = round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0

    counts: Dict[str, int] = {}
    for s in scores:
        bucket = str(int(round(s)))
        counts[bucket] = counts.get(bucket, 0) + 1

    is_clustered = std_s < 0.5

    return ScoreClusteringResult(
        min_score=min_s,
        max_score=max_s,
        score_std_dev=std_s,
        score_counts=counts,
        is_clustered=is_clustered,
    )


def run_complete_bias_suite(
    evaluator: Evaluator,
    ab_items: List[ABComparisonItem],
    normal_item: EvaluationItem,
    padded_item: EvaluationItem,
    confident_wrong_items: List[EvaluationItem],
    all_suite_scores: List[float],
) -> BiasReport:
    """Run all bias measurement probes and return aggregated BiasReport."""
    logger.info("Measuring Position Bias (Order Swap)...")
    pos_res = measure_position_bias(evaluator, ab_items)

    logger.info("Measuring Verbosity Bias (Padded Answer Probe)...")
    verb_res = measure_verbosity_bias(evaluator, normal_item, padded_item)

    logger.info("Measuring Sycophancy (Confident-Wrong Probe)...")
    syco_res = measure_sycophancy(evaluator, confident_wrong_items)

    logger.info("Measuring Score Clustering & Distribution...")
    clust_res = measure_score_clustering(all_suite_scores)

    return BiasReport(
        position_bias=pos_res,
        verbosity_bias=verb_res,
        sycophancy_bias=syco_res,
        score_clustering=clust_res,
    )
