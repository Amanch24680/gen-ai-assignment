import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.aggregation import aggregate_ab_results, aggregate_suite_results
from app.config import settings
from app.judge import JudgeClient
from app.parser import parse_ab_verdict, parse_single_item_verdict
from app.prompts import (
    AB_COMPARISON_JUDGE_SYSTEM_PROMPT,
    AB_COMPARISON_JUDGE_USER_PROMPT,
    SINGLE_ITEM_JUDGE_SYSTEM_PROMPT,
    SINGLE_ITEM_JUDGE_USER_PROMPT,
)
from app.schemas import (
    ABComparisonItem,
    ABSuiteReport,
    ABVerdict,
    EvaluationItem,
    JudgeVerdict,
    SuiteReport,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Orchestrates LLM-as-Judge evaluation workflows:
    - Single item evaluation
    - Suite evaluation
    - Pairwise A/B comparison evaluation
    """

    def __init__(self, judge_client: Optional[JudgeClient] = None):
        self.judge_client = judge_client or JudgeClient()

    def evaluate_item(self, item: EvaluationItem) -> JudgeVerdict:
        """Evaluate a single EvaluationItem using the LLM Judge."""
        user_prompt = SINGLE_ITEM_JUDGE_USER_PROMPT.format(
            input=item.input,
            system_prompt=item.system_prompt or "None",
            expected_output=item.expected_output or "None",
            model_output=item.model_output,
        )

        raw_response, latency_ms = self.judge_client.generate_judge_response(
            system_prompt=SINGLE_ITEM_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        verdict = parse_single_item_verdict(
            raw_response=raw_response,
            question_id=item.id,
            category=item.category or "general",
            pass_threshold=settings.pass_score_threshold,
            latency_ms=latency_ms,
        )
        return verdict

    def evaluate_suite(self, items: List[EvaluationItem]) -> SuiteReport:
        """Evaluate a list of EvaluationItem cases and aggregate into SuiteReport."""
        verdicts: List[JudgeVerdict] = []
        for item in items:
            logger.info(f"Evaluating case [{item.id}]...")
            verdict = self.evaluate_item(item)
            verdicts.append(verdict)

        report = aggregate_suite_results(verdicts, pass_threshold=settings.pass_score_threshold)
        return report

    def evaluate_ab_item(self, item: ABComparisonItem, swap_order: bool = False) -> ABVerdict:
        """
        Perform pairwise A/B comparison evaluation.
        If swap_order is True, Candidate A and Candidate B are swapped in the prompt.
        """
        candidate_a = item.candidate_b_output if swap_order else item.candidate_a_output
        candidate_b = item.candidate_a_output if swap_order else item.candidate_b_output

        user_prompt = AB_COMPARISON_JUDGE_USER_PROMPT.format(
            input=item.input,
            system_prompt=item.system_prompt or "None",
            expected_output=item.expected_output or "None",
            candidate_a=candidate_a,
            candidate_b=candidate_b,
        )

        raw_response, latency_ms = self.judge_client.generate_judge_response(
            system_prompt=AB_COMPARISON_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        parsed_data = parse_ab_verdict(raw_response, question_id=item.id, latency_ms=latency_ms)

        raw_winner = parsed_data["winner"]
        score_a = parsed_data["score_a"]
        score_b = parsed_data["score_b"]

        # Handle score and winner mapping when swapped
        if swap_order:
            if raw_winner == "A":
                winner = "B"
            elif raw_winner == "B":
                winner = "A"
            else:
                winner = "Tie"
            eff_score_a = score_b
            eff_score_b = score_a
        else:
            winner = raw_winner
            eff_score_a = score_a
            eff_score_b = score_b

        return ABVerdict(
            question_id=item.id,
            winner=winner,
            score_a=eff_score_a,
            score_b=eff_score_b,
            rationale=parsed_data["rationale"],
            latency_ms=latency_ms,
        )

    def evaluate_ab_suite(self, items: List[ABComparisonItem], swap_order: bool = False) -> ABSuiteReport:
        """Evaluate a suite of A/B comparison cases."""
        verdicts: List[ABVerdict] = []
        for item in items:
            logger.info(f"Evaluating A/B comparison [{item.id}]...")
            verdict = self.evaluate_ab_item(item, swap_order=swap_order)
            verdicts.append(verdict)

        report = aggregate_ab_results(verdicts)
        return report
