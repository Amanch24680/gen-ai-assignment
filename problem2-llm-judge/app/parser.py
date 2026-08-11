import json
import logging
import re
from typing import Any, Dict, Tuple

from app.schemas import CriterionScore, JudgeVerdict

logger = logging.getLogger(__name__)


DEFAULT_CRITERIA = ["correctness", "faithfulness", "completeness", "instruction_following", "tone_safety"]


def clean_json_text(text: str) -> str:
    """Clean markdown code block wrappers and leading/trailing whitespace."""
    if not text:
        return ""
    cleaned = text.strip()
    # Remove markdown codeblock syntax
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_single_item_verdict(
    raw_response: str,
    question_id: str,
    pass_threshold: float = 3.5,
    latency_ms: float = 0.0,
) -> JudgeVerdict:
    """
    Parse raw judge LLM response into structured JudgeVerdict.
    Uses multi-stage parsing with robust fallback handling.
    """
    cleaned_text = clean_json_text(raw_response)
    parsed_fallback = False
    data: Dict[str, Any] = {}

    # Stage 1: Direct JSON parsing
    try:
        data = json.loads(cleaned_text)
    except Exception:
        # Stage 2: Regex extraction of outermost JSON object
        json_match = re.search(r"(\{.*\})", cleaned_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except Exception:
                data = {}

    # Stage 3: Validate and extract criteria scores
    criteria_scores: Dict[str, CriterionScore] = {}
    raw_criteria = data.get("criteria_scores", {})

    if isinstance(raw_criteria, dict) and raw_criteria:
        for crit in DEFAULT_CRITERIA:
            crit_item = raw_criteria.get(crit, {})
            if isinstance(crit_item, dict):
                score = float(crit_item.get("score", 3.0))
                score = max(1.0, min(5.0, score))
                rationale = str(crit_item.get("rationale", "Standard automated rationale."))
                criteria_scores[crit] = CriterionScore(criterion=crit, score=score, rationale=rationale)
            elif isinstance(crit_item, (int, float)):
                score = max(1.0, min(5.0, float(crit_item)))
                criteria_scores[crit] = CriterionScore(
                    criterion=crit, score=score, rationale="Score extracted from scalar value."
                )

    # Fallback if parsing failed or criteria missing
    if len(criteria_scores) < len(DEFAULT_CRITERIA):
        parsed_fallback = True
        logger.warning(f"Fallback parsing triggered for [{question_id}]. Extracting via regex fallback.")

        for crit in DEFAULT_CRITERIA:
            if crit not in criteria_scores:
                # Regex search for criterion score
                pattern = rf'"{crit}"\s*:\s*\{{\s*"score"\s*:\s*([1-5]\.?[0-9]*)'
                match = re.search(pattern, raw_response, re.IGNORECASE)
                score = float(match.group(1)) if match else 3.0
                score = max(1.0, min(5.0, score))
                criteria_scores[crit] = CriterionScore(
                    criterion=crit,
                    score=score,
                    rationale="Extracted via fallback regex parser." if match else "Default fallback score.",
                )

    # Overall score computation
    if "overall_score" in data and isinstance(data["overall_score"], (int, float)):
        overall_score = max(1.0, min(5.0, float(data["overall_score"])))
    else:
        # Average of criteria scores
        overall_score = sum(c.score for c in criteria_scores.values()) / len(criteria_scores)

    overall_score = round(overall_score, 2)
    summary_rationale = str(data.get("summary_rationale", "Verdict generated via judge pipeline."))
    passed = overall_score >= pass_threshold

    return JudgeVerdict(
        question_id=question_id,
        criteria_scores=criteria_scores,
        overall_score=overall_score,
        passed=passed,
        summary_rationale=summary_rationale,
        raw_response=raw_response,
        latency_ms=round(latency_ms, 2),
        parsed_fallback=parsed_fallback,
    )


def parse_ab_verdict(
    raw_response: str,
    question_id: str,
    latency_ms: float = 0.0,
) -> Dict[str, Any]:
    """Parse raw judge LLM response for pairwise A/B comparison."""
    cleaned_text = clean_json_text(raw_response)
    data: Dict[str, Any] = {}

    try:
        data = json.loads(cleaned_text)
    except Exception:
        json_match = re.search(r"(\{.*\})", cleaned_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except Exception:
                data = {}

    winner = str(data.get("winner", "Tie")).strip().upper()
    if winner not in ["A", "B", "TIE"]:
        # Regex fallback for winner
        if re.search(r'"winner"\s*:\s*"A"', raw_response, re.IGNORECASE):
            winner = "A"
        elif re.search(r'"winner"\s*:\s*"B"', raw_response, re.IGNORECASE):
            winner = "B"
        else:
            winner = "Tie"

    score_a = float(data.get("score_a", 3.0)) if isinstance(data.get("score_a"), (int, float)) else 3.0
    score_b = float(data.get("score_b", 3.0)) if isinstance(data.get("score_b"), (int, float)) else 3.0
    score_a = max(1.0, min(5.0, score_a))
    score_b = max(1.0, min(5.0, score_b))

    rationale = str(data.get("rationale", "Pairwise A/B comparison completed."))

    return {
        "question_id": question_id,
        "winner": winner,
        "score_a": score_a,
        "score_b": score_b,
        "rationale": rationale,
        "latency_ms": round(latency_ms, 2),
    }
