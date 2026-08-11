import pytest
from app.parser import parse_ab_verdict, parse_single_item_verdict


def test_parse_valid_single_item_verdict():
    raw_json = """
    {
      "criteria_scores": {
        "correctness": {"score": 5.0, "rationale": "Perfect accuracy."},
        "faithfulness": {"score": 4.5, "rationale": "Grounded."},
        "completeness": {"score": 4.0, "rationale": "Mostly complete."},
        "instruction_following": {"score": 5.0, "rationale": "Followed instructions."},
        "tone_safety": {"score": 5.0, "rationale": "Safe."}
      },
      "overall_score": 4.7,
      "summary_rationale": "Excellent quality."
    }
    """
    verdict = parse_single_item_verdict(raw_json, question_id="test_01", pass_threshold=3.5)

    assert verdict.question_id == "test_01"
    assert verdict.overall_score == 4.7
    assert verdict.passed is True
    assert verdict.parsed_fallback is False
    assert verdict.criteria_scores["correctness"].score == 5.0
    assert verdict.criteria_scores["completeness"].score == 4.0


def test_parse_markdown_wrapped_json():
    raw = """```json
    {
      "criteria_scores": {
        "correctness": {"score": 4.0, "rationale": "Good."},
        "faithfulness": {"score": 4.0, "rationale": "Good."},
        "completeness": {"score": 4.0, "rationale": "Good."},
        "instruction_following": {"score": 4.0, "rationale": "Good."},
        "tone_safety": {"score": 4.0, "rationale": "Good."}
      },
      "overall_score": 4.0,
      "summary_rationale": "Good."
    }
    ```"""
    verdict = parse_single_item_verdict(raw, question_id="test_md")
    assert verdict.overall_score == 4.0
    assert verdict.parsed_fallback is False


def test_parse_malformed_json_fallback():
    raw_malformed = """
    Here is my evaluation:
    The output has correctness: {"score": 2.0, "rationale": "Incorrect"}
    faithfulness: {"score": 1.0, "rationale": "Hallucinated"}
    Overall this answer is poor.
    """
    verdict = parse_single_item_verdict(raw_malformed, question_id="test_malformed")

    assert verdict.question_id == "test_malformed"
    assert verdict.parsed_fallback is True
    assert 1.0 <= verdict.overall_score <= 5.0
    assert "correctness" in verdict.criteria_scores


def test_parse_ab_verdict():
    raw_ab = """
    {
      "winner": "A",
      "score_a": 4.5,
      "score_b": 2.0,
      "rationale": "Candidate A was accurate while Candidate B was incorrect."
    }
    """
    data = parse_ab_verdict(raw_ab, question_id="ab_01")
    assert data["winner"] == "A"
    assert data["score_a"] == 4.5
    assert data["score_b"] == 2.0
