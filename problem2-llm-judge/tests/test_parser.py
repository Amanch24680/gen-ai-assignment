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


def test_parse_single_item_python_calculated_overall():
    # LLM provided overall_score=5.0, but criteria mean is (5+3+3+5+4)/5 = 4.0
    raw_json = """
    {
      "criteria_scores": {
        "correctness": {"score": 5.0, "rationale": "Accurate."},
        "faithfulness": {"score": 3.0, "rationale": "Slight ungrounded detail."},
        "completeness": {"score": 3.0, "rationale": "Missed subpart."},
        "instruction_following": {"score": 5.0, "rationale": "Followed rules."},
        "tone_safety": {"score": 4.0, "rationale": "Acceptable tone."}
      },
      "overall_score": 5.0,
      "summary_rationale": "Overstated overall score by LLM."
    }
    """
    verdict = parse_single_item_verdict(raw_json, question_id="test_py_calc", category="test_cat", pass_threshold=3.5)

    assert verdict.question_id == "test_py_calc"
    assert verdict.category == "test_cat"
    assert verdict.overall_score == 4.0  # Calculated by Python
    assert verdict.llm_overall_score == 5.0  # Retained for audit/debugging
    assert verdict.passed is True


def test_parse_conversational_wrapped_json():
    raw_conv = """
    Sure, I'd be happy to evaluate this for you!
    {
      "criteria_scores": {
        "correctness": {"score": 1.0, "rationale": "Factually wrong."},
        "faithfulness": {"score": 1.0, "rationale": "Contradicts ref."},
        "completeness": {"score": 1.0, "rationale": "Incomplete."},
        "instruction_following": {"score": 1.0, "rationale": "Failed constraints."},
        "tone_safety": {"score": 1.0, "rationale": "Deceptive."}
      },
      "overall_score": 1.0,
      "summary_rationale": "Completely failed."
    }
    Hope this evaluation helps!
    """
    verdict = parse_single_item_verdict(raw_conv, question_id="test_conv")

    assert verdict.overall_score == 1.0
    assert verdict.passed is False
    assert verdict.parsed_fallback is False


def test_parse_out_of_bounds_scores():
    raw_invalid_range = """
    {
      "criteria_scores": {
        "correctness": {"score": 10.0, "rationale": "Over maximum."},
        "faithfulness": {"score": -5.0, "rationale": "Under minimum."},
        "completeness": {"score": 3.0, "rationale": "Normal."},
        "instruction_following": {"score": 3.0, "rationale": "Normal."},
        "tone_safety": {"score": 3.0, "rationale": "Normal."}
      }
    }
    """
    verdict = parse_single_item_verdict(raw_invalid_range, question_id="test_bounds")

    # Correctness bounded to 5.0, Faithfulness bounded to 1.0, others 3.0 -> mean (5+1+3+3+3)/5 = 3.0
    assert verdict.criteria_scores["correctness"].score == 5.0
    assert verdict.criteria_scores["faithfulness"].score == 1.0
    assert verdict.overall_score == 3.0


def test_parse_ab_verdict_invalid_winner_fallback():
    raw_ab = """
    {
      "winner": "Candidate A is much better than Candidate B",
      "score_a": 4.0,
      "score_b": 2.0,
      "rationale": "A won."
    }
    """
    data = parse_ab_verdict(raw_ab, question_id="ab_fallback")
    assert data["winner"] in ["A", "B", "Tie"]
    assert data["score_a"] == 4.0
    assert data["score_b"] == 2.0
