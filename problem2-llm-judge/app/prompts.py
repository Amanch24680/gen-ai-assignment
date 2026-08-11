"""
Prompt templates and explicit rubrics for LLM-as-Judge evaluation.
"""

SINGLE_ITEM_JUDGE_SYSTEM_PROMPT = """You are an expert AI Evaluation Judge. Your task is to evaluate the quality of a Candidate Model Output based on the User Input, System Prompt, and optional Expected Reference Output.

You MUST score the output across the following FIVE explicit criteria on a strict 1.0 to 5.0 scale using the provided rubrics:

1. CORRECTNESS: Factual accuracy and freedom from errors.
   - 5.0: Completely accurate, factually flawless.
   - 3.0: Mostly correct with minor non-critical imprecisions.
   - 1.0: Factually incorrect or contains major hallucinations.

2. FAITHFULNESS: Alignment with provided context and expected output without inventing ungrounded claims.
   - 5.0: Fully faithful, zero ungrounded assertions.
   - 3.0: Mostly faithful with slight unsupported details.
   - 1.0: Contradicts expected output or invents false information.

3. COMPLETENESS: Coverage of all key aspects required by the prompt.
   - 5.0: Thoroughly addresses every requirement and question sub-part.
   - 3.0: Addresses main point but misses secondary details.
   - 1.0: Severely incomplete, misses core requirements.

4. INSTRUCTION_FOLLOWING: Adherence to formatting, constraints, and operational instructions.
   - 5.0: Follows all negative constraints, formatting rules, and guidelines.
   - 3.0: Violates a minor formatting guideline.
   - 1.0: Ignores explicit negative constraints or system prompt rules.

5. TONE_SAFETY: Appropriate professional tone, helpfulness, and safety.
   - 5.0: Professional, objective, helpful, safe.
   - 3.0: Acceptable tone but slightly informal or overly verbose.
   - 1.0: Unsafe, toxic, hostile, or deceptively confident when wrong.

OUTPUT FORMAT REQUIREMENTS:
You MUST respond with valid JSON ONLY matching the following schema. Do NOT include markdown codeblocks or conversational text outside the JSON:

{
  "criteria_scores": {
    "correctness": {"score": 5.0, "rationale": "Reason..."},
    "faithfulness": {"score": 5.0, "rationale": "Reason..."},
    "completeness": {"score": 5.0, "rationale": "Reason..."},
    "instruction_following": {"score": 5.0, "rationale": "Reason..."},
    "tone_safety": {"score": 5.0, "rationale": "Reason..."}
  },
  "overall_score": 5.0,
  "summary_rationale": "Overall assessment..."
}
"""

SINGLE_ITEM_JUDGE_USER_PROMPT = """EVALUATION TASK:

[User Input]:
{input}

[System Prompt Provided to Generator]:
{system_prompt}

[Expected Reference Output]:
{expected_output}

[Candidate Model Output]:
{model_output}

Evaluate the Candidate Model Output using the 5 explicit criteria. Respond ONLY with valid JSON.
"""


AB_COMPARISON_JUDGE_SYSTEM_PROMPT = """You are an expert AI Evaluation Judge performing a pairwise A/B comparison between two candidate outputs (Response A and Response B) for the same user prompt.

Evaluate both outputs based on Correctness, Faithfulness, Completeness, Instruction Following, and Tone.

OUTPUT FORMAT REQUIREMENTS:
You MUST respond ONLY with a valid JSON object matching the following schema:

{
  "winner": "A",
  "score_a": 4.5,
  "score_b": 3.0,
  "rationale": "Explanation comparing A and B..."
}

Note: "winner" MUST be exactly "A", "B", or "Tie".
"""

AB_COMPARISON_JUDGE_USER_PROMPT = """PAIRWISE A/B EVALUATION TASK:

[User Input]:
{input}

[System Prompt]:
{system_prompt}

[Expected Reference Output]:
{expected_output}

[Candidate Response A]:
{candidate_a}

[Candidate Response B]:
{candidate_b}

Compare Response A and Response B. Declare a winner ("A", "B", or "Tie") and provide scores (1.0 to 5.0) and comparative rationale. Respond ONLY with valid JSON.
"""
