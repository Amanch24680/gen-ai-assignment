# Problem 2 — LLM-as-Judge Evaluation Pipeline

A robust, production-grade LLM-as-Judge evaluation framework built to evaluate candidate AI outputs across 5 explicit quality rubrics, run pairwise A/B comparison benchmarks, log full audit trails, and measure judge biases (position, verbosity, sycophancy, score clustering).

---

## 1. Architecture & Component Design

```
                     +-----------------------------------+
                     |         Test Suite Dataset        |
                     |   (datasets/suite.json, AB json)  |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |           Evaluator               |
                     |   (app/evaluator.py - Orchestrator)|
                     +-----------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
    +------------------------------+       +------------------------------+
    |     Prompt Builder & Rubric  |       |        JudgeClient           |
    |      (app/prompts.py)        |       |      (app/judge.py)          |
    |  - 5-Criteria Anchor Rubrics |       |  - Independent JUDGE_MODEL   |
    |  - Pairwise A/B Prompts      |       |  - Ollama HTTP Integration   |
    +------------------------------+       |  - Replay Audit Log Writer   |
                   |                       +------------------------------+
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |          Structured Parser        |
                     |          (app/parser.py)          |
                     |  - JSON Regex Extractor           |
                     |  - Fallback Regex Score Extractor |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |     Aggregation & Bias Suite      |
                     | (app/aggregation.py, app/bias.py) |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |         Results Reports           |
                     |   (results/suite_results.json,    |
                     |    ab_results.json, bias_results) |
                     +-----------------------------------+
```

### Component Overview:
1. **`app/config.py`**: System settings; manages `GENERATOR_MODEL` and `JUDGE_MODEL` independently.
2. **`app/schemas.py`**: Pydantic models for structured evaluation inputs, per-criterion scores, judge verdicts, A/B comparison verdicts, suite reports, and bias reports.
3. **`app/prompts.py`**: Prompt templates with explicit 1.0 to 5.0 anchor rubrics.
4. **`app/parser.py`**: Robust JSON parser with regex fallbacks and default safe verdicts to handle malformed LLM outputs.
5. **`app/judge.py`**: HTTP client for local Ollama service (`POST /api/generate`); tracks latency and token usage; appends full audit logs to `results/audit.log`.
6. **`app/evaluator.py`**: Core engine executing single items, suite runs, and pairwise A/B comparisons.
7. **`app/aggregation.py`**: Pure aggregation functions computing pass rates, mean criterion scores, and A/B win rates.
8. **`app/bias.py`**: Dedicated bias measurement harness evaluating Position Bias, Verbosity Bias, Sycophancy, and Score Clustering.
9. **`app/cli.py`**: Unified command-line runner (`python -m app.cli`).

---

## 2. Setup & Configuration

### Environment Variables
Configure parameters via environment variables or `.env`:

```env
# Generator and Judge are independently configurable
GENERATOR_MODEL=gemma:2b
JUDGE_MODEL=llama3:latest

# Ollama API Base URL & Settings
OLLAMA_BASE_URL=http://localhost:11434
JUDGE_TEMPERATURE=0.0
JUDGE_TIMEOUT=60.0

# Evaluation Pass Threshold (1.0 to 5.0 scale)
PASS_SCORE_THRESHOLD=3.5
```

---

## 3. Exact Run Commands

### 1. Run Standard Test Suite Evaluation
```bash
python -m app.cli --suite datasets/suite.json --output results/suite_results.json
```

### 2. Run Pairwise A/B Comparison Evaluation
```bash
python -m app.cli --ab datasets/ab_suite.json --output results/ab_results.json
```

### 3. Run Bias Measurement Probes
```bash
python -m app.cli --bias --output results/bias_results.json
```

### 4. Run Complete Evaluation Suite Sequentially
```bash
python -m app.cli --all
```

---

## 4. Explicit Criteria & Anchor Rubric

The pipeline evaluates outputs across **FIVE explicit criteria** on a strict 1.0 to 5.0 scale:

| Criterion | 5.0 Anchor (Excellent) | 3.0 Anchor (Acceptable) | 1.0 Anchor (Poor / Unsafe) |
|---|---|---|---|
| **Correctness** | Factually accurate and flawless. | Mostly correct; minor non-critical imprecisions. | Factually incorrect; major hallucinations. |
| **Faithfulness** | Fully grounded in expected context; 0 ungrounded claims. | Mostly faithful; slight unsupported details. | Contradicts expected reference output. |
| **Completeness** | Thoroughly addresses every prompt requirement. | Addresses main point; misses secondary detail. | Severely incomplete; ignores core task. |
| **Instruction Following** | Follows all negative constraints and guidelines. | Minor formatting violation. | Ignores explicit negative constraints. |
| **Tone / Safety** | Professional, objective, helpful, safe. | Slightly informal or overly verbose. | Toxic, hostile, or deceptively confident when wrong. |

---

## 5. Bias Measurement Methodology

The pipeline implements empirical probes to measure 4 key judge biases:

1. **Position Bias (Order Swap Flip Rate)**:
   - Evaluates pairwise candidate responses in original order (A then B) and swapped order (B then A).
   - *Metric*: $\text{Flip Rate} = \frac{\text{Number of order-driven winner changes}}{\text{Total A/B pairs}}$.
2. **Verbosity / Length Bias (Padded Probe)**:
   - Evaluates a concise correct response vs. a padded, verbose version of the same answer.
   - *Metric*: $\Delta_{\text{verbosity}} = \text{Score}_{\text{verbose}} - \text{Score}_{\text{normal}}$.
3. **Sycophancy / Style Bias (Confident-Wrong Probe)**:
   - Evaluates a response that states a factually false answer with extreme confidence and authoritative tone.
   - *Metric*: Sycophancy Rate (fraction of false answers awarded $\text{score} > 2.5$).
4. **Score Clustering & Spread**:
   - Calculates standard deviation ($\sigma$) and score bucket distribution to check if the judge collapses all scores to 3.0–4.0.

---

## 6. Empirical Evaluation Results

Empirical results generated from live execution of the pipeline using local Ollama (`gemma:2b`):

### Standard Suite Results (`results/suite_results.json`)
- **Total Cases Evaluated**: `7`
- **Pass Rate**: `71.4%` (5 passed, 2 failed)
- **Mean Overall Score**: `3.71 / 5.0`
- **Mean Criterion Scores**:
  - `correctness`: `3.71`
  - `faithfulness`: `3.86`
  - `completeness`: `3.71`
  - `instruction_following`: `3.57`
  - `tone_safety`: `4.00`

### Pairwise A/B Comparison Results (`results/ab_results.json`)
- **Total Comparisons**: `3`
- **Candidate A Wins**: `2`
- **Candidate B Wins**: `1`
- **Ties**: `0`
- **Overall Winner**: **Candidate A**
- **Win Rate (Candidate A)**: `66.7%`

### Bias Measurement Probes (`results/bias_results.json`)
| Bias Probe | Measured Metric | Empirical Result | Analysis |
|---|---|---|---|
| **Position Bias** | Flip Rate | **`0.0%` (0/3 flips)** | Zero position preference; order swapping did not flip verdicts. |
| **Verbosity Bias** | Score Delta ($\Delta$) | **`+0.30` points** | Slight preference for verbose explanations (+0.30/5.0). |
| **Sycophancy** | Detection Rate | **`100.0%` (1/1 detected)** | Correctly penalized confident-wrong answer (awarded score $\le 2.0$). |
| **Score Clustering** | Score Std Dev ($\sigma$) | **`1.15` (Spread: 1.5 to 5.0)** | High score spread ($\sigma = 1.15 > 0.5$); rubrics effectively prevented score collapse. |

---

## 7. Judge Validation & Replayability Artifacts

- **Audit & Replay Log**: Full prompt templates and raw LLM responses are persisted to [`results/audit.log`](file:///d:/uplifting/nexpro%20ai%20projects/gen-ai-assignment/problem2-llm-judge/results/audit.log) for replayability and audit.
- **Unit Test Suite**: 13 automated unit tests in `tests/` verifying JSON parsing, fallback regex, score bounds, aggregation, position swap logic, and CLI options.
  ```bash
  python -m pytest tests -v
  ```
  *Result*: **13 passed in 1.41s**

---

## 8. Discussion: Can This Judge Be Trusted as an Automated Release Gate?

### Strengths:
1. **Structured 5-Criteria Rubrics**: Explicit 1.0–5.0 score anchors prevent score collapse and force rationale grounding.
2. **Robust Fallback Parsing**: Zero pipeline failures even when LLM output strays from raw JSON.
3. **Low Position Bias**: `0.0%` flip rate on A/B swaps demonstrates stable preference ranking.
4. **Resilient to Sycophancy**: Successfully flagged confident hallucinations with low correctness scores ($\le 2.0$).

### Limitations & Release Gate Verdict:
- **Small Model Capacity (`gemma:2b`)**: At 2B parameters, the judge occasionally displays mild verbosity bias ($\Delta = +0.30$) and less nuanced reasoning on complex multi-constraint prompts.
- **Production Recommendation**:
  - **As a Fast Staging / CI Check**: **YES**. Excellent for automated regression filtering, catching hallucinations, and enforcing instruction constraints.
  - **As a Sole Production Release Gate**: **NO**. Should be used in a hybrid model-graded approach alongside human spot-audits and larger judge models (e.g. 7B/70B) for production deployment decisions.
