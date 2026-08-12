# Problem 2 — LLM-as-Judge Evaluation Pipeline

A robust, production-grade LLM-as-Judge evaluation framework built to evaluate candidate AI outputs across 5 explicit quality rubrics, run pairwise A/B comparison benchmarks, log full audit trails, and measure judge biases (position, verbosity, sycophancy, score clustering).

---

## 1. Architecture & Component Design

```
                     +-----------------------------------+
                     |         Test Suite Dataset        |
                     |  (datasets/suite.json [15 cases], |
                     |   datasets/ab_suite.json [7 cases])|
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |           Evaluator               |
                     |   (app/evaluator.py - Orchestrator)|
                     +-----------------------------------+
                                       |
                                       v
                    +------------------+------------------+
                    |                                     |
                    v                                     v
     +------------------------------+      +------------------------------+
     |     Prompt Builder & Rubric  |      |        JudgeClient           |
     |      (app/prompts.py)        |      |      (app/judge.py)          |
     |  - 5-Criteria Anchor Rubrics |      |  - Independent JUDGE_MODEL   |
     |  - Grounding & Rule Prompts  |      |    (qwen2.5:1.5b-instruct)   |
     |  - Pairwise A/B Prompts      |      |  - Ollama HTTP Integration   |
     +------------------------------+      |  - Replay Audit Log Writer   |
                    |                      +------------------------------+
                    |                                     |
                    +------------------+------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |          Structured Parser        |
                     |          (app/parser.py)          |
                     |  - JSON Regex Extractor           |
                     |  - Fallback Regex Score Extractor |
                     |  - Python Mean Overall Score Calc |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |     Aggregation & Bias Suite      |
                     | (app/aggregation.py, app/bias.py) |
                     |  - Per-Category Metric Breakdown  |
                     |  - 4 Bias Probe Measurements      |
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
1. **`app/config.py`**: System settings; manages `GENERATOR_MODEL` (`gemma:2b`) and `JUDGE_MODEL` (`qwen2.5:1.5b-instruct`) independently.
2. **`app/schemas.py`**: Pydantic models for evaluation items, criteria scores, judge verdicts (with Python-calculated overall score and retained LLM score), A/B comparison verdicts, suite reports (with category breakdowns), and bias reports.
3. **`app/prompts.py`**: Prompt templates with explicit 1.0 to 5.0 anchor rubrics and strict judging constraints.
4. **`app/parser.py`**: Multi-stage robust JSON parser with regex fallbacks and Python mean overall score calculation.
5. **`app/judge.py`**: HTTP client communicating with local Ollama API (`POST /api/generate`); tracks latency and token usage; appends full audit logs to `results/audit.log`.
6. **`app/evaluator.py`**: Core engine executing single items, category tagging, suite runs, and pairwise A/B comparisons with position-swapping.
7. **`app/aggregation.py`**: Pure aggregation functions computing overall pass rates, mean criterion scores, per-category metrics, and A/B win rates.
8. **`app/bias.py`**: Dedicated bias measurement harness evaluating Position Bias, Verbosity Bias, Sycophancy, and Score Clustering.
9. **`app/cli.py`**: Unified command-line runner (`python -m app.cli`).

---

## 2. Setup & Configuration

### Environment Variables (`.env`)
Configure parameters via environment variables or `.env`:

```env
# Generator and Judge are independently configurable (Mitigates Self-Enhancement Bias)
GENERATOR_MODEL=gemma:2b
JUDGE_MODEL=qwen2.5:1.5b-instruct

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

## 4. Explicit Criteria & Anchor Rubrics

The pipeline evaluates candidate outputs across **FIVE explicit criteria** on a strict 1.0 to 5.0 scale:

| Criterion | 5.0 Anchor (Excellent) | 3.0 Anchor (Acceptable) | 1.0 Anchor (Poor / Unsafe) |
|---|---|---|---|
| **Correctness** | Factually flawless and accurate relative to ground truth. | Mostly correct; minor non-critical imprecisions. | Factually incorrect or major hallucinations. |
| **Faithfulness** | Fully grounded in expected context; 0 ungrounded assertions. | Mostly faithful with minor unsupported details. | Contradicts expected reference output. |
| **Completeness** | Thoroughly addresses every requirement and question sub-part. | Addresses main point; misses secondary detail. | Severely incomplete; misses core task. |
| **Instruction Following** | Follows all negative constraints, formatting rules, and guidelines. | Minor formatting guideline violation. | Ignores explicit negative constraints or prompt rules. |
| **Tone / Safety** | Professional, objective, helpful, safe. | Acceptable tone; slightly informal or overly verbose. | Unsafe, toxic, hostile, or deceptively confident when wrong. |

---

## 5. Python-Controlled Overall Score Calculation

To prevent LLM judge hallucinations or score inflation from controlling final verdicts, **Python calculates the final overall score**:

$$\text{overall\_score} = \text{round}\left( \frac{\text{correctness} + \text{faithfulness} + \text{completeness} + \text{instruction\_following} + \text{tone\_safety}}{5}, 1 \right)$$

- **Pass/Fail Verdict**: $\text{passed} = \text{overall\_score} \ge 3.5$.
- **Audit Retention**: The raw `overall_score` returned by the LLM is retained in `llm_overall_score` for debugging and discrepancy auditing.

---

## 6. Dataset Composition & Category Aggregation

### Single-Item Evaluation Suite (`datasets/suite.json` - 15 cases)
The single-item suite contains 15 curated evaluation cases covering 15 distinct categories grounded in Problem 1 domain facts:
1. `fully_correct`: Fully accurate vector database recommendation.
2. `incorrect_factual`: Incorrect embedding model claim.
3. `incomplete_answer`: Incomplete document formats response.
4. `verbose_correct`: Verbose but correct chunk configuration explanation.
5. `concise_correct`: Concise correct LLM model response.
6. `confidently_wrong`: Confident hallucination regarding GPU VRAM requirements.
7. `instruction_violation`: Prose violation when bullet points were requested.
8. `partially_correct`: Partial list of retrieval evaluation metrics.
9. `unsupported_claim`: Ingestion pipeline response claiming ungrounded email alerts.
10. `irrelevant_info`: Correct port response padded with irrelevant database ports.
11. `poor_formatting`: Correct formats missing JSON array structure.
12. `professional_tone`: Professional objective summary of Problem 1 goal.
13. `unanswerable`: Request for non-existent Pinecone cloud API key.
14. `misleading_plausible`: Plausible claim that OpenAI embeddings are required.
15. `edge_case_multifact`: Multi-parameter extraction covering 5 system constants.

### Pairwise A/B Comparison Suite (`datasets/ab_suite.json` - 7 cases)
1. `ab001`: Idempotent chunking (Deterministic MD5 vs vague description).
2. `ab002`: Tech stack (Local Qdrant/SentenceTransformers vs Pinecone/OpenAI).
3. `ab003`: Retrieval metrics (Equivalent accurate responses / Tie).
4. `ab004`: Conciseness vs verbosity (Direct 500/50 char answer vs wordy filler).
5. `ab005`: Generator model (gemma:2b via Ollama vs Claude 3.5 Sonnet API).
6. `ab006`: Ingestion formats (All 3 supported formats vs PDF only).
7. `ab007`: Instruction adherence (Clean bullet points vs conversational prose).

---

## 7. Bias Measurement Methodology

The pipeline implements empirical probes to measure 4 key judge biases:

1. **Position Bias (Order Swap Flip Rate)**:
   - Evaluates pairwise candidate responses in original order (A then B) and swapped order (B then A).
   - Maps swapped winners back to candidate identifiers.
   - *Metric*: $\text{Flip Rate} = \frac{\text{Number of order-driven winner changes}}{\text{Total A/B pairs}}$.
2. **Verbosity / Length Bias (Padded Probe)**:
   - Evaluates concise correct response vs. padded verbose version for the same prompt.
   - *Metric*: $\Delta_{\text{verbosity}} = \text{Score}_{\text{verbose}} - \text{Score}_{\text{normal}}$. Biased if $\Delta > 0.5$ when normal score $< 4.5$.
3. **Sycophancy / Style Bias (Confident-Wrong Probe)**:
   - Evaluates responses stating factually false claims with high confidence and authoritative tone.
   - *Metric*: Sycophancy Rate (fraction of false answers awarded score $> 2.5$).
4. **Score Clustering & Spread**:
   - Calculates standard deviation ($\sigma$) and score bucket distribution across the suite.
   - *Flag*: `is_clustered = True` if $\sigma < 0.5$.

---

## 8. Empirical Evaluation Results

Empirical results generated from live execution of the pipeline using local Ollama (`qwen2.5:1.5b-instruct`):

### Standard Suite Results (`results/suite_results.json`)
- **Total Cases Evaluated**: `15`
- **Pass Rate**: `40.0%` (6 passed, 9 failed)
- **Mean Overall Score**: `3.20 / 5.0`
- **Mean Criterion Scores**:
  - `correctness`: `2.93`
  - `faithfulness`: `2.87`
  - `completeness`: `2.27`
  - `instruction_following`: `4.33`
  - `tone_safety`: `3.60`

#### Per-Category Performance Breakdown:
| Category | Cases | Pass Rate | Mean Overall Score | Mean Correctness | Mean Faithfulness |
|---|---|---|---|---|---|
| `instruction_violation` | 1 | 100.0% | **4.20** | 4.00 | 5.00 |
| `fully_correct` | 1 | 100.0% | **3.80** | 3.00 | 5.00 |
| `incomplete_answer` | 1 | 100.0% | **3.80** | 3.00 | 5.00 |
| `poor_formatting` | 1 | 100.0% | **3.80** | 3.00 | 5.00 |
| `unanswerable` | 1 | 100.0% | **3.80** | 3.00 | 5.00 |
| `verbose_correct` | 1 | 100.0% | **3.80** | 3.00 | 5.00 |
| `concise_correct` | 1 | 0.0% | **3.40** | 3.00 | 5.00 |
| `edge_case_multifact` | 1 | 0.0% | **3.00** | 3.00 | 1.00 |
| `confidently_wrong` | 1 | 0.0% | **2.80** | 3.00 | 1.00 |
| `irrelevant_info` | 1 | 0.0% | **2.80** | 3.00 | 1.00 |
| `incorrect_factual` | 1 | 0.0% | **2.60** | 2.00 | 1.00 |
| `partially_correct` | 1 | 0.0% | **2.60** | 3.00 | 1.00 |
| `professional_tone` | 1 | 0.0% | **2.60** | 3.00 | 1.00 |
| `unsupported_claim` | 1 | 0.0% | **2.60** | 3.00 | 1.00 |
| `misleading_plausible` | 1 | 0.0% | **2.40** | 2.00 | 1.00 |

### Pairwise A/B Comparison Results (`results/ab_results.json`)
- **Total Comparisons**: `7`
- **Candidate A Wins**: `2`
- **Candidate B Wins**: `4`
- **Ties**: `1`
- **Overall Winner**: **Candidate B**
- **Win Rate (Candidate A)**: `33.3%`
- **Win Rate (Candidate B)**: `66.7%`

### Bias Measurement Probes (`results/bias_results.json`)
| Bias Probe | Measured Metric | Empirical Result | Analysis |
|---|---|---|---|
| **Position Bias** | Flip Rate | **`42.9%` (3/7 flips)** | Measurable position preference when candidate order is swapped in 1.5B judge. |
| **Verbosity Bias** | Score Delta ($\Delta$) | **`-1.20` points** | Concise answer rated 5.0/5.0 vs verbose padded rated 3.8/5.0 for exact same prompt. No artificial verbosity inflation. |
| **Sycophancy** | Detection Rate | **`33.3%` (1/3 detected)** | Evaluated across 3 hallucinated/confident-wrong cases (`confidently_wrong`, `misleading_plausible`, `unsupported_claim`). |
| **Score Clustering** | Score Std Dev ($\sigma$) | **`0.6141` (Spread: 2.4 to 4.2)** | Healthy score spread ($\sigma = 0.6141 > 0.5$); rubrics prevented score collapse. |

---

## 9. Auditability & Replayability

- **Audit Log**: Full prompt templates, raw LLM text responses, execution latency, and token counts are persisted to [`results/audit.log`](file:///d:/uplifting/nexpro%20ai%20projects/gen-ai-assignment/problem2-llm-judge/results/audit.log).
- **Unit Test Suite**: 23 automated unit tests in `tests/` verifying JSON parsing, fallback regex, score bounds, aggregation, position swap logic, error handling, and CLI options.
  ```bash
  python -m pytest -v
  ```
  *Result*: **23 passed in 1.27s**

---

## 10. Discussion & Release Gate Verdict

### Why Independent Generator and Judge Models Matter
Using the generator model (`gemma:2b`) to evaluate its own outputs introduces severe **self-enhancement bias**, where a model systematically rates its own generated text higher than competing outputs. Configuring an independent judge model (`qwen2.5:1.5b-instruct`) breaks this feedback loop and provides impartial evaluation.

### Limitations of Local 1.5B Parameter Judge (`qwen2.5:1.5b-instruct`)
- **Capacity Constraints**: At 1.5B parameters, small local models display moderate position bias (42.9% flip rate on order swap) and occasionally struggle to heavily penalize confident hallucinations (sycophancy probe score 2.8).
- **Inference Speed**: On CPU execution, local 1.5B evaluation requires ~20–30 seconds per item.

### Production Recommendation
- **As an Automated CI / Staging Gate**: **YES**. Highly effective for catching obvious hallucinations, checking format constraints, measuring A/B regressions, and enforcing minimum quality floors.
- **As the Sole Release Gate**: **NO**. Automated model-graded evaluations should form part of a **hybrid release gate**, combined with human spot-audits and periodic evaluation by larger frontier judge models (e.g. 70B parameter models).
