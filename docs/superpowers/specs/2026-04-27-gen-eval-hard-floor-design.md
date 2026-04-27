# Generation Eval + Hard Retrieval Floor — Design Spec

**Date:** 2026-04-27
**Tasks:** Task 0 (generation eval harness) + Task 1 (hard retrieval floor)
**Status:** Approved

---

## Goals

- Task 0: Build a generation eval harness so changes to generation behavior (thinking mode, self-verification, prompt edits, LoRA on/off) can be measured against a baseline.
- Task 1: Add a hard floor on reranker scores so chunks that are semantically irrelevant (score < threshold) never ground an answer.

These two deliverables are developed together because Task 1's acceptance test ("hallucination_rate shouldn't go up") requires the Task 0 baseline to exist first.

---

## Overall Architecture

**Task 0** adds three new files under `scripts/eval/`:

```
scripts/eval/
  gen_questions.yaml        # 30–40 questions with reference answers + required claims
  gemini_judge.py           # thin Gemini client for claim-level scoring
  run_gen_eval.py           # eval runner: hits /query, judges, reports metrics
  gen_reference_answers.py  # one-shot helper: drafts reference answers via Gemini for human review
  baselines/                # auto-created; stores YYYY-MM-DD.json baseline snapshots
```

**Task 1** adds one field to `Settings` and two guarded checks in `rag_pipeline.py`. No new files.

---

## Task 0: Generation Eval Harness

### `gen_questions.yaml` Schema

```yaml
- id: el001                  # matches id in questions.yaml for cross-referencing
  query: "What is Pikachu's base Speed stat?"
  category: easy_lexical     # from existing taxonomy
  source_trust: pokeapi      # which source the answer must cite
  reference_answer: "Pikachu has a base Speed stat of 90."
  required_claims:
    - "Speed: 90"
  adversarial: false         # true for off-topic questions probing refusal_rate
```

**Question selection (30–40 total):**
- `easy_lexical`: 5–6 (sanity floor / baseline ceiling)
- `aggregation`: 6–8 (multi-chunk synthesis — where generation quality matters most)
- `comparative`: 5–6 (two entities, both chunks must rank)
- `multi_source`: 5–6 (facts spread across sources for same entity)
- `confusable`: 4–5 (entity discrimination)
- `adversarial`: 4–5 (known off-topic; expected: "context does not cover")

All IDs are drawn from existing `questions.yaml` so retrieval-eval and generation-eval results can be correlated.

### `gemini_judge.py`

Purpose-built Gemini client for eval only. Uses `gemini-3.1-flash-lite-preview`. Does not import from `scripts/training/` to avoid cross-directory coupling.

Three functions:

```python
def check_claim(answer: str, claim: str) -> bool:
    # Prompt: "Does this answer contain claim X? Reply yes or no."
    # Returns True/False parsed from first token of response.

def check_hallucination(answer: str, context: str) -> list[str]:
    # Prompt: "List factual claims in the answer NOT supported by the context,
    #          one per line. If all claims are supported, output exactly: SUPPORTED"
    # Returns [] if SUPPORTED, else list of unsupported claim strings.

@dataclass
class JudgeResult:
    claim_hits: list[bool]           # one per required_claim
    unsupported_claims: list[str]    # from hallucination check

def judge_answer(answer: str, claims: list[str], context: str) -> JudgeResult:
    # Calls check_claim for each claim + check_hallucination once.
    # One API call per claim (batching is unreliable for yes/no extraction).
```

### `run_gen_eval.py`

**Flow:**
1. Load `gen_questions.yaml`
2. For each question: `POST /query` → receive `answer` + `chunks` (context)
3. Call `judge_answer(answer, required_claims, context)`
4. Accumulate per-question results
5. Compute and print metrics

**Metrics:**
- `claim_recall` = claims_hit / total_claims (overall + per category)
- `hallucination_rate` = fraction of questions with ≥1 unsupported claim
- `refusal_rate` = fraction of questions where answer contains "context does not cover" (or similar)
- `pass@1` = claim_recall == 1.0 AND no unsupported claims, per question

**Baseline persistence:**
- `--save-baseline` flag writes results to `scripts/eval/baselines/YYYY-MM-DD.json`
- Subsequent runs print delta from most recent baseline when `--baseline` flag provided

**Usage:**
```bash
uv run python scripts/eval/run_gen_eval.py                        # run against live API
uv run python scripts/eval/run_gen_eval.py --save-baseline        # also save snapshot
uv run python scripts/eval/run_gen_eval.py --baseline 2026-04-27  # compare to baseline
```

The API must be running at `http://localhost:8000` (or `RAG_API_URL` env var).

### `gen_reference_answers.py`

One-shot helper, run once during setup:
```bash
uv run python scripts/eval/gen_reference_answers.py
```
Calls Gemini to draft `reference_answer` and `required_claims` for each question in `gen_questions.yaml`. Prints YAML to stdout for human review and paste-back. Not part of the eval loop.

### Testing

- Unit tests for `gemini_judge.py`: mock the Gemini API, assert `check_claim` parses "yes"/"no" correctly, assert `check_hallucination` handles "SUPPORTED" vs multi-line output.
- Unit test for `run_gen_eval.py`: mock `POST /query` and judge, assert metric computation (claim_recall formula, pass@1 logic).
- No GPU required; all mocked.

---

## Task 1: Hard Retrieval Floor

### `src/config.py`

Add to `Settings`:
```python
retrieval_hard_floor: float = -2.0
# env: RETRIEVAL_HARD_FLOOR
# Sigmoid of -2.0 ≈ 0.12. Any chunk scoring below this is treated as irrelevant.
# Set to a very negative number (e.g. -99.0) to effectively disable.
```

### `src/pipeline/rag_pipeline.py`

After rerank, before generation, in both `query()` and `aquery()`. Ordered after the existing "no documents" check:

```python
# Existing check (unchanged):
if not chunks:
    raise RetrievalError("Retrieval returned no documents for query")

# New check:
if max(c.score for c in chunks) < self._settings.retrieval_hard_floor:
    raise RetrievalError("All retrieved chunks below relevance floor")
```

The `if chunks` guard on the new check is implicit — it runs only after the existing check passes.

### API Layer

`src/api/app.py` already converts `RetrievalError` → HTTP 503. No change needed.

### Tests (TDD — write tests first)

**Unit tests (`tests/unit/test_rag_pipeline.py` or new file):**

1. Mock reranker returns chunks all scoring `-5.0` → `RAGPipeline.query()` raises `RetrievalError` with message containing "below relevance floor"
2. Chunks scoring `-1.5` (above default floor of `-2.0`) → no error, generation proceeds
3. `RETRIEVAL_HARD_FLOOR=-99.0` → no error even for chunks scoring `-5.0`
4. Single chunk at exactly `-2.0` → raises (boundary: floor is exclusive)

**Integration test (`tests/integration/`):**
- Query "what is the airspeed velocity of an unladen swallow" against live index → HTTP 503 response

**Acceptance (from task doc):**
- `hallucination_rate` from generation eval does not increase after Task 1 ships (measured against Task 0 baseline).

---

## Data Flow Summary

```
[gen_questions.yaml]
        │
        ▼
run_gen_eval.py ──POST /query──► RAGPipeline
        │                              │
        │                        [Task 1 floor check]
        │                              │
        │◄──── answer + chunks ────────┘
        │
        ▼
gemini_judge.py ──► Gemini API (gemini-3.1-flash-lite-preview)
        │
        ▼
  metrics printed + optionally saved to baselines/
```

---

## Non-Goals

- No streaming eval (streaming UX is evaluated manually).
- `gen_reference_answers.py` is not tested — it's a one-shot utility.
- Task 1 does not change refiner thresholds (`REFINER_LOWER_THRESHOLD`); the hard floor is a separate backstop.
- No UI or dashboard for results — stdout + JSON files only.
