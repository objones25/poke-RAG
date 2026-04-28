# Task 7 — Logging Instrumentation Design

**Date:** 2026-04-27
**Status:** Approved

---

## Goal

Add structured INFO/WARN logs across the retrieval pipeline so that HyDE fire rate, reranker truncation, chunk size anomalies, and per-query score distributions are visible in server logs after a day of traffic or an eval run. Extends original Task 7 scope with KnowledgeRefiner triage logging, motivated by the refiner silently dropping all chunks for high-confidence queries (e.g. "What is Pikachu's base Speed stat?").

---

## Approach

Surgical per-component additions. Each component already has `_LOG = logging.getLogger(__name__)` except `knowledge_refiner.py` (which gets one added). All new log lines use the `key=value` style consistent with the existing codebase pattern. No shared telemetry object — events are logged where they occur.

---

## Components

### 1 — HyDE fire rate (`src/retrieval/retriever.py`)

**Where:** Inside the two-pass block in `Retriever.retrieve` and `AsyncRetriever.retrieve`, replacing the existing human-readable INFO messages at the two decision points.

**Format:**
```
INFO  src.retrieval.retriever — hyde_fired=false top_raw_confidence=0.996 threshold=0.500
INFO  src.retrieval.retriever — hyde_fired=true  top_raw_confidence=0.312 threshold=0.500
```

**Existing lines to replace:**
```python
# skipping HyDE
_LOG.info(
    "Raw pass confidence %.3f >= threshold %.3f; skipping HyDE",
    top_confidence,
    threshold,
)
# running HyDE
_LOG.info(
    "Raw pass confidence %.3f < threshold %.3f; running HyDE pass",
    top_confidence,
    threshold,
)
```

**Replacement:**
```python
# skipping HyDE
_LOG.info(
    "hyde_fired=false top_raw_confidence=%.3f threshold=%.3f",
    top_confidence,
    threshold,
)
# running HyDE
_LOG.info(
    "hyde_fired=true top_raw_confidence=%.3f threshold=%.3f",
    top_confidence,
    threshold,
)
```

Both `Retriever` and `AsyncRetriever` have identical two-pass blocks; both get this change.

---

### 2 — Confidence distribution (`src/retrieval/retriever.py`)

**Where:** After each `reranker.rerank()` call that produces a final result. There are two exit points per retriever variant:
- The two-pass early return (after `raw_reranked`)
- The single-pass return (after `reranked`)

**Format:**
```
INFO  src.retrieval.retriever — score_distribution: top=4.21 min=-0.83 mean=1.94 n=5
```

**Code to insert** immediately before each `return RetrievalResult(...)` after a final rerank. The variable name differs by exit point:

- Two-pass early-return path: variable is `raw_reranked`
- Single-pass return path: variable is `reranked`

```python
# Two-pass early-return path (use raw_reranked):
if raw_reranked:
    scores = [c.score for c in raw_reranked]
    _LOG.info(
        "score_distribution: top=%.3f min=%.3f mean=%.3f n=%d",
        scores[0],
        scores[-1],
        sum(scores) / len(scores),
        len(scores),
    )

# Single-pass return path (use reranked):
if reranked:
    scores = [c.score for c in reranked]
    _LOG.info(
        "score_distribution: top=%.3f min=%.3f mean=%.3f n=%d",
        scores[0],
        scores[-1],
        sum(scores) / len(scores),
        len(scores),
    )
```

Note: both lists are sorted descending by score, so `scores[0]` is top and `scores[-1]` is min. Both retriever variants (`Retriever` and `AsyncRetriever`) get both insertions.

---

### 3 — Reranker truncation (`src/retrieval/reranker.py`)

**Where:** In `BGEReranker.rerank`, before the `compute_score` call.

**Tokenizer access:** `getattr(self._model, "tokenizer", None)`. If the tokenizer is present (production with a real `FlagReranker`), count pairs where the combined encoded length exceeds `_RERANKER_MAX_LENGTH`. If absent (mocked in tests), skip the count and emit no log line.

**Format** (only logged when truncated count > 0):
```
INFO  src.retrieval.reranker — reranker_truncated_pairs=3 of 75
```

**Code to insert** before `raw_scores = self._model.compute_score(...)`:
```python
tokenizer = getattr(self._model, "tokenizer", None)
if tokenizer is not None:
    truncated = sum(
        1
        for q, d in pairs
        if len(tokenizer.encode(q + " " + d)) > _RERANKER_MAX_LENGTH
    )
    if truncated:
        _LOG.info("reranker_truncated_pairs=%d of %d", truncated, len(pairs))
```

Silent (no log) when zero truncated pairs — avoids per-query noise on clean inputs.

---

### 4 — Chunk size warnings (`src/retrieval/chunker.py`)

**Where:** In `_recursive_split`, at the terminal `return [stripped]` case — when the text cannot be split further into paragraphs or sentences and is returned as a single chunk.

**When to warn:** Only when the single chunk exceeds `target_tokens`. If it fits, return silently (existing behavior).

**Format:**
```
WARN  src.retrieval.chunker — oversized_chunk: tokens≈680 target=400
```

`doc_id` is not available inside `_recursive_split` (private helper with no document context). Callers can correlate via log timestamp.

**Code to insert** before the final `return [stripped]`:
```python
approx = _approx_tokens(stripped, tokenize_fn=tokenize_fn)
if approx > target_tokens:
    _LOG.warning("oversized_chunk: tokens≈%d target=%d", approx, target_tokens)
```

---

### 5 — KnowledgeRefiner triage (`src/retrieval/knowledge_refiner.py`)

**Where:** Add `_LOG = logging.getLogger(__name__)` at module level (after the existing imports). Add three log lines in `refine()`.

**Three log lines:**

```python
# (1) after _triage() call
_LOG.info(
    "refiner_triage: accepted=%d uncertain=%d dropped=%d",
    len(accepted),
    len(uncertain),
    len(dropped),
)

# (2) after the strip-filter loop over accepted chunks
_LOG.info(
    "refiner_strips: pre_strip=%d survived=%d strip_dropped=%d",
    len(accepted),
    len(refined),       # chunks that survived strip filter (excludes uncertain, added later)
    len(accepted) - len(refined),
)

# (3) after refined.extend(uncertain)
_LOG.info(
    "refiner_output: chunks=%d gaps=%d",
    len(refined),
    len(gaps),
)
```

Note: line (2) must be inserted **before** `refined.extend(uncertain)` so `len(refined)` reflects only strip-filtered accepted chunks, not the uncertain ones yet appended.

---

## Testing

Tests live in `tests/unit/test_knowledge_refiner.py` (existing), `tests/unit/test_reranker.py` (existing), and `tests/unit/test_chunker.py` (existing). New tests assert log output using `pytest`'s `caplog` fixture.

Each log addition gets one unit test:

| Component | Test assertion |
|---|---|
| HyDE fire/skip | `"hyde_fired=true"` / `"hyde_fired=false"` appear in caplog at INFO |
| Score distribution | `"score_distribution:"` appears in caplog at INFO after retrieve |
| Reranker truncation | `"reranker_truncated_pairs="` appears when mock tokenizer returns long encoding |
| Chunker oversized | `"oversized_chunk:"` appears at WARN when sentence > target_tokens |
| Refiner triage | `"refiner_triage:"` appears; counts match partition sizes |
| Refiner strips | `"refiner_strips:"` appears; `pre_strip` == accepted count |
| Refiner output | `"refiner_output:"` appears; chunk count matches final refined list |

No new integration tests needed — the unit tests cover all observable log events.

---

## Out of Scope

- Log aggregation / metrics export (Prometheus, Datadog) — future work
- Per-chunk score logging (would be too verbose; aggregate stats are sufficient)
- `entity_filter_fallback` metric (mentioned in Code Critiques; separate task)
