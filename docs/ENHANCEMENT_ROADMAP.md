# Enhancement Roadmap

> Last updated: 2026-04-24

## Async Retrieval + Caching — Planning Notes

Both the async retrieval refactor and Redis-backed query caching are load-bearing architectural changes that must be planned carefully before implementation:

- **Async retrieval** requires converting `Retriever.retrieve()`, `RAGPipeline.query()`, and the Qdrant client to async throughout — this changes the contract for every caller including the API layer and tests.
- **Redis caching** introduces a new external dependency and requires a cache invalidation strategy (the vector index can be rebuilt at any time).

Both items are P0 in the roadmap. Implement async retrieval first; caching layers on top cleanly once the async base is in place.

---

## P0 — Critical Path

### 1. Async Retrieval Pipeline

**File(s):** `src/retrieval/retriever.py`, `src/pipeline/rag_pipeline.py`, `src/api/app.py`  
**Effort:** L | **Benefit:** 40–60% latency reduction for multi-source retrieval

Replace the `ThreadPoolExecutor` in `_run_search()` with `asyncio` and the async `QdrantClient`. Convert `Retriever.retrieve()` and `RAGPipeline.query()` to `async def`. The API layer already wraps `pipeline.query()` in `asyncio.to_thread()` — remove that wrapper once the pipeline is natively async.

**Risk:** Concurrency bugs at the protocol boundary; full integration test pass required before shipping.  
**Dependency:** qdrant-client ≥ 1.17.1 (already pinned) exposes async client.

---

### 2. Semantic Query Caching

**File(s):** new `src/retrieval/cache.py`, `src/retrieval/protocols.py`, `src/config.py`  
**Effort:** M | **Benefit:** 90%+ speedup on repeated queries; ~20–30% of queries are repeats

Two-level cache: embedding cache in `Retriever` (keyed on normalized query text) and retrieval result cache in `RAGPipeline` (keyed on query + source list hash).

Start with in-process `LRUCache` (`cachetools`); add Redis backing behind the same protocol once the LRU is proven. New env vars: `CACHE_ENABLED`, `CACHE_TTL_SECONDS`.

**Risk:** Stale results if the vector index is rebuilt; add `/cache/invalidate` admin endpoint or TTL-only strategy.  
**Dependency:** Implement after async retrieval (#1).

---

### 3. ColBERT Multi-Vector (Lazy Enable)

**File(s):** `src/retrieval/embedder.py`, `src/retrieval/vector_store.py`, `src/config.py`  
**Effort:** M | **Benefit:** 5–10% recall improvement

`BGEEmbedder.encode()` hardcodes `return_colbert_vecs=False`. Add `COLBERT_ENABLED=true/false` (default: false). When enabled, populate the optional `colbert_vecs` field on `EmbeddingOutput` and include ColBERT as a third `Prefetch` modality in Qdrant RRF fusion.

**Risk:** ~15–20% embedding cost increase; test recall improvement on eval set before enabling by default.

---

## P1 — High Value

### 4. Retrieval Diversity Deduplication

**File(s):** `src/retrieval/retriever.py`  
**Effort:** S | **Benefit:** Cleaner generator context; fewer near-duplicate chunks

Post-retrieval pass: pairwise cosine similarity on dense vectors; discard chunks with similarity > threshold (default 0.95). New env var: `DEDUP_SIMILARITY_THRESHOLD`.

---

### 5. Streaming Generation (SSE)

**File(s):** `src/generation/inference.py`, `src/api/app.py`  
**Effort:** M | **Benefit:** 50% perceived latency improvement

Add `/query-stream` endpoint returning `StreamingResponse` with SSE. Modify `Inferencer` to yield tokens via a generator instead of collecting all output IDs first.

---

### 6. `/explain` Endpoint

**File(s):** `src/api/app.py`, `src/api/models.py`  
**Effort:** S | **Benefit:** Transparency, hallucination spotting

Return answer + retrieved chunks + confidence score in one response. No logic change — `PipelineResult` already carries this data. Expose it via a new response model.

---

### 7. Per-Query Reranker Toggle

**File(s):** `src/api/models.py`, `src/retrieval/retriever.py`  
**Effort:** S | **Benefit:** 40–50% latency reduction for latency-sensitive clients

Add `enable_reranking: bool = True` to `QueryRequest`. Pass through to `Retriever.retrieve()`. When false, return top-k from raw RRF fusion, skip reranker.

---

### 8. Feedback-Driven DPO Fine-Tuning Loop

**File(s):** new `src/api/app.py` `/feedback` endpoint, `scripts/training/`  
**Effort:** L | **Benefit:** 8–12% answer quality improvement over time

Collect thumbs-up/down via `/feedback`. Weekly batch job generates DPO pairs from approved vs. rejected answers. Fine-tune with TRL `DPOTrainer` (already in `train` group). Quality gate (see #13) required before deploying new adapters.

**Risk:** Noisy labels degrade model; require confidence > 0.7 before including in training data.  
**Dependency:** #13 (post-fine-tune quality gate).

---

## P2 — Future

### 9. Automated RAGAS Eval Harness

**File(s):** new `tests/eval/test_rag_quality.py`  
**Effort:** M | **Benefit:** Objective regression detection for retrieval + generation quality

Golden Q&A dataset (50–100 pairs). Run after each deployment; fail if faithfulness drops > 5%.  
**Cost:** ~$5–10/run (Gemini judge). Use fixed eval set for reproducibility.

---

### 10. Prometheus `/metrics` Endpoint

**File(s):** new `src/api/metrics.py`, `src/api/app.py`  
**Effort:** S | **Benefit:** Query counts, latency histograms, confidence distribution for monitoring

Add `prometheus_client` to `api` group. Expose counters and histograms at `/metrics`.

---

### 11. Structured JSON Logging (OpenTelemetry-style)

**File(s):** `src/utils/logging.py`, call sites in retriever, pipeline, generation  
**Effort:** M | **Benefit:** Trace-level debugging; per-span latency breakdowns

Add `structlog`. Emit `query_start/end`, `retrieval_start/end`, `generation_start/end` events with latency and key metadata.

---

### 12. Bulk `/batch-query` Endpoint

**File(s):** `src/api/app.py`, `src/api/models.py`  
**Effort:** S | **Benefit:** 5–10× faster eval runs

Accept up to `BATCH_QUERY_MAX_SIZE` queries (default 50). Use `asyncio.gather()` to parallelize. Rate limiting applies per query.  
**Dependency:** Requires async pipeline (#1).

---

### 13. Post-Fine-Tune Quality Gate

**File(s):** `scripts/training/`  
**Effort:** M | **Benefit:** Prevents accidental quality regressions when deploying new LoRA adapters

Run RAGAS eval (#9) on base + fine-tuned model after each training run. Reject adapter if faithfulness drops > 5%.

---

### 14. HyDE Multi-Draft — Finalize

**File(s):** `src/retrieval/query_transformer.py`, `src/retrieval/retriever.py`  
**Effort:** S | **Benefit:** 3–5% recall improvement

`MultiDraftHyDETransformer` already exists. Wire `Retriever.retrieve()` to call `transform_to_embedding()` when `HYDE_NUM_DRAFTS > 1`. Recommended default in prod: `HYDE_NUM_DRAFTS=3`.

---

### 15. SFT Data Generation CI/CD

**File(s):** `scripts/training/generate_sft_data.py`, new GitHub Actions workflow  
**Effort:** M | **Benefit:** Continuous fine-tuning data generation without manual runs

Weekly scheduled job: sample new chunks from `processed/`, call Gemini API, append to `sft_data.jsonl`, trigger training when threshold reached (e.g., 10k pairs).  
**Cost:** ~$0.10–0.30/1k pairs.

---

### 16. Query Routing ML Classifier

**File(s):** `src/retrieval/query_router.py`  
**Effort:** L | **Benefit:** Better routing accuracy on edge cases the regex patterns miss

Replace hand-crafted regex with a learned logistic regression or tiny BERT classifier. Fallback to all-sources when confidence < 0.6 (current default behavior). Train on labeled queries from the feedback loop (#8).  
**Timeline:** Post-v1; lower priority than all other items.

---

## Dependency Graph

```
#1 Async Retrieval
  └─► #2 Semantic Caching
        └─► #12 Bulk Query
#8 DPO Loop
  └─► #13 Quality Gate
        └─► #15 SFT CI/CD
#9 RAGAS Eval
  └─► #13 Quality Gate
```

## Effort Key

| Label | Range    |
| ----- | -------- |
| S     | < 1 day  |
| M     | 1–3 days |
| L     | 3–7 days |
| XL    | > 1 week |
