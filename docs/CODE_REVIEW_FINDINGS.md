# Code Review Findings

Three parallel agents (code quality, bug detection, security) reviewed the full repo on 2026-04-26.  
Branch at review time: `fix/refiner-gap-detection` (commit `a1842ba`).

Items marked **DONE** have been fixed. Remaining items are ordered by priority.

---

## Status Legend

- [ ] Open
- [x] Fixed

---

## CRITICAL

| #   | Source   | Issue                                                                                                                                                                                              | Location     | Status |
| --- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------ |
| B1  | Bug      | `/query/stream` calls `get_async_pipeline()` unconditionally — always crashes when `ASYNC_PIPELINE_ENABLED=false` (the default)                                                                    | `app.py:324` | [x]    |
| B2  | Bug      | Lifespan creates a redundant sync `QdrantClient` in async mode instead of reusing `async_qdrant_client` from `build_async_pipeline()`; `/stats` then wraps it in `asyncio.to_thread` unnecessarily | `app.py:179` | [x]    |
| S1  | Security | `/stats` endpoint completely unauthenticated when `STATS_API_KEY` env var is unset — exposes full collection schema                                                                                | `app.py:264` | [x]    |
| S2  | Security | CORS defaults to `allow_origins=["*"]` — should require explicit configuration in production                                                                                                       | `app.py:205` | [x]    |

---

## HIGH

| #   | Source   | Issue                                                                                                                                          | Location                             | Status |
| --- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------ |
| B3  | Bug      | Confidence score computed on `chunks[0]` assuming it is the top-ranked chunk, but `RerankerProtocol` does not guarantee descending order       | `rag_pipeline.py:84,152`             | [x]    |
| B4  | Bug      | `except* RetrievalError` re-raises only `eg.exceptions[0]`, silently dropping all other concurrent source failures                             | `retriever.py:375`                   | [x]    |
| B5  | Bug      | Entity-name filter silently retried without filter when it returns 0 results — caller receives unfiltered data with no indication              | `vector_store.py:273` (sync + async) | [x]    |
| B6  | Bug      | `os.environ["QDRANT_URL"]` raises a bare `KeyError` instead of using the `_parse_string_required()` pattern used by every other required field | `config.py:250`                      | [x]    |
| C1  | Quality  | `_sigmoid()` defined identically in both `rag_pipeline.py:22` and `retriever.py:25` — should live in `src/utils/math.py`                       | —                                    | [x]    |
| C2  | Quality  | Async `ensure_collections()` uses bare `raise exc` without exception chaining                                                                  | `vector_store.py:307,325`            | [x]    |
| C3  | Quality  | Sync `_query()` missing the `if p.payload is None` guard present in the async version                                                          | `vector_store.py:213`                | [x]    |
| C4  | Quality  | `async_qdrant_client.close()` is only called on the exception path of lifespan teardown; normal teardown leaves it open                        | `app.py:191-200`                     | [x]    |
| S3  | Security | Rate limiter uses FIFO eviction — attacker can cycle IPs to bypass per-IP limits                                                               | `app.py:38`                          | [x]    |
| S4  | Security | `X-Forwarded-For` not validated when behind an undeclared proxy; warns only if `TRUSTED_PROXY_COUNT` is explicitly set                         | `app.py:42`                          | [ ]    |
| S5  | Security | HSTS disabled by default (`HTTPS_ENABLED=false`) — must be enabled explicitly in production                                                    | `app.py:136`                         | [ ]    |
| S6  | Security | Query `max_length=2000` chars is too permissive for a RAG API; leaves more surface for prompt injection and increases tokenisation cost        | `models.py:15`                       | [ ]    |

---

## MEDIUM

| #   | Source   | Issue                                                                                                                           | Location                   | Status |
| --- | -------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ------ |
| B7  | Bug      | Reranker `strict=True` zip will raise `ValueError` if model returns fewer scores than input documents                           | `reranker.py:55`           | [x]    |
| B8  | Bug      | `RateLimitMiddleware` `OrderedDict` FIFO eviction enables IP-cycling bypass (overlap with S3)                                   | `app.py:112`               | [x]    |
| B9  | Bug      | No `strip_threshold` ordering validation in `Settings.from_env()`; invalid values surface only at `KnowledgeRefiner.__init__()` | `config.py:181`            | [x]    |
| C5  | Quality  | Single-character helper names `_w()`, `_p()`, `_prefix()` in `query_router.py` used across 500+ lines                           | `query_router.py:10-23`    | [x]    |
| C6  | Quality  | `top_k` not validated (positive, reasonable range) at the vector store boundary                                                 | `vector_store.py:243`      | [ ]    |
| C7  | Quality  | Dropped chunks discarded silently — no audit trail in `RefinementResult`                                                        | `knowledge_refiner.py:156` | [ ]    |
| S7  | Security | Entity name validated before NFKC normalisation — Unicode homograph characters can bypass the regex                             | `models.py:8,31`           | [ ]    |
| S8  | Security | Global exception handler logs full stack traces unconditionally; should gate on `DEBUG` level                                   | `app.py:251`               | [ ]    |
| S9  | Security | `qdrant_client` logger not suppressed — auth failures can leak Qdrant URL or credentials                                        | `logging.py:31`            | [ ]    |

---

## LOW

| #   | Source   | Issue                                                                                                                           | Location             | Status |
| --- | -------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------ |
| B10 | Bug      | `make_chunk` default `original_doc_id=f"doc_{chunk_index}"` never matches production ID format; can mask format-validation bugs | `conftest.py:48`     | [ ]    |
| C8  | Quality  | `_RERANKER_MAX_LENGTH = 512` has no comment explaining its origin                                                               | `reranker.py:15`     | [ ]    |
| C9  | Quality  | `import builtins` only used to qualify `builtins.TimeoutError`; built-in is already in scope                                    | `retriever.py:6`     | [x]    |
| S10 | Security | Qdrant connectivity not verified at startup — first request fails if Qdrant is down                                             | `dependencies.py:40` | [ ]    |
| S11 | Security | No audit logging of `entity_name` / `sources` per query — no access trail                                                       | `app.py:277`         | [ ]    |

---

## Positive Findings (no action required)

- Prompt injection mitigated: NFKC normalisation + control-character stripping in `prompt_builder.py`
- `hmac.compare_digest()` used for timing-safe token comparison
- Pydantic `SecretStr` for all API keys
- No hardcoded credentials found
- Rate limiting implemented per IP
- Security headers present (CSP, X-Frame-Options, Referrer-Policy, X-Content-Type-Options)
- SQL injection N/A — Qdrant payload filters used throughout
- All core dependencies up to date; no known CVEs flagged

---

## Fix Order

1. **B1, B2** — Streaming crash + wrong client type in async mode ← _in progress_
2. **B3** — Confidence score computed on unordered chunk
3. **C3, C4** — Payload null check + client close in lifespan
4. **B4** — ExceptionGroup drops concurrent failures
5. **B6** — Bare KeyError for missing QDRANT_URL
6. **C1** — Deduplicate `_sigmoid()`
7. **B5** — Silent entity-filter fallback
8. **B7** — Reranker strict-zip crash
9. **S1, S2** — Stats auth + CORS wildcard default
10. Remaining MEDIUM / LOW items
