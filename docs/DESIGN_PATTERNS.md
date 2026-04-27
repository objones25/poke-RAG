# Design Pattern Improvement Findings

> Generated 2026-04-27 by parallel `python-design-patterns` agent analysis.
> All findings are prioritised and fed into `docs/superpowers/plans/2026-04-27-design-patterns-refactor.md`.

---

## Summary

11 findings across 7 source files. 3 HIGH, 5 MEDIUM, 3 LOW.

| # | Priority | File | Issue | Pattern |
|---|----------|------|-------|---------|
| 1 | HIGH | `src/api/dependencies.py` | `build_pipeline` / `build_async_pipeline` duplicate ~220 lines | Builder / Extract Function |
| 2 | HIGH | `src/retrieval/chunker.py` | `chunk_file()` if/elif dispatch — adding a source requires touching two places | Strategy + Registry |
| 3 | HIGH | `src/retrieval/vector_store.py` | `QdrantVectorStore` / `AsyncQdrantVectorStore` share identical pure helper logic | Template Method helpers |
| 4 | MEDIUM | `src/retrieval/retriever.py` | `hasattr` duck-type check instead of typed Protocol | Protocol |
| 5 | MEDIUM | `src/generation/inference.py` | Input-preparation block copy-pasted across `infer()` / `stream_infer()` | Extract Method |
| 6 | MEDIUM | `src/api/dependencies.py` | `_build_cache()` pattern could be extended; cache construction is already a factory | Factory |
| 7 | MEDIUM | `src/retrieval/knowledge_refiner.py` | Constraint keyword set is hard-coded, blocking per-query extension | Registry |
| 8 | LOW | `src/retrieval/retriever.py` | `QueryRouter` conditional injected from `dependencies.py` — tight coupling | DI clarification |
| 9 | LOW | `src/api/app.py` | Exception → HTTP status mapping is inline `if/elif` | Exception Handler Registry |
| 10 | LOW | `src/types.py` | `RetrievalError` / `EmbeddingError` flat hierarchy; no way to catch "any retrieval-related error" | Exception Hierarchy |
| 11 | LOW | `src/pipeline/rag_pipeline.py` | Sync `RAGPipeline` and `AsyncRAGPipeline` share identical 7-step orchestration logic | Extract pure helpers |

---

## HIGH Priority

### H-1 · DRY in `dependencies.py`

`build_pipeline()` (lines 70–178) and `build_async_pipeline()` (lines 181–291) each build the same
embedder, reranker, generation config, model loader, inferencer, HyDE transformer, generator,
router, refiner, and cache.  They differ only in the Qdrant client type, vector-store type,
retriever type, and pipeline type.

**Fix:** extract `_build_shared_components(settings) -> _SharedComponents` dataclass covering the
ten shared objects. Each builder calls it then wires the three sync/async-specific objects.
Eliminates ~110 lines of duplication.

### H-2 · Strategy Registry in `chunker.py`

`chunk_file()` (lines 534–582) dispatches via `if source == "pokeapi" / "smogon" / "bulbapedia"`.
Adding a fourth source requires editing two functions and adding new top-level helpers.

**Fix:** extract three private file-chunker functions with a common signature
`(text, *, path, tokenize_fn, topic_lookup) -> list[RetrievedChunk]` and register them in a
module-level `_CHUNKERS: dict[Source, Callable]`. `chunk_file()` becomes a two-liner lookup.

### H-3 · Template Method helpers in `vector_store.py`

`QdrantVectorStore` and `AsyncQdrantVectorStore` contain identically-implemented helpers:
`_build_vectors_config`, `_build_point_structs`, `_build_prefetch`, `_parse_points_to_chunks`,
`_build_entity_filter`.  These pure functions do not touch `self` and are copy-pasted verbatim.

**Fix:** hoist them to module-level functions; both classes call them.  No base class needed.

---

## MEDIUM Priority

### M-4 · `hasattr` in `retriever.py`

`Retriever._embed_for_search()` and `AsyncRetriever._embed_for_search()` both use
`hasattr(self._query_transformer, "transform_to_embedding")` to detect `MultiDraftHyDETransformer`.
This is a stringly-typed duck-type check that bypasses the Protocol system.

**Fix:** add `FusedEmbeddingTransformerProtocol` to `src/retrieval/protocols.py`, decorated
`@runtime_checkable`.  Replace `hasattr` with `isinstance`.

### M-5 · Extract `_prepare_inputs()` in `inference.py`

`infer()` (lines 29–36) and `stream_infer()` (lines 93–99) both build `messages`, call
`apply_chat_template`, call `processor(text=..., return_tensors="pt").to(device)`.  Four lines
duplicated verbatim.

**Fix:** extract `_prepare_inputs(prompt) -> tuple[dict, int]`.

### M-6 · Cache Factory already exists — extend consistently

`_build_cache()` already follows the Factory pattern.  No structural change needed; flagged as
confirmation that the pattern is correct and the only cache-construction path.

### M-7 · Constraint Registry in `knowledge_refiner.py`

Generation/tier constraint keywords are hard-coded as regex patterns inside `KnowledgeRefiner`.
These cannot be extended without editing the class.

**Fix:** extract keyword sets into module-level `frozenset` constants (`_GEN_KEYWORDS`,
`_TIER_KEYWORDS`) so callers can inspect or extend them without subclassing.  Low-risk.

---

## LOW Priority

### L-8 · DI for `QueryRouter`

`QueryRouter` is constructed inside `_build_shared_components`.  It is stateless and its only
configurable surface is the keyword registry.  Current design is acceptable; noted for future
extension if custom routing rules per-deployment are needed.

### L-9 · Exception Handler Registry in `app.py`

HTTP error mapping is a short `if/elif` today.  Worth converting to a dict only if the handler
list grows beyond ~5 entries.

### L-10 · Exception Hierarchy in `types.py`

Consider `RAGError` base, then `RetrievalError(RAGError)` and `EmbeddingError(RAGError)`.
Callers that need to catch any RAG-related error can then use `except RAGError`.

### L-11 · Extract pure helpers from `rag_pipeline.py`

`RAGPipeline.query()` and `AsyncRAGPipeline.query()` run identical 7-step orchestration; only
`await` differs.  Extracting pure, synchronous helpers (`_resolve_chunks`, `_compute_confidence`,
`_build_result`) would reduce the shared logic and make both `query()` methods thinner.
Medium effort; consider after the HIGH items land.

---

## Status

| Finding | Planned | Implemented | Tests Pass |
|---------|---------|-------------|------------|
| H-1 DRY dependencies | ✓ | ✓ 2026-04-27 | ✓ |
| H-2 Chunker registry | ✓ | ✓ 2026-04-27 | ✓ |
| H-3 VectorStore helpers | ✓ | ✓ 2026-04-27 | ✓ |
| M-4 FusedEmbedding protocol | ✓ | ✓ 2026-04-27 | ✓ |
| M-5 _prepare_inputs | ✓ | ✓ 2026-04-27 | ✓ |
| M-7 Constraint constants | — | — | — |
| L-10 Exception hierarchy | — | — | — |
| L-11 Pipeline helpers | — | — | — |
