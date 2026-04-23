# CLAUDE.md

## What this project is

Agentic RAG system for Pokémon knowledge, backed by `google/gemma-4-E4B-it`. Queries hit a vector index built from three sources (Bulbapedia, PokéAPI, Smogon), retrieve grounded context, and pass it to the model for generation. Optional LoRA fine-tuning on RunPod. Eventually served as an HTTP API.

## Commands

```bash
uv sync --all-extras          # install all deps including dev/train
uv run pytest                 # all tests
uv run pytest -m "not gpu"    # skip GPU tests (default for local dev)
uv run pytest tests/unit/     # unit tests only
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src/              # type check
uv run python scripts/build_index.py   # embed + index processed/ data
```

## Dependencies

Managed via `uv` and `pyproject.toml`. Groups:

| Group | Key packages                                                                                 |
| ----- | -------------------------------------------------------------------------------------------- |
| core  | `transformers`, `torch`, `accelerate`, `FlagEmbedding`, `qdrant-client`, `pydantic`, `numpy` |
| api   | `fastapi`, `uvicorn[standard]`                                                               |
| dev   | `pytest`, `pytest-mock`, `pytest-cov`, `ruff`, `mypy`                                        |
| train | `unsloth`, `trl`, `peft`, `bitsandbytes`, `datasets`                                         |

The `train` group is RunPod-only. Don't install it locally unless you have a CUDA GPU.

> **Embeddings**: Use `FlagEmbedding` (`BGEM3FlagModel`), not `sentence-transformers`. Only `FlagEmbedding` exposes all three BGE-M3 output types (dense, sparse, ColBERT). `sentence-transformers` gives dense only.
>
> **Gemma 4 loads via `AutoModelForImageTextToText` + `AutoProcessor`** (not `AutoModelForCausalLM`). Use `dtype=` (not `torch_dtype=`). For MPS: omit `device_map`, call `.to("mps")` after loading. Always verify via Context7 before writing any HuggingFace model-loading code — the API changes frequently.
>
> **FlagEmbedding 1.3.5 + transformers 5.x**: `src/retrieval/_compat.py` patches two APIs removed in transformers 5.x that FlagEmbedding still calls: `is_torch_fx_available` and `PreTrainedTokenizerBase.prepare_for_model`. The `build_inputs_with_special_tokens` removal is handled by inlining its XLMRobertaTokenizer special-token logic directly inside the `prepare_for_model` shim. This file must be imported before FlagEmbedding — `embedder.py` and `reranker.py` do this automatically. Do not remove this import.

## Codebase layout

```text
src/
  retrieval/    embedding, indexing, vector search, reranking, routing, query transformation
    protocols.py          abstract interface definitions
    query_router.py       keyword-based source router (pokeapi/smogon/bulbapedia)
    query_transformer.py  HyDE and passthrough transformers
    embedder.py, indexer.py, reranker.py, searcher.py  ...
  generation/   model loading, inference wrapper
    protocols.py          GeneratorProtocol
  pipeline/     RAG orchestration, multi-round retrieval
    rag_pipeline.py       RAGPipeline orchestrates retrieval → generation
    types.py              PipelineResult (with confidence_score field)
  api/          FastAPI app
    app.py                FastAPI instance with rate limiting middleware
    models.py             QueryRequest, QueryResponse (with confidence_score field)
  config.py     environment configuration (Settings dataclass)
  utils/        shared helpers, logging
    logging.py            setup_logging(), suppresses httpx INFO logs

tests/
  conftest.py            shared fixtures
  unit/                  no I/O, no model, fast
    test_config.py
    test_inference.py
    test_loader.py
    ... (other unit tests)
  integration/           real disk/index I/O, uses fixture data
    test_api.py
    test_api_lifespan.py
    ... (other integration tests)
  e2e/                   full pipeline, GPU required

scripts/
  build_index.py              embed and index processed/ data (run once)
  training/
    generate_sft_data.py      SFT data generation via Gemini API
    train_sft.py              SFT training with Unsloth on RunPod
    clean_sft_data.py         data cleaning/validation
    gemini_client.py          Gemini API wrapper
    sampler.py                sampling utilities
    schemas.py                training data types
    pokesage_system.py         system prompt constant
    runpod_setup.sh           RunPod environment provisioning
    RUNPOD_SETUP_NOTES.md      RunPod setup guide

processed/              READ ONLY
```

## Data sources and chunking

| Path                    | Format                                 | Chunking strategy                                                                                       |
| ----------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `processed/bulbapedia/` | `Title: ...\n<body>`, one doc per line | Split at `Title:` boundary first, then recursive by `\n\n` → sentence. Target 512 tokens, ~10% overlap. |
| `processed/pokeapi/`    | one entry per line, no header          | No chunking — each line is already an atomic fact (~100–300 tokens). One doc per line.                  |
| `processed/smogon/`     | `Name (tier): ...`, one entry per line | Recursive sentence-aware split. Target 256–512 tokens, ~10% overlap.                                    |

Every chunk must carry these metadata fields in its Qdrant payload: `source` (`bulbapedia` / `pokeapi` / `smogon`), `entity_name` (if extractable), `entity_type`, `chunk_index`, `original_doc_id`.

`_aug.txt` variants are paraphrased rewrites for training/retrieval diversity. Read-only like the originals.

## Retrieval pipeline

**Query routing** (optional, enabled via `ROUTING_ENABLED=true`):
- `QueryRouter` in `src/retrieval/query_router.py` classifies queries into sources via keyword patterns
- Patterns use whole-word (`\b...\b`), phrase-prefix, and stem matching (case-insensitive)
- If no patterns match, routes to all three sources
- Implements `QueryRouterProtocol`

**Query transformation** (optional, enabled via `HYDE_ENABLED=true`):
- `HyDETransformer` in `src/retrieval/query_transformer.py` generates hypothetical answers before embedding
- Shifts retrieval from query-to-doc to answer-to-answer similarity
- Configurable max tokens via `HYDE_MAX_TOKENS` (default 150)
- Falls back to original query on inference failure
- Implements `QueryTransformerProtocol`; `PassthroughTransformer` is the identity function

**Embedding**:
- **Model**: `BAAI/bge-m3` via `FlagEmbedding.BGEM3FlagModel`
- **Output types**: Dense (1024-dim, always indexed), Sparse (keyword/lexical, free), ColBERT multi-vector (optional, higher recall/cost)

**Vector storage**:
- **DB**: Qdrant (local Docker in dev, hosted or RunPod-attached in prod)
- **Collections**: Each source (`bulbapedia`, `pokeapi`, `smogon`) is a separate Qdrant collection
- **Retrieval**: Hybrid dense + sparse in a single BGE-M3 pass, fused with Qdrant `Prefetch` + `Fusion.RRF` (RRF chosen over weighted sum — no per-collection tuning required)
- **Reranking**: Optional BGE-M3 reranker (`BAAI/bge-reranker-v2-m3`) applied post-retrieval

**Error handling**:
- If retrieval returns no documents, `RAGPipeline.query()` raises `RetrievalError("Retrieval returned no documents for query")`
- Generator is never called if retrieval fails

## Protocols

All major components implement protocols from `src/retrieval/protocols.py` and `src/generation/protocols.py`:

- `EmbedderProtocol` — `encode(texts: list[str]) -> EmbeddingOutput`
- `VectorStoreProtocol` — `ensure_collections()`, `upsert()`, `search()`
- `RerankerProtocol` — `rerank(query, documents, top_k) -> list[RetrievedChunk]`
- `RetrieverProtocol` — `retrieve(query, top_k, sources, entity_name) -> RetrievalResult`
- `QueryRouterProtocol` — `route(query: str) -> list[Source]` (returns non-empty sorted list)
- `QueryTransformerProtocol` — `transform(query: str) -> str` (returns original on failure)
- `GeneratorProtocol` — `generate(query, chunks) -> GenerationResult`

These protocols enable unit testing with mocks instead of loading real models.

## Pipeline and API

**`build_pipeline()` return type**: `tuple[RAGPipeline, ModelLoader, QdrantClient]`

**RAGPipeline constructor**:
- Requires: `retriever: RetrieverProtocol`, `generator: GeneratorProtocol`
- Optional: `query_router: QueryRouterProtocol | None = None`
- If `query_router` is provided and `sources=None` in `query()` call, router classifies the query

**Response types**:

- `PipelineResult` (in `src/pipeline/types.py`): includes `confidence_score: float | None = None` (sigmoid of top chunk score)
- `QueryResponse` (in `src/api/models.py`): includes `confidence_score: float | None = None`

**API endpoints**:

- `POST /query` — Main RAG endpoint (20 req/min/IP rate limit via `RATE_LIMIT_ENABLED`)
- `GET /stats` — Returns dict of Qdrant collection names → bool

**Environment configuration** (in `src/config.py` Settings):

- `ROUTING_ENABLED=true/false` — Enable QueryRouter (default: false)
- `HYDE_ENABLED=true/false` — Enable HyDETransformer (default: false)
- `HYDE_MAX_TOKENS=N` — Max tokens for HyDE output (default: 150)
- `LORA_ADAPTER_PATH=/path/to/adapter` — Path to LoRA weights (optional, only if fine-tuned on RunPod)
- `qdrant_api_key` — `SecretStr` (Pydantic), masked in logs/repr

**Security & logging**:

- Query prompt injection: `src/generation/prompt_builder.py` strips newlines from user queries
- Rate limiting via `RateLimitMiddleware` in `src/api/app.py` (configurable via `RATE_LIMIT_ENABLED`)
- `httpx` INFO logs suppressed in `setup_logging()` — prevents Qdrant URL leakage in server logs

## Non-negotiable rules

- **Never write to `processed/`**
- **Never call the generator if retrieval fails** — raise an exception, don't silently fall back
- **TDD always** — write a failing test before any implementation
- **Context7 before any library call** — never assume HuggingFace, vector DB, or training API behaviour from memory
- **`scripts/training/` is isolated** — nothing in `src/` may import from it

## See also

`CONTRIBUTING.md` · `TESTING.md` · `SECURITY.md` · `.claude/rules.md`
