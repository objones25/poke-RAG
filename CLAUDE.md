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
  retrieval/    embedding, indexing, vector search, optional reranker
  generation/   model loading, inference wrapper
  pipeline/     RAG orchestration, multi-round retrieval
  api/          FastAPI app
  config.py     environment configuration
  utils/        shared helpers, logging

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
  build_index.py        embed and index processed/ data (run once)
  training/             LoRA fine-tuning scripts (RunPod only)

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

## Embedding and retrieval

**Model**: `BAAI/bge-m3` via `FlagEmbedding.BGEM3FlagModel`
**Vector DB**: Qdrant (local Docker in dev, hosted or RunPod-attached in prod)
**Retrieval mode**: hybrid — dense + sparse in a single BGE-M3 pass, fused with Qdrant `Prefetch` + `Fusion.RRF` (RRF chosen over weighted sum — no per-collection tuning required), then reranked with `BAAI/bge-reranker-v2-m3`

BGE-M3 output types and what to store in Qdrant:

- **Dense** (1024-dim): always index — primary semantic search
- **Sparse**: always index — keyword/lexical search, free alongside dense
- **ColBERT multi-vector**: optional — higher recall, significantly more storage and query cost; add later if recall is insufficient

Each source (`bulbapedia`, `pokeapi`, `smogon`) is a **separate Qdrant collection**. Queries target one or more collections via namespace parameter. This enables source-specific retrieval (e.g. stats-only queries hit `pokeapi` only).

## Pipeline and API

**`build_pipeline()` return type**: `tuple[RAGPipeline, ModelLoader, QdrantClient]`

**Response types**:

- `PipelineResult` (in `src/pipeline/types.py`): includes `confidence_score: float | None = None`
- `QueryResponse` (in `src/api/models.py`): includes `confidence_score: float | None = None`

**API endpoints**:

- `POST /query` — Main RAG endpoint (20 req/min/IP rate limit)
- `GET /stats` — Returns dict of Qdrant collection names → bool

**Config changes**:

- `qdrant_api_key` in `src/config.py` is now `SecretStr` (Pydantic) — masked in logs/repr

**Security & rate limiting**:

- Query prompt injection: `src/generation/prompt_builder.py` strips newlines from user queries
- Rate limiting via `RateLimitMiddleware` in `src/api/app.py` (configurable via `RATE_LIMIT_ENABLED`)

## Non-negotiable rules

- **Never write to `processed/`**
- **Never call the generator if retrieval fails** — raise an exception, don't silently fall back
- **TDD always** — write a failing test before any implementation
- **Context7 before any library call** — never assume HuggingFace, vector DB, or training API behaviour from memory
- **`scripts/training/` is isolated** — nothing in `src/` may import from it

## See also

`CONTRIBUTING.md` · `TESTING.md` · `SECURITY.md` · `.claude/rules.md`
