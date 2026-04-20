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

| Group | Key packages |
|---|---|
| core | `transformers`, `torch`, `accelerate`, `FlagEmbedding`, `qdrant-client`, `pydantic`, `numpy` |
| api | `fastapi`, `uvicorn[standard]` |
| dev | `pytest`, `pytest-mock`, `pytest-cov`, `ruff`, `mypy` |
| train | `unsloth`, `trl`, `peft`, `bitsandbytes`, `datasets` |

The `train` group is RunPod-only. Don't install it locally unless you have a CUDA GPU.

> **Embeddings**: Use `FlagEmbedding` (`BGEM3FlagModel`), not `sentence-transformers`. Only `FlagEmbedding` exposes all three BGE-M3 output types (dense, sparse, ColBERT). `sentence-transformers` gives dense only.

> **Gemma 4 loads via `AutoModelForImageTextToText`, not `AutoModelForCausalLM`.** Verify via Context7 before writing any HuggingFace model-loading code — the API changes frequently.

## Codebase layout

```
src/
  retrieval/    embedding, indexing, vector search, optional reranker
  generation/   model loading, inference wrapper
  pipeline/     RAG orchestration, multi-round retrieval
  api/          FastAPI app
  utils/        shared helpers, logging

tests/
  unit/         no I/O, no model, fast
  integration/  real disk/index I/O, uses fixture data
  e2e/          full pipeline, GPU required

scripts/
  build_index.py        embed and index processed/ data (run once)
  training/             LoRA fine-tuning scripts (RunPod only)

processed/              READ ONLY
```

## Data sources and chunking

| Path | Format | Chunking strategy |
|---|---|---|
| `processed/bulbapedia/` | `Title: ...\n<body>`, one doc per line | Split at `Title:` boundary first, then recursive by `\n\n` → sentence. Target 512 tokens, ~10% overlap. |
| `processed/pokeapi/` | one entry per line, no header | No chunking — each line is already an atomic fact (~100–300 tokens). One doc per line. |
| `processed/smogon/` | `Name (tier): ...`, one entry per line | Recursive sentence-aware split. Target 256–512 tokens, ~10% overlap. |

Every chunk must carry these metadata fields in its Qdrant payload: `source` (`bulbapedia` / `pokeapi` / `smogon`), `pokemon_name` (if extractable), `chunk_index`, `original_doc_id`.

`_aug.txt` variants are paraphrased rewrites for training/retrieval diversity. Read-only like the originals.

## Embedding and retrieval

**Model**: `BAAI/bge-m3` via `FlagEmbedding.BGEM3FlagModel`
**Vector DB**: Qdrant (local Docker in dev, hosted or RunPod-attached in prod)
**Retrieval mode**: hybrid — dense + sparse in a single BGE-M3 pass, fused with RRF or weighted sum, then reranked with `BAAI/bge-reranker-v2-m3`

BGE-M3 output types and what to store in Qdrant:
- **Dense** (1024-dim): always index — primary semantic search
- **Sparse**: always index — keyword/lexical search, free alongside dense
- **ColBERT multi-vector**: optional — higher recall, significantly more storage and query cost; add later if recall is insufficient

Each source (`bulbapedia`, `pokeapi`, `smogon`) is a **separate Qdrant collection**. Queries target one or more collections via namespace parameter. This enables source-specific retrieval (e.g. stats-only queries hit `pokeapi` only).

## Non-negotiable rules

- **Never write to `processed/`**
- **Never call the generator if retrieval fails** — raise an exception, don't silently fall back
- **TDD always** — write a failing test before any implementation
- **Context7 before any library call** — never assume HuggingFace, vector DB, or training API behaviour from memory
- **`scripts/training/` is isolated** — nothing in `src/` may import from it

## See also

`CONTRIBUTING.md` · `TESTING.md` · `SECURITY.md` · `.claude/rules.md`
