# poke-RAG

Agentic retrieval-augmented generation (RAG) system for Pokémon knowledge, powered by `google/gemma-2-2b-it` and grounded in three authoritative sources: Bulbapedia, PokéAPI, and Smogon.

## Overview

poke-RAG is a production-ready RAG pipeline that answers Pokémon questions with citations. It retrieves contextual passages from a vector index built from three knowledge sources, ranks them by relevance, and feeds them to a language model for generation. The system enforces a hard invariant: **the generator is never called if retrieval fails**, ensuring no hallucination from empty context.

### Architecture

```text
User Query
    ↓
[Query Parser] — parse source/filter constraints
    ↓
[BGE-M3 Embedder] — dense + sparse vectors in one pass
    ↓
[Qdrant Hybrid Search] — 3 separate collections (bulbapedia/pokeapi/smogon)
    ↓              ↓              ↓
bulbapedia     pokeapi        smogon
(dense+sparse) (dense+sparse)  (dense+sparse)
    ↓              ↓              ↓
[Qdrant RRF Fusion] — reciprocal rank fusion
    ↓
[BGE Reranker v2-m3] — final relevance score
    ↓
[Context Assembler] — token-bounded context
    ↓
[Gemma 2 Generator] — answer + attribution
    ↓
QueryResponse (answer, sources, chunks_used, confidence_score, model_name)
```

Each query hits one or more collections via source filtering. Dense and sparse vectors are fused with Qdrant's reciprocal rank fusion (RRF), avoiding per-collection tuning.

## Data Sources

Three read-only sources in `processed/`:

| Source         | Format                              | Chunking Strategy                                               | Target Size     | Overlap |
| -------------- | ----------------------------------- | --------------------------------------------------------------- | --------------- | ------- |
| **Bulbapedia** | `Title: ...\n<body>` (one per line) | Split at `Title:` boundary, then recursive by `\n\n` → sentence | 512 tokens      | ~10%    |
| **PokéAPI**    | One entry per line, no header       | None — each line is atomic                                      | ~100–300 tokens | 0%      |
| **Smogon**     | `Name (tier): ...` (one per line)   | Recursive, sentence-aware                                       | 256–512 tokens  | ~10%    |

Every chunk carries metadata: `source`, `entity_name` (Pokémon/move/ability name if extractable), `entity_type`, `chunk_index`, `original_doc_id`. This enables source-specific and entity-specific retrieval (e.g., "stats-only" queries hit `pokeapi` exclusively).

Augmented variants (`*_aug.txt`) are paraphrased rewrites for training diversity—read-only like originals.

## Prerequisites

- **Python**: 3.11 or higher
- **uv**: Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Docker**: Required for local Qdrant (development only)
- **GPU** (optional): NVIDIA CUDA for fine-tuning, E2E tests

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/objones25/poke-RAG.git
cd poke-RAG

# Install all dependencies (core + API + dev tools)
uv sync --all-extras

# Verify setup
uv run pytest -m "not gpu" --tb=short
uv run ruff check .
uv run mypy src/
```

### 2. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Qdrant will be available at `http://localhost:6333`. Set `QDRANT_URL=http://localhost:6333` in your `.env`.

### 3. Build the Vector Index

```bash
uv run python scripts/build_index.py
```

This discovers files in `processed/` (bulbapedia, pokeapi, smogon), chunks them, embeds with BGE-M3, and upserts into Qdrant. A checkpoint file `.build_index_checkpoint.json` tracks progress; re-run safely to index only new files.

Optional flags:

- `--source bulbapedia` — index only one source
- `--batch-size 16` — adjust GPU memory usage
- `--dry-run` — log without writing
- `--no-checkpoint` — rebuild from scratch

### 4. Run the API

```bash
uv run uvicorn src.api.app:app --reload
```

Server listens on `http://localhost:8000`. See `/docs` for Swagger UI.

Rate limiting: 20 requests per minute per IP on `/query` endpoint (configurable via `RATE_LIMIT_ENABLED`).

### 5. Check Service Status

```bash
curl http://localhost:8000/stats
```

Response lists available Qdrant collections:

```json
{
  "bulbapedia": true,
  "pokeapi": true,
  "smogon": true
}
```

### 6. Example Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Charizard'\''s base stats?", "sources": ["pokeapi"]}'
```

Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What are Charizard's base stats?",
        "sources": ["pokeapi"],  # optional filter
    },
)
result = response.json()
print(result["answer"])
print(f"Sources: {result['sources_used']}")
print(f"Chunks: {result['num_chunks_used']}")
print(f"Confidence: {result.get('confidence_score')}")
```

**Response format:**

```json
{
  "answer": "Charizard has a Base Speed of 100, Special Attack of 109, ...",
  "sources_used": ["pokeapi"],
  "num_chunks_used": 3,
  "confidence_score": 0.87,
  "model_name": "google/gemma-2-2b-it",
  "query": "What are Charizard's base stats?"
}
```

## Commands Reference

### Development

```bash
uv sync --all-extras          # Install all deps (core + api + dev + train)
uv run pytest                 # Run all tests (~398 tests)
uv run pytest -m "not gpu"    # Skip GPU tests (local dev default)
uv run pytest tests/unit/     # Unit tests only
uv run pytest --cov=src --cov-report=html  # Coverage report
uv run ruff check .           # Lint
uv run ruff format .          # Format
uv run mypy src/              # Type check (strict mode)
```

### Indexing

```bash
uv run python scripts/build_index.py                    # Build index from processed/ data
uv run python scripts/build_index.py --source pokeapi   # Index one source only
uv run python scripts/build_index.py --batch-size 16    # Adjust batch size
```

### Running the API

```bash
uv run uvicorn src.api.app:app --reload --port 8000
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000  # For production
```

## Project Layout

```text
poke-RAG/
├── README.md                   # This file
├── CLAUDE.md                   # Authoritative project definition
├── CONTRIBUTING.md             # Development guidelines
├── TESTING.md                  # Testing framework & patterns
├── SECURITY.md                 # Security checklist
├── pyproject.toml              # Project metadata & dependencies
├── uv.lock                     # Pinned dependency versions
│
├── src/                        # Main application package
│   ├── api/                    # FastAPI app, routes, models
│   │   ├── app.py              # Application entry point
│   │   ├── models.py           # Request/response Pydantic models
│   │   ├── dependencies.py     # Dependency injection (pipeline construction)
│   │   └── query_parser.py     # Query string parsing & source extraction
│   │
│   ├── retrieval/              # Embedding, indexing, vector search
│   │   ├── embedder.py         # BGEEmbedder wrapper (BGE-M3)
│   │   ├── chunker.py          # Source-specific chunking logic
│   │   ├── vector_store.py     # QdrantVectorStore (wrapper)
│   │   ├── retriever.py        # Orchestrates embed → search → rerank
│   │   ├── reranker.py         # BGE Reranker v2-m3
│   │   ├── types.py            # EmbeddingOutput, etc.
│   │   └── protocols.py        # Interfaces (RetrieverProtocol, etc.)
│   │
│   ├── generation/             # Model inference
│   │   ├── generator.py        # Gemma 2 generation wrapper
│   │   ├── protocols.py        # GeneratorProtocol, PromptBuilderProtocol
│   │   └── prompts.py          # System prompts, context assembly
│   │
│   ├── pipeline/               # RAG orchestration
│   │   ├── rag_pipeline.py     # Main query orchestrator
│   │   └── types.py            # PipelineResult, etc.
│   │
│   ├── config.py               # Settings (env vars, device, URLs)
│   ├── types.py                # Domain types (RetrievedChunk, RetrievalError, etc.)
│   └── utils/                  # Shared helpers
│       ├── logging.py          # Logging setup
│       └── ...                 # Additional utilities
│
├── tests/                      # Test suite
│   ├── conftest.py             # Shared fixtures (mock embedder, generator, etc.)
│   ├── unit/                   # Pure logic tests (no I/O, no model)
│   │   ├── test_chunker.py
│   │   ├── test_retriever.py
│   │   ├── test_pipeline.py
│   │   └── ...
│   ├── integration/            # Real I/O against fixture data
│   │   ├── test_embedder.py
│   │   ├── test_qdrant_store.py
│   │   ├── test_api.py
│   │   └── ...
│   ├── e2e/                    # Full pipeline with real model (GPU required)
│   │   └── test_pokemon_queries.py
│   └── fixtures/               # Small sample data files
│       ├── sample_bulbapedia.txt
│       ├── sample_pokeapi.txt
│       └── sample_smogon.txt
│
├── scripts/
│   ├── build_index.py          # Embed processed/ and upsert to Qdrant (main script)
│   ├── training/               # LoRA fine-tuning (RunPod only, not imported by src/)
│   │   ├── train_lora.py
│   │   └── ...
│   └── __init__.py
│
├── processed/                  # READ-ONLY knowledge sources (never write here)
│   ├── bulbapedia/             # Bulbapedia articles
│   │   ├── *.txt               # One Wikipedia-style doc per file
│   │   └── *_aug.txt           # Paraphrased variants
│   ├── pokeapi/                # PokéAPI extracts
│   │   ├── *.txt
│   │   └── *_aug.txt
│   └── smogon/                 # Smogon strategy guides
│       ├── *.txt
│       └── *_aug.txt
│
├── data/                       # Output directory
│   ├── embeddings/             # (Future) cached embeddings
│   └── indexed/                # (Future) index checkpoints
│
├── docs/                       # Additional documentation
└── .env                        # Local environment variables (never commit)
```

## Configuration

Key environment variables (set in `.env`):

```bash
# Qdrant vector database
QDRANT_URL=http://localhost:6333           # Local dev (Docker)
QDRANT_API_KEY=                            # Optional, for cloud

# Embedding & generation models
EMBED_MODEL=BAAI/bge-m3                    # BGE-M3, don't change
GEN_MODEL=google/gemma-2-2b-it             # Gemma 2, don't change

# Device / GPU
DEVICE=cuda                                # cpu, cuda, or mps

# API settings
RATE_LIMIT_ENABLED=true                    # Enable/disable rate limiting
ALLOWED_ORIGINS=*                          # CORS allowed origins
LOG_LEVEL=INFO                             # Log level (INFO, DEBUG, WARNING, ERROR)
```

Load these in code via:

```python
from src.config import Settings
settings = Settings.from_env()
```

## Core Dependencies

| Group     | Key Packages                             | Purpose                            |
| --------- | ---------------------------------------- | ---------------------------------- |
| **core**  | `transformers`, `torch`, `accelerate`    | Model loading & inference          |
|           | `FlagEmbedding` ≥1.3.5                   | BGE-M3 embeddings (dense + sparse) |
|           | `qdrant-client` ≥1.17.1                  | Vector DB client                   |
|           | `pydantic`, `numpy`                      | Data validation, numerics          |
| **api**   | `fastapi`, `uvicorn[standard]`           | HTTP server                        |
| **dev**   | `pytest`, `pytest-mock`, `pytest-cov`    | Testing                            |
|           | `ruff`, `mypy`                           | Linting & type checking            |
| **train** | `unsloth`, `trl`, `peft`, `bitsandbytes` | LoRA fine-tuning (RunPod only)     |

Never use `pip install` — use `uv add` only. See `CONTRIBUTING.md` for dependency management.

## Embedding & Retrieval Details

### Model: BGE-M3 (via FlagEmbedding)

Do **not** use `sentence-transformers` — it only returns dense vectors. `FlagEmbedding.BGEM3FlagModel` returns all three types in one pass:

```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
output = model.encode(
    texts=["Charizard's base stats"],
    return_dense=True,
    return_sparse=True,
    # return_colbert_vecs=True  # optional, not used yet
)
# output["dense_vecs"]      — 1024-dim float vectors (primary)
# output["lexical_weights"] — sparse token weights (keywords)
# output["colbert_vecs"]    — multi-vector (not indexed yet)
```

### Vector DB: Qdrant

Three separate collections — one per source:

- `bulbapedia` — dense + sparse vectors
- `pokeapi` — dense + sparse vectors
- `smogon` — dense + sparse vectors

Each collection stores both `vectors_config` (dense, 1024-dim, cosine) and `sparse_vectors_config` (sparse weights). This enables simultaneous dense and sparse search, fused with reciprocal rank fusion (RRF) via Qdrant `Prefetch` + `Fusion.RRF`.

### Retrieval Pipeline

1. **Embed query** — BGE-M3 dense + sparse
2. **Hybrid search** — Qdrant searches all or selected collections with RRF fusion
3. **Rerank** — Top-K candidates reranked with `BAAI/bge-reranker-v2-m3`
4. **Assemble context** — Chunks truncated to token budget, ordered by score
5. **Generate** — Gemma 2 answers with retrieved context

## Generation: Gemma 2 2B-it

**Important**: Load Gemma 2 via `AutoModelForCausalLM`, causal LM (not a multimodal model):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype="auto",
)
```


## Non-Negotiable Rules

These are enforced by code review and tests:

1. **Never write to `processed/`** — It is read-only, version-controlled knowledge.
2. **Never call the generator if retrieval fails** — If `retriever.retrieve()` raises `RetrievalError`, propagate it; do not fall back to generation. This is tested explicitly in `test_pipeline_does_not_call_generator_when_retrieval_fails`.
3. **TDD always** — Write a failing test first. See `TESTING.md`.
4. **Type annotations required** — All function signatures must have types. `mypy` runs in strict mode.
5. **Immutable data** — Use `@dataclass(frozen=True)` or Pydantic models. Never mutate domain objects after creation.
6. **No hardcoded secrets** — Use environment variables via `python-dotenv`.
7. **`scripts/training/` is isolated** — Nothing in `src/` may import from it.

## Development Workflow

### Branch Strategy

```text
main          — stable, CI always green
dev           — integration branch
feature/<n>   — new features (branch from dev)
fix/<n>       — bug fixes
experiment/   — exploratory work, relaxed test requirements
```

### TDD (Mandatory for all `src/` changes)

1. Write a **failing** test
2. Run `uv run pytest -k "your_test"` — confirm it fails
3. Implement the **minimum** code to pass
4. Refactor
5. Commit test + implementation together

See `TESTING.md` for full details on test organization, markers, mocking, and coverage expectations.

### Code Standards

- **Format**: `uv run ruff format .` (before pushing)
- **Lint**: `uv run ruff check .` (zero errors)
- **Types**: `uv run mypy src/` (strict mode, zero errors)
- **Coverage**: `≥80%` per module. Critical paths (retrieval, pipeline) should be 100%.

### Commit Messages

[Conventional Commits](https://www.conventionalcommits.org/) format:

```text
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`, `experiment`

Examples:

```text
feat(retrieval): add entity_type filtering to vector search
fix(pipeline): raise RetrievalError when index is empty
test(pipeline): cover no-fallback invariant
docs(readme): clarify BGE-M3 setup
experiment(lora): add gemma4 qlora training script
```

## Testing

Run locally:

```bash
# All tests
uv run pytest

# Skip GPU/slow tests (recommended locally)
uv run pytest -m "not gpu and not slow"

# Coverage
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

Test organization:

- **unit/** — No I/O, no model, fast (~<1s each)
- **integration/** — Real Qdrant, real embedder, fixture data (~1-10s each)
- **e2e/** — Real Gemma 2 model, requires GPU (~10-60s each)

See `TESTING.md` for mocking patterns (embedder, generator), the no-fallback invariant test, and fixtures.

## Fine-Tuning on RunPod

LoRA adapter scripts are in `scripts/training/` (isolated from `src/`). The serving API works with or without an adapter.

Recommended GPU: **RTX 4090 (24GB)** on RunPod community (~$0.35–$0.69/hr). Gemma 2 2B requires ~17GB VRAM with 4-bit quantization via Unsloth.

Steps:

1. Spin up RunPod with PyTorch template
2. Attach network volume for checkpoints
3. `git clone`, `uv sync --all-extras`
4. `uv run python scripts/training/train_lora.py` (check `--adapter-path` argument)
5. Save adapter to network volume; load at inference time

See `CONTRIBUTING.md` for full RunPod workflow.

## See Also

- **[CLAUDE.md](./CLAUDE.md)** — Canonical project definition (what this is, non-negotiable rules)
- **[CONTRIBUTING.md](./CONTRIBUTING.md)** — Development setup, dependency management, git workflow, code standards, RunPod fine-tuning
- **[TESTING.md](./TESTING.md)** — Testing framework, TDD workflow, mocking patterns, coverage expectations
- **[SECURITY.md](./SECURITY.md)** — Security checklist, secret management, vulnerability scanning

---

**Last updated**: 2026-04-20  
**Status**: Active development
