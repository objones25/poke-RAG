# poke-RAG

Agentic retrieval-augmented generation (RAG) system for Pokémon knowledge, powered by `google/gemma-4-E4B-it` and grounded in three authoritative sources: Bulbapedia, PokéAPI, and Smogon. Optional supervised fine-tuning (SFT) with LoRA on RunPod.

## Overview

poke-RAG is a production-ready RAG pipeline that answers Pokémon questions with citations. It retrieves contextual passages from a vector index built from three knowledge sources, ranks them by relevance, and feeds them to a language model for generation. The system enforces a hard invariant: **the generator is never called if retrieval fails**, ensuring no hallucination from empty context.

### Architecture

```text
User Query
    ↓
[Query Parser] — parse source/filter constraints
    ↓
[Query Router] (optional) — classify query into sources via keyword patterns
    ↓
[HyDE Transformer] (optional) — generate hypothetical document embedding
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
[Gemma 4 Generator] — answer + attribution (optional LoRA adapter)
    ↓
QueryResponse (answer, sources, chunks_used, confidence_score, model_name)
```

Each query hits one or more collections via source filtering or keyword routing. Dense and sparse vectors are fused with Qdrant's reciprocal rank fusion (RRF), avoiding per-collection tuning. Optional query transformation (HyDE) and source routing for smarter retrieval.

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
  "model_name": "google/gemma-4-E4B-it",
  "query": "What are Charizard's base stats?"
}
```

The `confidence_score` is the sigmoid of the top-ranked chunk's BGE Reranker v2-m3 score (0.0–1.0), indicating how confident the system is in the retrieved evidence. Use this to filter low-confidence responses in production. If reranking is skipped or disabled, `confidence_score` is `null`.

### 7. Streaming Query

The `/query/stream` endpoint streams tokens one-at-a-time via [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) as the model produces them — making responses feel instant rather than waiting for the full answer.

```bash
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What moves does Gengar learn?"}' \
  --no-buffer
```

Each line of the stream is a JSON event on a `data:` prefix:

```
data: {"token": "Gengar"}
data: {"token": " learns"}
data: {"token": " Shadow"}
data: {"token": " Ball"}
...
data: {"done": true}
```

Python (`httpx` is already available as a transitive dependency via `qdrant-client`):

```python
import json
import httpx

with httpx.stream(
    "POST",
    "http://localhost:8000/query/stream",
    json={"query": "What moves does Gengar learn?"},
    timeout=None,
) as response:
    response.raise_for_status()
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line[len("data: "):])
        if event.get("done"):
            break
        if "error" in event:
            raise RuntimeError(f"Stream error: {event['error']}")
        print(event["token"], end="", flush=True)
print()  # newline after stream ends
```

**Event format:**

| Event | Payload | When |
|-------|---------|------|
| Token | `{"token": "..."}` | Each token produced by the model |
| Done  | `{"done": true}`   | Stream complete (always last event) |
| Error | `{"error": "Stream generation failed"}` | Retrieval or generation failure |

The streaming endpoint requires `ASYNC_PIPELINE_ENABLED=true` (the default when running the API normally). Rate limiting and body size limits apply identically to `/query`.

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
│   ├── retrieval/              # Embedding, indexing, vector search, routing, transformation
│   │   ├── embedder.py         # BGEEmbedder wrapper (BGE-M3)
│   │   ├── chunker.py          # Source-specific chunking logic
│   │   ├── vector_store.py     # QdrantVectorStore (wrapper)
│   │   ├── retriever.py        # Orchestrates embed → search → rerank
│   │   ├── reranker.py         # BGE Reranker v2-m3
│   │   ├── query_router.py     # Keyword-based source classification (optional)
│   │   ├── query_transformer.py # HyDE & passthrough transformers (optional)
│   │   ├── context_assembler.py # Token-bounded context assembly
│   │   ├── types.py            # EmbeddingOutput, etc.
│   │   ├── protocols.py        # Interfaces (RetrieverProtocol, etc.)
│   │   ├── _compat.py          # FlagEmbedding + transformers 5.x compatibility shims
│   │   └── constants.py        # Retrieval constants
│   │
│   ├── generation/             # Model loading and inference
│   │   ├── loader.py           # ModelLoader for base model + LoRA adapter
│   │   ├── generator.py        # Gemma 4 generation wrapper
│   │   ├── inference.py        # Low-level inference execution
│   │   ├── prompt_builder.py   # System prompts, prompt injection guards
│   │   ├── models.py           # Generation request/response models
│   │   └── protocols.py        # GeneratorProtocol, PromptBuilderProtocol
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
│   ├── training/               # SFT fine-tuning (RunPod only, not imported by src/)
│   │   ├── train_sft.py        # QLoRA SFT training via Unsloth + TRL
│   │   ├── generate_sft_data.py # Generate SFT pairs from retrieval chunks
│   │   ├── clean_sft_data.py   # Data cleaning & deduplication
│   │   ├── gemini_client.py    # Gemini API client for QA generation
│   │   ├── sampler.py          # Stratified data sampling
│   │   ├── pokesage_system.py  # System prompt definition
│   │   ├── schemas.py          # Data class schemas
│   │   ├── runpod_setup.sh     # RunPod environment setup
│   │   └── RUNPOD_SETUP_NOTES.md # RunPod instructions
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
RERANK_MODEL=BAAI/bge-reranker-v2-m3       # BGE Reranker v2-m3, don't change
GEN_MODEL=google/gemma-4-E4B-it            # Gemma 4, don't change

# LoRA fine-tuning
LORA_ADAPTER_PATH=                         # Optional: path to local LoRA adapter

# Query transformation & routing (optional)
ROUTING_ENABLED=false                      # Enable keyword-based query router
HYDE_ENABLED=false                         # Enable HyDE query transformation
HYDE_MAX_TOKENS=150                        # Max tokens for HyDE pseudo-answer

# Generation parameters
TEMPERATURE=0.7                            # Model temperature (0.0-2.0)
MAX_NEW_TOKENS=512                         # Max tokens to generate
TOP_P=0.9                                  # Top-P nucleus sampling (0.0-1.0)
DO_SAMPLE=true                             # Use sampling vs. greedy decoding

# Device / GPU
DEVICE=cuda                                # cpu, cuda, or mps

# API settings
RATE_LIMIT_ENABLED=true                    # Enable/disable rate limiting
QUERY_TIMEOUT_SECONDS=120                  # Query timeout (increase for MPS)
ALLOWED_ORIGINS=*                          # CORS allowed origins
LOG_LEVEL=INFO                             # Log level (DEBUG, INFO, WARNING, ERROR)
TRUSTED_PROXY_COUNT=0                      # Number of trusted proxies (for X-Forwarded-For)
```

Load these in code via:

```python
from src.config import Settings
settings = Settings.from_env()
```

### Configuration Details

| Variable | Default | Required | Description |
| -------- | ------- | -------- | ----------- |
| `QDRANT_URL` | `http://localhost:6333` | Yes | Qdrant vector DB URL (Docker locally, hosted in prod) |
| `QDRANT_API_KEY` | (none) | No | API key for cloud Qdrant; omit for local |
| `EMBED_MODEL` | `BAAI/bge-m3` | No | BGE-M3 embedding model (do not change) |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | No | BGE Reranker v2-m3 model (do not change) |
| `GEN_MODEL` | `google/gemma-4-E4B-it` | No | Gemma 4 generation model (do not change) |
| `LORA_ADAPTER_PATH` | (none) | No | Path to local PEFT LoRA adapter (e.g. `models/pokesage-lora`). If set but path doesn't exist, falls back to `objones25/pokesage-lora` on HF Hub. Omit to run base model only. |
| `ROUTING_ENABLED` | `false` | No | Enable keyword-based query router to classify queries into sources |
| `HYDE_ENABLED` | `false` | No | Enable HyDE query transformation (generates pseudo-answer for better retrieval) |
| `HYDE_MAX_TOKENS` | `150` | No | Maximum tokens for HyDE pseudo-answer generation |
| `TEMPERATURE` | `0.7` | No | Model temperature for generation (0.0–2.0, higher = more creative) |
| `MAX_NEW_TOKENS` | `512` | No | Maximum tokens to generate in response |
| `TOP_P` | `0.9` | No | Top-P nucleus sampling (0.0–1.0) |
| `DO_SAMPLE` | `true` | No | Use sampling vs. greedy decoding |
| `DEVICE` | `cuda` | No | Device: `cpu`, `cuda`, or `mps` (Apple Silicon) |
| `RATE_LIMIT_ENABLED` | `true` | No | Enable/disable rate limiting on `/query` endpoint (20 req/min/IP) |
| `QUERY_TIMEOUT_SECONDS` | `120` | No | Timeout for inference. Increase for MPS (e.g. `300`). |
| `ALLOWED_ORIGINS` | `*` | No | CORS allowed origins (comma-separated or `*` for all) |
| `LOG_LEVEL` | `INFO` | No | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `TRUSTED_PROXY_COUNT` | `0` | No | Number of trusted proxies for X-Forwarded-For header parsing |

## Core Dependencies

| Group     | Key Packages                             | Purpose                            |
| --------- | ---------------------------------------- | ---------------------------------- |
| **core**  | `transformers` ≥5.5.0, `torch` ≥2.11.0  | Model loading & inference          |
|           | `accelerate`                             | Multi-GPU inference support        |
|           | `FlagEmbedding` ≥1.3.5                   | BGE-M3 embeddings (dense + sparse) |
|           | `qdrant-client` ≥1.17.1                  | Vector DB client                   |
|           | `peft` ≥0.14.0                           | LoRA adapter loading at inference  |
|           | `pydantic` ≥2.13.3, `numpy` ≥2.4.4      | Data validation, numerics          |
| **api**   | `fastapi` ≥0.136.0, `uvicorn[standard]`  | HTTP server with standard middleware |
| **dev**   | `pytest` ≥9.0.3, `pytest-mock`, `pytest-cov` | Testing & coverage                |
|           | `ruff` ≥0.15.11, `mypy` ≥1.20.1         | Linting & type checking (strict)   |
| **train** | `unsloth`, `trl`, `bitsandbytes`         | SFT fine-tuning (RunPod only)      |
|           | `datasets`                               | Dataset handling for training      |

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

#### Compatibility Notes

**FlagEmbedding 1.3.5 + transformers 5.x**: `src/retrieval/_compat.py` patches two APIs removed in transformers 5.x that FlagEmbedding still calls: `is_torch_fx_available` and `PreTrainedTokenizerBase.prepare_for_model` (with `build_inputs_with_special_tokens` logic inlined inside the `prepare_for_model` shim). This file must be imported before FlagEmbedding — `embedder.py` and `reranker.py` do this automatically. Do not remove this import.

### Vector DB: Qdrant

Three separate collections — one per source:

- `bulbapedia` — dense + sparse vectors
- `pokeapi` — dense + sparse vectors
- `smogon` — dense + sparse vectors

Each collection stores both `vectors_config` (dense, 1024-dim, cosine) and `sparse_vectors_config` (sparse weights). This enables simultaneous dense and sparse search, fused with reciprocal rank fusion (RRF) via Qdrant `Prefetch` + `Fusion.RRF`.

### Retrieval Pipeline

1. **Classify query** (optional) — Keyword-based router classifies query into one or more sources via regex patterns (enable with `ROUTING_ENABLED=true`)
2. **Transform query** (optional) — HyDE transformer generates a hypothetical document embedding instead of raw query (enable with `HYDE_ENABLED=true`)
3. **Embed query** — BGE-M3 dense + sparse (original or HyDE-transformed query)
4. **Hybrid search** — Qdrant searches selected or all collections with RRF fusion
5. **Rerank** — Top-K candidates reranked with `BAAI/bge-reranker-v2-m3`
6. **Assemble context** — Chunks truncated to token budget, ordered by score, with metadata
7. **Generate** — Gemma 4 answers with retrieved context, optionally using LoRA adapter

### Query Router (Optional)

The keyword-based `QueryRouter` in `src/retrieval/query_router.py` automatically classifies queries into sources without explicit user filtering:

- **PokéAPI patterns**: stats, types, abilities, evolution, breeding info
- **Bulbapedia patterns**: competitive movesets, type matchups, game mechanics, lore
- **Smogon patterns**: competitive tiers, usage data, set recommendations

Enable with `ROUTING_ENABLED=true`. Router outputs a `source_classification: dict[Source, float]` (confidence scores per source). If scores are low (no clear match), queries fallback to searching all sources.

### HyDE Query Transformation (Optional)

The `HyDETransformer` in `src/retrieval/query_transformer.py` generates a hypothetical document (pseudo-answer) and uses that text for retrieval instead of the raw query, shifting retrieval to answer-to-answer similarity. This often improves recall on conceptual questions.

Enable with `HYDE_ENABLED=true`, configure max tokens with `HYDE_MAX_TOKENS=150` (default). The transformer:
1. Sends query through Gemma 4 to generate a hypothetical answer
2. Uses that answer text for embedding & retrieval
3. Falls back to original query on any failure

Recommended for complex Pokémon knowledge questions; disable for factual lookups.

## Generation: Gemma 4 4B-it

**Important**: Load Gemma 4 via `AutoModelForImageTextToText` with `AutoProcessor` (not `AutoModelForCausalLM`):

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-E4B-it",
    attn_implementation="sdpa",
    dtype="float16",
)
# For MPS: omit device_map, load on CPU, then call .to("mps")
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
- **e2e/** — Real Gemma 4 model, requires GPU (~10-60s each)

See `TESTING.md` for mocking patterns (embedder, generator), the no-fallback invariant test, and fixtures.

## Fine-Tuning with LoRA (SFT)

The `pokesage-lora` adapter is a PEFT LoRA adapter trained with Supervised Fine-Tuning (SFT) on Gemma 4 4B-it using Unsloth + TRL on RunPod.

### Training Method

**SFT (Supervised Fine-Tuning)** with QLoRA (quantized LoRA):

- **Framework**: Unsloth + TRL `SFTTrainer` (4-bit quantization for memory efficiency)
- **Model**: `google/gemma-4-E4B-it`
- **Adapter Config**: LoRA with r=16, alpha=16, targeting all linear layers
- **Adapter Size**: ~147MB
- **Data Format**: JSONL with conversation messages (`role`/`content`)
- **Data Source**: SFT pairs generated from retrieval chunks via `generate_sft_data.py` using Gemini API

### Training Data Pipeline

1. **generate_sft_data.py** — Chunks from `processed/` → Gemini API to generate Q&A pairs
2. **clean_sft_data.py** — Deduplication, validation, quality filtering
3. **sampler.py** — Stratified sampling across sources (bulbapedia, pokeapi, smogon)
4. **train_sft.py** — QLoRA training with TRL SFTTrainer

### Published Adapters

| Adapter | Status | Base Model | Checkpoint |
|---------|--------|-----------|------------|
| `pokesage-lora` (objones25/pokesage-lora) | v1 | gemma-4-E4B-it | Epoch 2 (best val_loss) |

Available on [HuggingFace Hub](https://huggingface.co/objones25/pokesage-lora).

### Using the LoRA Adapter at Inference

The adapter is automatically loaded at startup if `LORA_ADAPTER_PATH` is set:

```bash
# Use a local adapter
export LORA_ADAPTER_PATH=models/pokesage-lora
uv run uvicorn src.api.app:app

# Or use the HF Hub version (auto-download)
export LORA_ADAPTER_PATH=objones25/pokesage-lora
uv run uvicorn src.api.app:app

# Omit LORA_ADAPTER_PATH to run base model only
uv run uvicorn src.api.app:app
```

During startup, `ModelLoader._apply_lora_adapter()` wraps the base model with `PeftModel.from_pretrained`:
1. If `LORA_ADAPTER_PATH` is set, tries to load from local path
2. If local path doesn't exist, falls back to HF Hub
3. Raises `RuntimeError` if the adapter cannot be loaded

### Performance Notes

On **Apple Silicon (MPS)**, Gemma 4 4B-it inference takes ~120 seconds per query. To avoid timeouts:

```bash
export QUERY_TIMEOUT_SECONDS=300  # 5 minutes
uv run uvicorn src.api.app:app
```

### Training on RunPod (SFT)

SFT scripts are in `scripts/training/` (isolated from `src/`, no imports to or from `src/`). Recommended GPU: **RTX 4090 (24GB)** on RunPod community (~$0.35–$0.69/hr). Gemma 4 4B with 4-bit quantization requires ~8–10 GB VRAM.

**Workflow**:

1. Spin up RunPod with PyTorch template (CUDA 12.6, torch 2.7 recommended)
2. Attach network volume for checkpoints and data
3. `git clone`, `uv sync --all-extras` (includes `train` extra)
4. Run `scripts/training/runpod_setup.sh` — patches Unsloth and installs exact dependencies
5. Generate SFT data: `uv run python scripts/training/generate_sft_data.py --data-dir /mnt/volume/data/`
6. Clean data: `uv run python scripts/training/clean_sft_data.py`
7. Train: `uv run python scripts/training/train_sft.py --data data/sft/train.jsonl --output-dir models/pokesage-lora --epochs 3`
8. Save adapter to network volume; load at inference time via `LORA_ADAPTER_PATH`

See `scripts/training/RUNPOD_SETUP_NOTES.md` for detailed instructions and troubleshooting.

## See Also

- **[CLAUDE.md](./CLAUDE.md)** — Canonical project definition (what this is, non-negotiable rules)
- **[CONTRIBUTING.md](./CONTRIBUTING.md)** — Development setup, dependency management, git workflow, code standards, RunPod fine-tuning
- **[TESTING.md](./TESTING.md)** — Testing framework, TDD workflow, mocking patterns, coverage expectations
- **[SECURITY.md](./SECURITY.md)** — Security checklist, secret management, vulnerability scanning

## API Security & Rate Limiting

The FastAPI application includes several security features:

1. **Rate Limiting** — 20 requests per minute per IP on `/query` endpoint (configurable via `RATE_LIMIT_ENABLED`)
2. **Prompt Injection Guards** — User queries are stripped of newlines in `prompt_builder.py` to prevent prompt injection
3. **CORS Middleware** — Configurable allowed origins via `ALLOWED_ORIGINS` env var (default `*`)
4. **Request Size Limits** — Max 64 KB per request body
5. **X-Forwarded-For Parsing** — Trusted proxy support via `TRUSTED_PROXY_COUNT` for real IP detection
6. **Response Latency Tracking** — `X-Response-Time-Ms` header added to all responses
7. **Masked Secrets** — `QDRANT_API_KEY` masked in logs (Pydantic `SecretStr`)

---

**Last updated**: 2026-04-25  
**Status**: Active development
