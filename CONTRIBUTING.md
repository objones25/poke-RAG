# CONTRIBUTING.md

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Git

For GPU work (fine-tuning, E2E tests): a CUDA-capable GPU locally, or a RunPod instance with a 24GB+ GPU.

## Local setup

```bash
git clone https://github.com/objones25/poke-RAG.git
cd poke-RAG

# Install all dependency groups
uv sync --all-extras

# Verify
uv run pytest -m "not gpu" --tb=short
uv run ruff check .
uv run mypy src/
```

If setup fails, check that Python 3.11+ is active (`uv python list`) and that CUDA drivers are installed if you need GPU support.

## Dependency groups

Dependencies live in `pyproject.toml` and are managed exclusively with `uv`. Never use `pip install` in this project.

```toml
[project]
dependencies = [
    # Model inference
    "transformers>=5.5.0,<6.0.0",
    "torch>=2.11.0",
    "accelerate>=1.13.0",
    # Embeddings — use FlagEmbedding, NOT sentence-transformers
    # sentence-transformers only returns dense vectors; FlagEmbedding
    # exposes all three BGE-M3 output types (dense, sparse, ColBERT)
    "FlagEmbedding>=1.3.5",
    # Vector DB
    "qdrant-client>=1.17.1",
    # Data / validation
    "pydantic>=2.13.3",
    "numpy>=2.4.4",
    # Vision-language model dependencies (Gemma 4)
    "pillow>=12.2.0",
    "torchvision>=0.26.0",
]

[project.optional-dependencies]
api = [
    "fastapi>=0.136.0",
    "uvicorn[standard]>=0.44.0",
]
dev = [
    "pytest>=9.0.3",
    "pytest-mock>=3.15.1",
    "pytest-cov>=7.1.0",
    "ruff>=0.15.11",
    "mypy>=1.20.1",
]
train = [
    # RunPod only — requires CUDA
    "unsloth",
    "trl>=0.8.0",
    "peft>=0.14.0",
    "bitsandbytes>=0.43.0",
    "datasets>=2.18.0",
]
gen = [
    # Optional: Google Generative AI models
    "google-genai>=1.0.0",
]
```

> These versions are starting points. Pin them in `uv.lock` after resolving. Update this table when you change `pyproject.toml`.

To add a dependency:

```bash
uv add <package>                    # core
uv add --optional dev <package>     # dev only
uv add --optional train <package>   # training only
```

Commit both `pyproject.toml` and `uv.lock`.

## Vector DB and embeddings

**Vector DB**: Qdrant. In development and tests, Qdrant is a remote hosted instance accessed via `QDRANT_URL` environment variable. Set this before running tests:

```bash
export QDRANT_URL="http://localhost:6333"  # or your remote Qdrant server
```

Three separate collections — one per source namespace: `bulbapedia`, `pokeapi`, `smogon`. Never merge sources into a single flat collection.

**Embedding model**: `BAAI/bge-m3` via `FlagEmbedding`. Do not use `sentence-transformers` — it only returns dense vectors. `FlagEmbedding.BGEM3FlagModel` returns all three types in one pass:

```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
output = model.encode(texts, return_dense=True, return_sparse=True)
# output["dense_vecs"]   — 1024-dim float vectors
# output["lexical_weights"] — sparse token weights
```

ColBERT (`return_colbert_vecs=True`) is optional. Start without it; add if recall is insufficient.

Each Qdrant collection must be created with both `vectors_config` (dense, 1024-dim, cosine) and `sparse_vectors_config` (sparse). Check current Qdrant client docs via Context7 before writing collection creation code — the API has changed between versions.

**Generation model**: Gemma 4 is a vision-language model and must be loaded with `AutoModelForImageTextToText` + `AutoProcessor`, not `AutoModelForCausalLM`. See `CLAUDE.md` for the exact loading pattern.

**Chunking strategy per source**:

| Source       | Strategy                                                             | Target size     | Overlap        |
| ------------ | -------------------------------------------------------------------- | --------------- | -------------- |
| `pokeapi`    | None — each line is one document                                     | ~100–300 tokens | 0              |
| `smogon`     | Recursive, sentence-aware (or 3-level hierarchy for smogon_data.txt) | 256–512 tokens  | ~10%           |
| `bulbapedia` | Split at `Title:` then recursive `\n\n` → sentence                   | 512 tokens      | Test 0% vs 10% |

**Smogon structured data (smogon_data.txt)**:

The `smogon_data.txt` file uses a three-level hierarchical parser (`chunk_smogon_data_file()`):

1. Pokémon blocks delimited by `={80}` lines
2. Format sections delimited by `-{40}` lines (e.g., "gen9ou", "gen9uu")
3. Overview and Set sections (`[ Overview ]`, `[ Set: Name ]`)

This generates ~17,336 chunks total (~3,094 overview, ~14,242 set), each with enriched metadata including format name, generation, tier, and (for sets) battle attributes (Item, Ability, Nature, Tera Type).

Every chunk's Qdrant payload must include: `source`, `entity_name` (string, e.g. "Garchomp"), `entity_type` (string, e.g. "pokemon", "move", "item"), `chunk_index`, `original_doc_id`, and `metadata` (optional dict of source-specific enrichments). This enables payload-filtered queries (e.g. `source == "smogon" AND entity_name == "Garchomp" AND entity_type == "pokemon"`) before the vector search runs.

**Metadata enrichment** (optional but recommended):

- **Smogon**: `metadata["format_name"]`, `metadata["generation"]` (int), `metadata["tier"]`, `metadata["chunk_kind"]` ("overview" or "set"), `metadata["set_name"]`, `metadata["item"]`, `metadata["ability"]`, `metadata["nature"]`, `metadata["tera_type"]`
- **PokeAPI**: `metadata["entity_subtype"]` ("species", "moves", "encounters", "ability", "item", "move")
- **Bulbapedia**: `metadata["topics"]` (list of topic strings), `metadata["entity_type_hint"]` (string)

**Retrieval pipeline**: dense+sparse hybrid fused with Qdrant `Prefetch` + `Fusion.RRF` → rerank top-K with `BAAI/bge-reranker-v2-m3` → assemble context. The reranker uses `FlagEmbedding.FlagReranker`, not the embedding model class. Generator dependency injection uses `PromptBuilderProtocol` (defined in `src/generation/protocols.py`) — any callable `(str, tuple[RetrievedChunk, ...]) -> str` satisfies it.

**Deduplication key**: In `ContextAssembler`, chunk deduplication uses `f"{original_doc_id}:{chunk_index}"` as the key, not a full text hash.

**Confidence scores**: `PipelineResult` and `QueryResponse` now include `confidence_score: float | None = None`, reflecting the confidence level of retrieval and generation.

**Rate limiting**: `RateLimitMiddleware` enforces 20 requests per minute per IP on `/query` endpoints. The `RATE_LIMIT_ENABLED` environment variable can be set to `"false"` to disable rate limiting. In the test suite, rate limiting is disabled by default in `tests/conftest.py` via an autouse fixture (`_disable_rate_limiting`), which also sets `QDRANT_URL=http://localhost:6333` to ensure the FastAPI lifespan can initialize settings correctly.

**Query routing**: Optional per-query source selection via `QueryRouter` (keyword-based heuristics). Disabled by default; enable with `ROUTING_ENABLED=true`.

**Query transformation**: Optional HyDE (Hypothetical Document Embeddings) expansion via `HyDETransformer`. Disabled by default; enable with `HYDE_ENABLED=true` and provide an inferencer instance to the transformer.

**Post-retrieval refinement**: Optional CRAG-style `KnowledgeRefiner` for chunk score triage, strip-level filtering, and constraint gap detection. Disabled by default; enable with `REFINER_ENABLED=true` and set threshold environment variables:

- `REFINER_UPPER_THRESHOLD` (default: 0.0) — chunks with score ≥ this are accepted unconditionally
- `REFINER_LOWER_THRESHOLD` (default: -3.0) — chunks with score < this are dropped
- `REFINER_STRIP_THRESHOLD` (default: -1.0) — sentence strips with score ≥ this are retained during filtering

Refinement detects knowledge gaps by searching for constraint keywords (gen1–gen9, tier names like "OU", "UU") in the query that don't appear in final chunks. These gaps are returned in `PipelineResult.knowledge_gaps` and `QueryResponse.knowledge_gaps`.

## Workflow

### Branches

```text
main          stable, CI always green
dev           integration branch
feature/<n>   new features (branch from dev)
fix/<n>       bug fixes
experiment/   LoRA/training work (relaxed test requirements, note this explicitly)
```

### TDD — required for all `src/` changes

1. Write a failing test
2. Confirm it fails: `uv run pytest -k "your_test" -m "not gpu"`
3. Implement the minimum code to pass
4. Refactor
5. Commit test and implementation together

Before committing, run the full test suite locally:

```bash
uv run ruff check .                    # zero errors
uv run mypy src/                       # zero errors
uv run pytest -m "not gpu" --tb=short  # all pass, coverage ≥80%
```

See `TESTING.md` for full details.

### Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) format:

```text
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`, `experiment`

Examples:

```text
feat(retrieval): add namespace filtering to vector search
fix(pipeline): raise RetrievalError when index is empty
test(pipeline): cover no-fallback invariant
experiment(lora): add gemma4 qlora training script for runpod
chore(deps): bump transformers to 4.41.0
```

### PR checklist

- [ ] All CI gates pass (`ruff`, `mypy`, `pytest -m "not gpu"`)
- [ ] Coverage doesn't drop below 80% on any module
- [ ] `pyproject.toml` and `uv.lock` committed together if deps changed
- [ ] No secrets, tokens, or API keys in any committed file
- [ ] PR description explains _why_, not just _what_

## Code standards

**Formatting and linting**: `ruff format` + `ruff check`. Both enforced in CI. Run before pushing.

**Types**: All function signatures require type annotations. `mypy` runs in strict mode — zero errors.

**Data objects**: Use `dataclasses(frozen=True)` or Pydantic models for all domain types (`Document`, `RetrievalResult`, etc.). Never mutate them after creation.

**Error handling**: Raise specific named exceptions. Never catch bare `except:` or `except Exception: pass`.

## Fine-tuning on RunPod

Training scripts are in `scripts/training/` and are **not imported by `src/`**. The serving layer works with or without a LoRA adapter.

Recommended GPU for Gemma 4 4B QLoRA: **RTX 4090 (24GB)** on RunPod community cloud (~$0.35–$0.69/hr). The model requires ~8–10 GB VRAM with 4-bit quantization via Unsloth.

Steps:

1. Spin up a RunPod pod with the PyTorch template
2. Attach a network volume for checkpoints (survives pod termination)
3. `git clone` the repo, `uv sync --extra train`
4. Run the training script: `uv run python scripts/training/train_lora.py`
5. Save the adapter to the network volume

The adapter must be loadable at inference time without retraining. See the script for `--adapter-path` argument.

## Data files

`processed/` is read-only. Never write to it from application or training code. Output goes to `data/embeddings/`, `data/indexed/`, or wherever the script specifies.
