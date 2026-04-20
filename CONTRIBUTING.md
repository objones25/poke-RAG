# CONTRIBUTING.md

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Git

For GPU work (fine-tuning, E2E tests): a CUDA-capable GPU locally, or a RunPod instance with a 24GB+ GPU.

## Local setup

```bash
git clone <repo-url>
cd <repo>

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
    "transformers>=4.40.0",
    "torch>=2.2.0",
    "accelerate>=0.27.0",
    # Embeddings — use FlagEmbedding, NOT sentence-transformers
    # sentence-transformers only returns dense vectors; FlagEmbedding
    # exposes all three BGE-M3 output types (dense, sparse, ColBERT)
    "FlagEmbedding>=1.2.0",
    # Vector DB
    "qdrant-client>=1.9.0",
    # Data / validation
    "pydantic>=2.0.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
api = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.4.0",
    "mypy>=1.9.0",
]
train = [
    # RunPod only — requires CUDA
    "unsloth",
    "trl>=0.8.0",
    "peft>=0.10.0",
    "bitsandbytes>=0.43.0",
    "datasets>=2.18.0",
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

**Vector DB**: Qdrant. Run locally via Docker for development:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
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

**Chunking strategy per source**:

| Source | Strategy | Target size | Overlap |
|---|---|---|---|
| `pokeapi` | None — each line is one document | ~100–300 tokens | 0 |
| `smogon` | Recursive, sentence-aware | 256–512 tokens | ~10% |
| `bulbapedia` | Split at `Title:` then recursive `\n\n` → sentence | 512 tokens | Test 0% vs 10% |

Every chunk's Qdrant payload must include: `source`, `pokemon_name` (if extractable from title/header), `chunk_index`, `original_doc_id`. This enables payload-filtered queries (e.g. `source == "smogon" AND pokemon_name == "Garchomp"`) before the vector search runs.

**Retrieval pipeline**: dense+sparse hybrid → rerank top-K with `BAAI/bge-reranker-v2-m3` → assemble context. The reranker uses `FlagEmbedding.FlagReranker`, not the embedding model class.

## Workflow

### Branches

```
main          stable, CI always green
dev           integration branch
feature/<n>   new features (branch from dev)
fix/<n>       bug fixes
experiment/   LoRA/training work (relaxed test requirements, note this explicitly)
```

### TDD — required for all `src/` changes

1. Write a failing test
2. Confirm it fails: `uv run pytest -k "your_test"`
3. Implement the minimum code to pass
4. Refactor
5. Commit test and implementation together

See `TESTING.md` for full details.

### Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`, `experiment`

Examples:
```
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
- [ ] PR description explains *why*, not just *what*

## Code standards

**Formatting and linting**: `ruff format` + `ruff check`. Both enforced in CI. Run before pushing.

**Types**: All function signatures require type annotations. `mypy` runs in strict mode — zero errors.

**Data objects**: Use `dataclasses(frozen=True)` or Pydantic models for all domain types (`Document`, `RetrievalResult`, etc.). Never mutate them after creation.

**Error handling**: Raise specific named exceptions. Never catch bare `except:` or `except Exception: pass`.

## Fine-tuning on RunPod

Training scripts are in `scripts/training/` and are **not imported by `src/`**. The serving layer works with or without a LoRA adapter.

Recommended GPU for Gemma 4 E4B QLoRA: **RTX 4090 (24GB)** on RunPod community cloud (~$0.35–$0.69/hr). The model requires ~17GB VRAM with 4-bit quantization via Unsloth.

Steps:
1. Spin up a RunPod pod with the PyTorch template
2. Attach a network volume for checkpoints (survives pod termination)
3. `git clone` the repo, `uv sync --extra train`
4. Run the training script: `uv run python scripts/training/train_lora.py`
5. Save the adapter to the network volume

The adapter must be loadable at inference time without retraining. See the script for `--adapter-path` argument.

## Data files

`processed/` is read-only. Never write to it from application or training code. Output goes to `data/embeddings/`, `data/indexed/`, or wherever the script specifies.
