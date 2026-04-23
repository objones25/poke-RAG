# TESTING.md

## Stack

- **Runner**: `pytest`
- **Mocking**: `pytest-mock` (`mocker` fixture)
- **Coverage**: `pytest-cov`
- **Type checking**: `mypy` (not a test runner, but part of the quality gate)

Install dev dependencies:

```bash
uv sync --extra dev
```

## Running tests

```bash
# Everything (local dev — skips GPU tests)
uv run pytest -m "not gpu"

# All tests including GPU
uv run pytest

# Skip GPU and slow tests (fastest local iteration)
uv run pytest -m "not gpu and not slow"

# A specific file
uv run pytest tests/unit/test_retriever.py

# A specific test
uv run pytest -k "test_retriever_raises_on_empty_query"

# With coverage report in terminal
uv run pytest --cov=src --cov-report=term-missing -m "not gpu"

# HTML coverage report
uv run pytest --cov=src --cov-report=html -m "not gpu"
open htmlcov/index.html
```

**Note**: GPU tests are skipped by default with `-m "not gpu"` when run locally. They require a CUDA-capable device and run only on explicit trigger (via `-m gpu` or in CI with a GPU runner). Rate limiting is disabled automatically for all tests via the autouse fixture in `conftest.py`.

## Test layout

```text
tests/
  conftest.py             shared fixtures (make_chunk, make_retrieval_result, make_generation_result, autouse rate limit disable)
  unit/                   pure logic — no disk, no network, no model loading
    test_chunker.py
    test_retriever.py
    test_reranker.py
    test_context_assembler.py
    test_pipeline.py
    test_query_parser.py
    test_query_router.py                 QueryRouter keyword-based routing (~79 tests)
    test_query_transformer.py            HyDE and passthrough transformers
    test_config.py                       Settings validation, device detection, SecretStr masking
    test_inference.py                    Inferencer model lifecycle
    test_loader.py                       ModelLoader coverage
    test_api.py                          FastAPI route handlers with mocks
    test_embedder.py                     BGEEmbedder API shape (mocks only)
    test_vector_store.py                 VectorStore interface and argument shapes
    test_generator.py
    test_models.py
    test_prompt_builder.py
    test_sampler.py
  integration/            real I/O, mocked models, or fixture data
    test_embedder.py                     BGE-M3 real model: dense + sparse shapes, types (marked @pytest.mark.slow)
    test_qdrant_store.py                 Qdrant client API shapes (uses MagicMock client, not real Qdrant)
    test_rag_pipeline.py
    test_api.py                          FastAPI with mocked pipeline
    test_api_lifespan.py                 FastAPI lifespan (startup/shutdown/failure)
    test_build_index.py
    test_generation_pipeline.py
    test_generate_sft_data.py
  e2e/                    full query → answer with real Qdrant and real models
    test_pokemon_queries.py              (requires QDRANT_URL and CUDA GPU; marked @pytest.mark.gpu)
  fixtures/               small sample data files committed to the repo
    sample_bulbapedia.txt                ~5 representative Title: entries
    sample_pokeapi.txt                   ~5 representative stat entries
    sample_smogon.txt                    ~5 representative tier entries
```

## Markers

Registered in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "unit: no I/O, no model",
    "integration: real I/O against fixture data",
    "e2e: full pipeline with real model",
    "gpu: requires a CUDA device",
    "slow: takes more than 5 seconds",
]
```

Tag every test:

```python
@pytest.mark.unit
def test_context_assembler_truncates_at_token_limit(): ...

@pytest.mark.integration
@pytest.mark.slow
def test_vector_store_roundtrip(): ...

@pytest.mark.e2e
@pytest.mark.gpu
def test_charizard_stats_query(): ...
```

## TDD workflow

1. Write a failing test that describes the behaviour
2. Run it: `uv run pytest -k "your_test"` — confirm it fails
3. Write the minimum implementation to pass it
4. Refactor
5. Commit test + implementation together

Never commit implementation without a test. If you're modifying existing code that lacks a test, add the test first.

## Unit test rules

- No filesystem access, no network, no model loading
- Mock all external dependencies with `pytest-mock`
- Each test covers one behaviour
- Name pattern: `test_<subject>_<condition>_<expected>`

```python
# Good
def test_retriever_raises_retrieval_error_on_empty_query(): ...
def test_context_assembler_returns_empty_string_when_no_docs(): ...

# Bad
def test_retrieval(): ...
def test_it_works(): ...
```

## Mocking the embedder

Never load `BAAI/bge-m3` in unit tests. Mock at the `BGEEmbedder` level — `BGEEmbedder.encode()` returns an `EmbeddingOutput` dataclass, not the raw `BGEM3FlagModel` dict:

```python
from src.retrieval.embedder import EmbeddingOutput

@pytest.fixture
def mock_embedder(mocker):
    embedder = mocker.MagicMock()
    embedder.encode.return_value = EmbeddingOutput(
        dense=[[0.1] * 1024],          # shape: (n_docs, 1024)
        sparse=[{"pokemon": 0.8}],     # sparse token weights
    )
    return embedder
```

Do **not** mock `BGEM3FlagModel.encode` directly with the raw `{"dense_vecs": ..., "lexical_weights": ...}` dict — that is the internal FlagEmbedding format, not what the rest of the system sees.

Integration tests in `test_embedder.py` that call the real model are marked `@pytest.mark.slow` and `@pytest.mark.integration`. They verify output shapes and types, not semantic content.

## Mocking the model

Never load `gemma-4-E4B-it` in unit or integration tests:

```python
@pytest.fixture
def mock_generator(mocker):
    gen = mocker.MagicMock()
    gen.generate.return_value = "Charizard is a Fire/Flying type."
    return gen
```

E2E tests that need real generation are marked `@pytest.mark.gpu` and excluded from CI unless a GPU runner is configured.

## API integration tests and rate limiting

Rate limiting is disabled by default for all tests via the autouse fixture `_disable_rate_limiting` in `conftest.py`, which sets `RATE_LIMIT_ENABLED=false`. This prevents the `RateLimitMiddleware` (20 req/min/IP on `/query`) from blocking rapid test requests.

If you need to test rate limiting behavior itself, re-enable it in that specific test:

```python
def test_rate_limiting_blocks_excess_requests(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    # test code here
```

## The no-fallback invariant

The pipeline must never call the generator when retrieval fails. This must have an explicit test:

```python
@pytest.mark.unit
def test_pipeline_does_not_call_generator_when_retrieval_fails(mocker):
    retriever = mocker.MagicMock()
    retriever.retrieve.side_effect = RetrievalError("index unavailable")
    generator = mocker.MagicMock()

    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    with pytest.raises(RetrievalError):
        pipeline.query("What are Mewtwo's base stats?")

    generator.generate.assert_not_called()
```

## Test count

Approximately **~613 tests** total across unit, integration, and e2e markers.

## Fixtures

Shared fixtures in `tests/conftest.py`:
- **`make_chunk`** — factory fixture for creating `RetrievedChunk` objects with customizable fields
- **`make_retrieval_result`** — factory fixture for `RetrievalResult` objects
- **`make_generation_result`** — factory fixture for `GenerationResult` objects
- **`_disable_rate_limiting`** (autouse) — disables rate limiting for all tests to prevent spurious failures

Small sample data files in `tests/fixtures/` — a handful of representative entries from each source, enough to exercise parsing and indexing logic.

Do not read from `processed/` in tests. Copy what you need into `tests/fixtures/`.

## Coverage

Target: **≥80% per module**. Critical paths (retrieval, pipeline, no-fallback logic) should be 100%.

Drops below 80% on any module block the PR. Use `# pragma: no cover` sparingly and only with a comment explaining why.

## Environment variables for E2E and integration tests

E2E tests require real Qdrant and real models:

```bash
export QDRANT_URL="http://localhost:6333"      # Qdrant server (or remote hosted instance)
export QDRANT_API_KEY="optional-api-key"       # Optional if Qdrant requires authentication
export DEVICE="cuda"                            # or "cpu", "mps" (auto-detected by default)
export GEN_MODEL="google/gemma-4-E4B-it"       # Generation model (default)
export EMBED_MODEL="BAAI/bge-m3"               # Embedding model (default)
export RERANK_MODEL="BAAI/bge-reranker-v2-m3" # Reranking model (default)
```

Set `PYTEST_E2E_CLEANUP=1` to delete e2e Qdrant collections after the run (useful in CI).

## CI gates (in order)

1. `uv run ruff check .` — zero errors
2. `uv run mypy src/` — zero errors
3. `uv run pytest -m "not gpu" --cov=src` — all pass, coverage ≥80%

GPU tests run only on explicit trigger (manual dispatch, GPU runner, or `[gpu]` tag in commit message).
