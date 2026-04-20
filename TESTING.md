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
# Everything
uv run pytest

# Skip GPU/slow tests (use this locally)
uv run pytest -m "not gpu and not slow"

# A specific file
uv run pytest tests/unit/test_retriever.py

# A specific test
uv run pytest -k "test_retriever_raises_on_empty_query"

# With coverage
uv run pytest --cov=src --cov-report=term-missing

# HTML coverage report
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Test layout

```
tests/
  conftest.py             shared fixtures (sample docs, mock generator, mock embedder, etc.)
  unit/                   pure logic — no disk, no network, no model
    test_chunker.py           chunking logic per source (pokeapi/smogon/bulbapedia)
    test_retriever.py
    test_reranker.py
    test_context_assembler.py
    test_pipeline.py
    test_query_parser.py
  integration/            real I/O against fixture data and local index
    test_embedder.py          BGE-M3 encoding: dense + sparse shapes, types
    test_qdrant_store.py      collection creation, upsert, hybrid search, payload filtering
    test_rag_pipeline.py
    test_api.py
  e2e/                    full query → answer with real model
    test_pokemon_queries.py
  fixtures/               small sample data files committed to the repo
    sample_bulbapedia.txt     ~5 representative Title: entries
    sample_pokeapi.txt        ~5 representative stat entries
    sample_smogon.txt         ~5 representative tier entries
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

Never load `BAAI/bge-m3` in unit tests. Mock `BGEM3FlagModel.encode` to return correctly shaped outputs:

```python
@pytest.fixture
def mock_embedder(mocker):
    embedder = mocker.MagicMock()
    embedder.encode.return_value = {
        "dense_vecs": [[0.1] * 1024],          # shape: (n_docs, 1024)
        "lexical_weights": [{"pokemon": 0.8}],  # sparse token weights
    }
    return embedder
```

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

## Fixtures

Shared fixtures in `tests/conftest.py`. Small, committed sample files in `tests/fixtures/` — a handful of representative entries from each source, enough to exercise parsing and indexing logic.

Do not read from `processed/` in tests. Copy what you need into `tests/fixtures/`.

## Coverage

Target: **≥80% per module**. Critical paths (retrieval, pipeline, no-fallback logic) should be 100%.

Drops below 80% on any module block the PR. Use `# pragma: no cover` sparingly and only with a comment explaining why.

## CI gates (in order)

1. `uv run ruff check .` — zero errors
2. `uv run mypy src/` — zero errors  
3. `uv run pytest -m "not gpu" --cov=src` — all pass, coverage ≥80%

GPU tests run only on explicit trigger (manual dispatch or `[gpu]` tag in commit message).
