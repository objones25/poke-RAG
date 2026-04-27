"""Unit tests for AsyncRetriever in src/retrieval/retriever.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.retrieval.protocols import AsyncRetrieverProtocol, AsyncVectorStoreProtocol
from src.retrieval.retriever import AsyncRetriever
from src.retrieval.types import EmbeddingOutput
from src.types import RetrievalError, RetrievalResult
from tests.conftest import make_chunk


def _make_embedder(dense_dim: int = 1024) -> MagicMock:
    mock = MagicMock()
    mock.encode.return_value = EmbeddingOutput(
        dense=[[0.1] * dense_dim],
        sparse=[{1: 0.5, 2: 0.3}],
    )
    return mock


def _make_async_store(chunks=None) -> AsyncMock:
    mock = AsyncMock(spec=AsyncVectorStoreProtocol)
    mock.search.return_value = [make_chunk()] if chunks is None else chunks
    return mock


def _make_reranker(chunks=None) -> MagicMock:
    mock = MagicMock()
    mock.rerank.return_value = [make_chunk()] if chunks is None else chunks
    return mock


@pytest.mark.unit
class TestAsyncRetrieverInit:
    def test_raises_on_zero_candidates_per_source(self) -> None:
        with pytest.raises(ValueError, match="candidates_per_source must be positive"):
            AsyncRetriever(
                embedder=_make_embedder(),
                vector_store=_make_async_store(),
                reranker=_make_reranker(),
                candidates_per_source=0,
            )

    def test_raises_on_negative_candidates_per_source(self) -> None:
        with pytest.raises(ValueError, match="candidates_per_source must be positive"):
            AsyncRetriever(
                embedder=_make_embedder(),
                vector_store=_make_async_store(),
                reranker=_make_reranker(),
                candidates_per_source=-1,
            )

    def test_accepts_valid_candidates_per_source(self) -> None:
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=_make_async_store(),
            reranker=_make_reranker(),
            candidates_per_source=10,
        )
        assert r is not None


@pytest.mark.unit
class TestAsyncRetrieverRetrieve:
    @pytest.mark.anyio
    async def test_returns_retrieval_result(self) -> None:
        chunk = make_chunk()
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=_make_async_store([chunk]),
            reranker=_make_reranker([chunk]),
        )
        result = await r.retrieve("What is Pikachu?")
        assert isinstance(result, RetrievalResult)
        assert result.query == "What is Pikachu?"
        assert len(result.documents) == 1

    @pytest.mark.anyio
    async def test_raises_on_non_positive_top_k(self) -> None:
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=_make_async_store(),
            reranker=_make_reranker(),
        )
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            await r.retrieve("test", top_k=0)

    @pytest.mark.anyio
    async def test_raises_on_empty_sources(self) -> None:
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=_make_async_store(),
            reranker=_make_reranker(),
        )
        with pytest.raises(RetrievalError, match="sources must not be empty"):
            await r.retrieve("test", sources=[])

    @pytest.mark.anyio
    async def test_raises_on_no_candidates(self) -> None:
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=_make_async_store([]),
            reranker=_make_reranker([]),
        )
        with pytest.raises(RetrievalError, match="No candidates found"):
            await r.retrieve("test")

    @pytest.mark.anyio
    async def test_raises_on_empty_reranker_output(self) -> None:
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=_make_async_store([make_chunk()]),
            reranker=_make_reranker([]),
        )
        with pytest.raises(RetrievalError, match="No documents found"):
            await r.retrieve("test")

    @pytest.mark.anyio
    async def test_searches_all_three_sources_by_default(self) -> None:
        store = _make_async_store()
        r = AsyncRetriever(embedder=_make_embedder(), vector_store=store, reranker=_make_reranker())
        await r.retrieve("test")
        searched = {c.kwargs["collection"] for c in store.search.call_args_list}
        assert searched == {"bulbapedia", "pokeapi", "smogon"}

    @pytest.mark.anyio
    async def test_searches_only_specified_sources(self) -> None:
        store = _make_async_store()
        r = AsyncRetriever(embedder=_make_embedder(), vector_store=store, reranker=_make_reranker())
        await r.retrieve("test", sources=["pokeapi"])
        searched = {c.kwargs["collection"] for c in store.search.call_args_list}
        assert searched == {"pokeapi"}

    @pytest.mark.anyio
    async def test_passes_entity_name_to_all_searches(self) -> None:
        store = _make_async_store()
        r = AsyncRetriever(embedder=_make_embedder(), vector_store=store, reranker=_make_reranker())
        await r.retrieve("test", entity_name="Pikachu")
        for call in store.search.call_args_list:
            assert call.kwargs["entity_name"] == "Pikachu"

    @pytest.mark.anyio
    async def test_raises_retrieval_error_on_oserror(self) -> None:
        store = _make_async_store()
        store.search.side_effect = OSError("Connection refused")
        r = AsyncRetriever(embedder=_make_embedder(), vector_store=store, reranker=_make_reranker())
        with pytest.raises(RetrievalError, match="Vector search failed"):
            await r.retrieve("test", sources=["pokeapi"])

    @pytest.mark.anyio
    async def test_raises_retrieval_error_on_reranker_runtime_error(self) -> None:
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=_make_async_store([make_chunk()]),
            reranker=_make_reranker(),
        )
        r._reranker.rerank.side_effect = RuntimeError("Model crashed")
        with pytest.raises(RetrievalError, match="Reranking failed"):
            await r.retrieve("test")

    @pytest.mark.anyio
    async def test_satisfies_async_retriever_protocol(self) -> None:
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=_make_async_store(),
            reranker=_make_reranker(),
        )
        assert isinstance(r, AsyncRetrieverProtocol)


@pytest.mark.unit
class TestAsyncRetrieverExceptionHandling:
    @pytest.mark.anyio
    async def test_single_source_retrieval_error_raises_when_only_source(self) -> None:
        store = AsyncMock(spec=AsyncVectorStoreProtocol)
        store.search.side_effect = RuntimeError("Connection failed")
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=store,
            reranker=_make_reranker(),
            candidates_per_source=10,
        )
        with pytest.raises(RetrievalError):
            await r.retrieve("test", sources=["pokeapi"])

    @pytest.mark.anyio
    async def test_all_sources_raise_exception_group_raises_retrieval_error(self) -> None:
        store = AsyncMock(spec=AsyncVectorStoreProtocol)
        store.search.side_effect = RuntimeError("All failed")
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=store,
            reranker=_make_reranker(),
        )
        with pytest.raises(RetrievalError):
            await r.retrieve("test")

    @pytest.mark.anyio
    async def test_multiple_source_failures_all_reported_in_message(self) -> None:
        """B4: when multiple sources fail, all failures must appear in the error message."""
        call_count = 0

        async def _fail_with_distinct_message(**kwargs: object) -> list:
            nonlocal call_count
            call_count += 1
            src = kwargs.get("collection", f"source_{call_count}")
            raise RuntimeError(f"failed_{src}")

        store = AsyncMock(spec=AsyncVectorStoreProtocol)
        store.search.side_effect = _fail_with_distinct_message
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=store,
            reranker=_make_reranker(),
        )
        with pytest.raises(RetrievalError) as exc_info:
            await r.retrieve("test", sources=["pokeapi", "smogon"])
        msg = str(exc_info.value)
        assert "pokeapi" in msg or "smogon" in msg
        # Both sources must be represented — not silently dropped
        assert "2" in msg or ("pokeapi" in msg and "smogon" in msg)

    @pytest.mark.anyio
    async def test_async_task_group_creates_concurrent_tasks(self) -> None:
        store = AsyncMock(spec=AsyncVectorStoreProtocol)
        store.search.return_value = [make_chunk()]
        r = AsyncRetriever(
            embedder=_make_embedder(),
            vector_store=store,
            reranker=_make_reranker(),
        )
        result = await r.retrieve("test")
        assert store.search.call_count == 3
        assert isinstance(result, RetrievalResult)
