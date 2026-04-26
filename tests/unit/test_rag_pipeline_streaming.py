"""Unit tests for AsyncRAGPipeline.stream_query() — async token streaming."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline.rag_pipeline import AsyncRAGPipeline
from src.retrieval.protocols import AsyncRetrieverProtocol
from src.types import RetrievalError, RetrievalResult
from tests.conftest import make_chunk


def _make_retriever(chunks=None) -> AsyncMock:
    mock = AsyncMock(spec=AsyncRetrieverProtocol)
    chunk = make_chunk(score=0.9) if chunks is None else None
    mock.retrieve.return_value = RetrievalResult(
        documents=tuple([chunk] if chunks is None else chunks),
        query="test",
    )
    return mock


def _make_generator(tokens: list[str] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.stream_generate.return_value = iter(tokens or [])
    return mock


@pytest.mark.unit
class TestStreamQueryValidation:
    @pytest.mark.anyio
    async def test_empty_query_raises_value_error(self):
        pipeline = AsyncRAGPipeline(retriever=_make_retriever(), generator=_make_generator())
        with pytest.raises(ValueError, match="query must not be empty"):
            async for _ in pipeline.stream_query(""):
                pass

    @pytest.mark.anyio
    async def test_whitespace_only_query_raises_value_error(self):
        pipeline = AsyncRAGPipeline(retriever=_make_retriever(), generator=_make_generator())
        with pytest.raises(ValueError, match="query must not be empty"):
            async for _ in pipeline.stream_query("   "):
                pass

    @pytest.mark.anyio
    async def test_no_retrieval_results_raises_retrieval_error(self):
        retriever = _make_retriever(chunks=[])
        pipeline = AsyncRAGPipeline(retriever=retriever, generator=_make_generator())
        with pytest.raises(RetrievalError):
            async for _ in pipeline.stream_query("no results"):
                pass

    @pytest.mark.anyio
    async def test_generator_not_called_on_retrieval_failure(self):
        retriever = _make_retriever(chunks=[])
        gen = _make_generator()
        pipeline = AsyncRAGPipeline(retriever=retriever, generator=gen)
        with pytest.raises(RetrievalError):
            async for _ in pipeline.stream_query("no results"):
                pass
        gen.stream_generate.assert_not_called()


@pytest.mark.unit
class TestStreamQueryYieldsTokens:
    @pytest.mark.anyio
    async def test_yields_tokens_in_order(self):
        tokens = ["Pika", "chu", " is", " Electric"]
        gen = _make_generator(tokens)
        pipeline = AsyncRAGPipeline(retriever=_make_retriever(), generator=gen)
        result = []
        async for token in pipeline.stream_query("tell me about Pikachu"):
            result.append(token)
        assert result == tokens

    @pytest.mark.anyio
    async def test_empty_token_stream_yields_nothing(self):
        pipeline = AsyncRAGPipeline(retriever=_make_retriever(), generator=_make_generator([]))
        result = []
        async for token in pipeline.stream_query("query"):
            result.append(token)
        assert result == []

    @pytest.mark.anyio
    async def test_single_token_yielded(self):
        pipeline = AsyncRAGPipeline(
            retriever=_make_retriever(), generator=_make_generator(["hello"])
        )
        result = []
        async for token in pipeline.stream_query("query"):
            result.append(token)
        assert result == ["hello"]


@pytest.mark.unit
class TestStreamQueryCallsDependencies:
    @pytest.mark.anyio
    async def test_stream_generate_called_with_query_and_chunks(self):
        chunk = make_chunk(score=0.9)
        retriever = _make_retriever(chunks=[chunk])
        gen = _make_generator()
        pipeline = AsyncRAGPipeline(retriever=retriever, generator=gen)
        async for _ in pipeline.stream_query("pikachu moves"):
            pass
        gen.stream_generate.assert_called_once_with("pikachu moves", (chunk,))

    @pytest.mark.anyio
    async def test_retriever_called_with_query(self):
        retriever = _make_retriever()
        pipeline = AsyncRAGPipeline(retriever=retriever, generator=_make_generator())
        async for _ in pipeline.stream_query("pikachu moves"):
            pass
        retriever.retrieve.assert_called_once()
        call_args = retriever.retrieve.call_args
        assert call_args.args[0] == "pikachu moves"

    @pytest.mark.anyio
    async def test_stream_generate_called_once(self):
        gen = _make_generator(["tok"])
        pipeline = AsyncRAGPipeline(retriever=_make_retriever(), generator=gen)
        async for _ in pipeline.stream_query("query"):
            pass
        gen.stream_generate.assert_called_once()


@pytest.mark.unit
class TestStreamQueryWithRouter:
    @pytest.mark.anyio
    async def test_router_routes_query_when_sources_none(self):
        mock_router = MagicMock()
        mock_router.route.return_value = ["bulbapedia"]
        pipeline = AsyncRAGPipeline(
            retriever=_make_retriever(),
            generator=_make_generator(),
            query_router=mock_router,
        )
        async for _ in pipeline.stream_query("pikachu"):
            pass
        mock_router.route.assert_called_once_with("pikachu")

    @pytest.mark.anyio
    async def test_router_not_called_when_sources_provided(self):
        mock_router = MagicMock()
        pipeline = AsyncRAGPipeline(
            retriever=_make_retriever(),
            generator=_make_generator(),
            query_router=mock_router,
        )
        async for _ in pipeline.stream_query("pikachu", sources=["smogon"]):
            pass
        mock_router.route.assert_not_called()
