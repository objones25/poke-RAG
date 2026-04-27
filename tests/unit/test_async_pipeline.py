"""Unit tests for AsyncRAGPipeline in src/pipeline/rag_pipeline.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline.rag_pipeline import AsyncRAGPipeline
from src.pipeline.types import PipelineResult
from src.retrieval.protocols import AsyncRetrieverProtocol
from src.types import GenerationResult, RetrievalError, RetrievalResult
from tests.conftest import make_chunk


def _make_async_retriever(chunks=None) -> AsyncMock:
    mock = AsyncMock(spec=AsyncRetrieverProtocol)
    chunk = make_chunk(score=1.5) if chunks is None else None
    mock.retrieve.return_value = RetrievalResult(
        documents=tuple([chunk] if chunks is None else chunks),
        query="test",
    )
    return mock


def _make_generator() -> MagicMock:
    mock = MagicMock()
    mock.generate.return_value = GenerationResult(
        answer="Pikachu is Electric-type.",
        sources_used=("pokeapi",),
        model_name="google/gemma-4-E4B-it",
        num_chunks_used=1,
    )
    return mock


@pytest.mark.unit
class TestAsyncRAGPipeline:
    @pytest.mark.anyio
    async def test_returns_pipeline_result(self) -> None:
        pipeline = AsyncRAGPipeline(
            retriever=_make_async_retriever(),
            generator=_make_generator(),
        )
        result = await pipeline.query("What type is Pikachu?")
        assert isinstance(result, PipelineResult)
        assert result.answer == "Pikachu is Electric-type."
        assert result.query == "What type is Pikachu?"

    @pytest.mark.anyio
    async def test_raises_on_empty_query(self) -> None:
        pipeline = AsyncRAGPipeline(
            retriever=_make_async_retriever(),
            generator=_make_generator(),
        )
        with pytest.raises(ValueError, match="query must not be empty"):
            await pipeline.query("   ")

    @pytest.mark.anyio
    async def test_raises_retrieval_error_on_no_documents(self) -> None:
        pipeline = AsyncRAGPipeline(
            retriever=_make_async_retriever(chunks=[]),
            generator=_make_generator(),
        )
        with pytest.raises(RetrievalError):
            await pipeline.query("test")

    @pytest.mark.anyio
    async def test_generator_not_called_on_retrieval_failure(self) -> None:
        retriever = AsyncMock(spec=AsyncRetrieverProtocol)
        retriever.retrieve.side_effect = RetrievalError("No results")
        gen = _make_generator()
        pipeline = AsyncRAGPipeline(retriever=retriever, generator=gen)
        with pytest.raises(RetrievalError):
            await pipeline.query("test")
        gen.generate.assert_not_called()

    @pytest.mark.anyio
    async def test_confidence_score_is_set(self) -> None:
        pipeline = AsyncRAGPipeline(
            retriever=_make_async_retriever(),
            generator=_make_generator(),
        )
        result = await pipeline.query("test")
        assert result.confidence_score is not None
        assert 0.0 < result.confidence_score < 1.0

    @pytest.mark.anyio
    async def test_confidence_score_uses_max_score_not_position(self) -> None:
        """B3: async pipeline confidence uses max score regardless of chunk ordering."""
        import math

        # chunks intentionally NOT in descending score order
        chunks = [make_chunk(score=2.0), make_chunk(score=5.0)]
        retriever = _make_async_retriever(chunks=chunks)
        pipeline = AsyncRAGPipeline(retriever=retriever, generator=_make_generator())

        result = await pipeline.query("test")
        expected = 1.0 / (1.0 + math.exp(-5.0))
        assert result.confidence_score == pytest.approx(expected)

    @pytest.mark.anyio
    async def test_passes_sources_to_retriever(self) -> None:
        retriever = _make_async_retriever()
        pipeline = AsyncRAGPipeline(retriever=retriever, generator=_make_generator())
        await pipeline.query("test", sources=["pokeapi"])
        retriever.retrieve.assert_called_once()
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["sources"] == ["pokeapi"]

    @pytest.mark.anyio
    async def test_passes_entity_name_to_retriever(self) -> None:
        retriever = _make_async_retriever()
        pipeline = AsyncRAGPipeline(retriever=retriever, generator=_make_generator())
        await pipeline.query("test", entity_name="Pikachu")
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["entity_name"] == "Pikachu"

    @pytest.mark.anyio
    async def test_query_router_routes_when_sources_none(self) -> None:
        router = MagicMock()
        router.route.return_value = ["pokeapi"]
        retriever = _make_async_retriever()
        pipeline = AsyncRAGPipeline(
            retriever=retriever, generator=_make_generator(), query_router=router
        )
        await pipeline.query("test")
        router.route.assert_called_once_with("test")
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["sources"] == ["pokeapi"]

    @pytest.mark.anyio
    async def test_query_router_not_called_when_sources_provided(self) -> None:
        router = MagicMock()
        router.route.return_value = ["pokeapi"]
        pipeline = AsyncRAGPipeline(
            retriever=_make_async_retriever(),
            generator=_make_generator(),
            query_router=router,
        )
        await pipeline.query("test", sources=["smogon"])
        router.route.assert_not_called()

    @pytest.mark.anyio
    async def test_concurrent_queries_resolve_independently(self) -> None:
        import asyncio

        retriever1 = _make_async_retriever()
        retriever2 = _make_async_retriever()
        gen1 = _make_generator()
        gen2 = _make_generator()

        pipeline1 = AsyncRAGPipeline(retriever=retriever1, generator=gen1)
        pipeline2 = AsyncRAGPipeline(retriever=retriever2, generator=gen2)

        results = await asyncio.gather(
            pipeline1.query("What type is Pikachu?"),
            pipeline2.query("What type is Charizard?"),
        )

        assert len(results) == 2
        queries = ["What type is Pikachu?", "What type is Charizard?"]
        assert all(r.query in queries for r in results)

    @pytest.mark.anyio
    async def test_retrieval_error_propagates_not_caught(self) -> None:
        retriever = AsyncMock(spec=AsyncRetrieverProtocol)
        retriever.retrieve.side_effect = RetrievalError("Connection timeout")
        gen = _make_generator()
        pipeline = AsyncRAGPipeline(retriever=retriever, generator=gen)

        with pytest.raises(RetrievalError, match="Connection timeout"):
            await pipeline.query("test")

    @pytest.mark.anyio
    async def test_timeout_integration_asyncio_timeout(self) -> None:
        import asyncio

        retriever = AsyncMock(spec=AsyncRetrieverProtocol)

        async def slow_retrieve(*args, **kwargs):
            await asyncio.sleep(0.05)
            return RetrievalResult(documents=tuple([make_chunk(score=1.5)]), query="test")

        retriever.retrieve.side_effect = slow_retrieve
        gen = _make_generator()
        pipeline = AsyncRAGPipeline(retriever=retriever, generator=gen)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(pipeline.query("test"), timeout=0.01)
