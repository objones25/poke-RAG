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
