"""Unit tests for cache integration in RAGPipeline and AsyncRAGPipeline."""

from __future__ import annotations

import pytest

from src.pipeline.types import PipelineResult
from src.retrieval.cache import CacheKey, to_cache_dict
from src.types import GenerationResult, RetrievalResult
from tests.conftest import make_chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rr(*chunks) -> RetrievalResult:
    return RetrievalResult(documents=tuple(chunks), query="q")


def _gr(answer: str = "answer") -> GenerationResult:
    return GenerationResult(
        answer=answer,
        sources_used=("pokeapi",),
        model_name="gemma-4-E4B-it",
        num_chunks_used=1,
    )


def _pr(answer: str = "answer", query: str = "q") -> PipelineResult:
    return PipelineResult(
        answer=answer,
        sources_used=("pokeapi",),
        num_chunks_used=1,
        model_name="gemma-4-E4B-it",
        query=query,
    )


# ---------------------------------------------------------------------------
# Sync RAGPipeline cache tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRAGPipelineCacheHit:
    def test_cache_hit_skips_retrieval(self, mocker) -> None:
        """On cache hit the retriever must not be called."""
        from src.pipeline.rag_pipeline import RAGPipeline

        cached = _pr()
        cache = mocker.AsyncMock()
        cache.get.return_value = to_cache_dict(cached)

        retriever = mocker.MagicMock()
        generator = mocker.MagicMock()

        pipeline = RAGPipeline(retriever=retriever, generator=generator, cache=cache)
        result = pipeline.query("q", top_k=5, sources=None, entity_name=None)

        retriever.retrieve.assert_not_called()
        generator.generate.assert_not_called()
        assert result.answer == cached.answer

    def test_cache_miss_calls_retrieval_and_stores(self, mocker) -> None:
        """On cache miss the full pipeline runs and the result is stored."""
        from src.pipeline.rag_pipeline import RAGPipeline

        cache = mocker.AsyncMock()
        cache.get.return_value = None

        chunk = make_chunk(score=0.9)
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _rr(chunk)

        generator = mocker.MagicMock()
        generator.generate.return_value = _gr()

        pipeline = RAGPipeline(retriever=retriever, generator=generator, cache=cache)
        pipeline.query("q", top_k=5, sources=None, entity_name=None)

        retriever.retrieve.assert_called_once()
        generator.generate.assert_called_once()
        cache.set.assert_called_once()

    def test_cache_key_uses_make_rag_key(self, mocker) -> None:
        """The key passed to cache.get must match CacheKey.make_rag_key."""
        from src.pipeline.rag_pipeline import RAGPipeline

        cache = mocker.AsyncMock()
        cache.get.return_value = to_cache_dict(_pr())

        pipeline = RAGPipeline(
            retriever=mocker.MagicMock(),
            generator=mocker.MagicMock(),
            cache=cache,
        )
        pipeline.query("q", top_k=3, sources=["pokeapi"], entity_name="pikachu")

        expected_key = CacheKey.make_rag_key("q", ["pokeapi"], "pikachu", 3)
        cache.get.assert_called_once_with(expected_key)

    def test_no_cache_when_none(self, mocker) -> None:
        """When cache=None the pipeline runs without any cache calls."""
        from src.pipeline.rag_pipeline import RAGPipeline

        chunk = make_chunk(score=0.9)
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _rr(chunk)

        generator = mocker.MagicMock()
        generator.generate.return_value = _gr()

        pipeline = RAGPipeline(retriever=retriever, generator=generator, cache=None)
        result = pipeline.query("q")

        retriever.retrieve.assert_called_once()
        assert result.answer == "answer"

    def test_cache_error_on_get_degrades_gracefully(self, mocker) -> None:
        """If cache.get raises, pipeline still runs normally."""
        from src.pipeline.rag_pipeline import RAGPipeline

        cache = mocker.AsyncMock()
        cache.get.side_effect = Exception("redis down")

        chunk = make_chunk(score=0.9)
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _rr(chunk)

        generator = mocker.MagicMock()
        generator.generate.return_value = _gr()

        pipeline = RAGPipeline(retriever=retriever, generator=generator, cache=cache)
        result = pipeline.query("q")

        assert result.answer == "answer"

    def test_cache_error_on_set_degrades_gracefully(self, mocker) -> None:
        """If cache.set raises, the already-computed result is still returned."""
        from src.pipeline.rag_pipeline import RAGPipeline

        cache = mocker.AsyncMock()
        cache.get.return_value = None
        cache.set.side_effect = Exception("write failure")

        chunk = make_chunk(score=0.9)
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _rr(chunk)

        generator = mocker.MagicMock()
        generator.generate.return_value = _gr()

        pipeline = RAGPipeline(retriever=retriever, generator=generator, cache=cache)
        result = pipeline.query("q")

        assert result.answer == "answer"


# ---------------------------------------------------------------------------
# Async RAGPipeline cache tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
class TestAsyncRAGPipelineCacheHit:
    async def test_cache_hit_skips_retrieval(self, mocker) -> None:
        from src.pipeline.rag_pipeline import AsyncRAGPipeline

        cached = _pr()
        cache = mocker.AsyncMock()
        cache.get.return_value = to_cache_dict(cached)

        retriever = mocker.AsyncMock()
        generator = mocker.MagicMock()

        pipeline = AsyncRAGPipeline(retriever=retriever, generator=generator, cache=cache)
        result = await pipeline.query("q", top_k=5, sources=None, entity_name=None)

        retriever.retrieve.assert_not_called()
        generator.generate.assert_not_called()
        assert result.answer == cached.answer

    async def test_cache_miss_calls_retrieval_and_stores(self, mocker) -> None:
        from src.pipeline.rag_pipeline import AsyncRAGPipeline

        cache = mocker.AsyncMock()
        cache.get.return_value = None

        chunk = make_chunk(score=0.9)
        retriever = mocker.AsyncMock()
        retriever.retrieve.return_value = _rr(chunk)

        generator = mocker.MagicMock()
        generator.generate.return_value = _gr()

        pipeline = AsyncRAGPipeline(retriever=retriever, generator=generator, cache=cache)
        await pipeline.query("q", top_k=5, sources=None, entity_name=None)

        retriever.retrieve.assert_called_once()
        cache.set.assert_called_once()

    async def test_stream_query_does_not_touch_cache(self, mocker) -> None:
        """stream_query must never read from or write to the cache."""
        from src.pipeline.rag_pipeline import AsyncRAGPipeline

        cache = mocker.AsyncMock()

        chunk = make_chunk(score=0.9)
        retriever = mocker.AsyncMock()
        retriever.retrieve.return_value = _rr(chunk)

        generator = mocker.MagicMock()
        generator.stream_generate.return_value = iter(["tok1", "tok2"])

        pipeline = AsyncRAGPipeline(retriever=retriever, generator=generator, cache=cache)
        tokens = [t async for t in pipeline.stream_query("q")]

        assert tokens == ["tok1", "tok2"]
        cache.get.assert_not_called()
        cache.set.assert_not_called()

    async def test_cache_error_degrades_gracefully(self, mocker) -> None:
        from src.pipeline.rag_pipeline import AsyncRAGPipeline

        cache = mocker.AsyncMock()
        cache.get.side_effect = Exception("redis down")

        chunk = make_chunk(score=0.9)
        retriever = mocker.AsyncMock()
        retriever.retrieve.return_value = _rr(chunk)

        generator = mocker.MagicMock()
        generator.generate.return_value = _gr()

        pipeline = AsyncRAGPipeline(retriever=retriever, generator=generator, cache=cache)
        result = await pipeline.query("q")

        assert result.answer == "answer"
