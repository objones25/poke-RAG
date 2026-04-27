"""Unit tests for AsyncRAGPipeline hard floor feature."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Settings
from src.generation.protocols import GeneratorProtocol
from src.pipeline.rag_pipeline import AsyncRAGPipeline
from src.retrieval.protocols import AsyncRetrieverProtocol
from src.types import GenerationResult, RetrievalError, RetrievalResult
from tests.conftest import make_chunk


@pytest.mark.unit
class TestAsyncRAGPipelineHardFloor:
    """Test AsyncRAGPipeline hard floor guard against low-scoring chunks."""

    def _make_pipeline(self, floor: float | None = None) -> AsyncRAGPipeline:
        """Create an AsyncRAGPipeline with a given hard floor value.

        If floor is None, uses the default floor (-2.0) by not passing settings.
        """
        retriever = AsyncMock(spec=AsyncRetrieverProtocol)
        generator = MagicMock(spec=GeneratorProtocol)
        if floor is None:
            return AsyncRAGPipeline(retriever=retriever, generator=generator)

        # For non-default floors, create a Settings with just the floor set
        # We need all required fields; use minimal valid values
        settings = Settings(
            qdrant_url="http://localhost:6333",
            qdrant_api_key=None,
            embed_model="BAAI/bge-m3",
            rerank_model="BAAI/bge-reranker-v2-m3",
            gen_model="google/gemma-4-E4B-it",
            temperature=0.7,
            max_new_tokens=512,
            top_p=0.9,
            do_sample=True,
            tokenizer_max_length=8192,
            return_tensors="pt",
            truncation=True,
            device="cpu",
            retrieval_hard_floor=floor,
        )
        return AsyncRAGPipeline(retriever=retriever, generator=generator, settings=settings)

    @pytest.mark.anyio
    async def test_all_chunks_below_floor_raises_retrieval_error(self) -> None:
        """Chunks all scoring -5.0 with floor -2.0 → RetrievalError."""
        pipeline = self._make_pipeline(floor=-2.0)
        chunk = make_chunk(score=-5.0)
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=(chunk,),
            query="test",
        )
        with pytest.raises(RetrievalError, match="below relevance floor"):
            await pipeline.query("test query")
        # Generator must never be called when chunks are below floor
        pipeline._generator.generate.assert_not_called()

    @pytest.mark.anyio
    async def test_chunk_above_floor_proceeds_to_generation(self) -> None:
        """Chunk scoring -1.5 with floor -2.0 → generation proceeds."""
        pipeline = self._make_pipeline(floor=-2.0)
        chunk = make_chunk(score=-1.5)
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=(chunk,),
            query="test",
        )
        pipeline._generator.generate.return_value = GenerationResult(
            answer="ok",
            sources_used=("pokeapi",),
            model_name="test",
            num_chunks_used=1,
        )
        result = await pipeline.query("test query")
        assert result.answer == "ok"
        pipeline._generator.generate.assert_called_once()

    @pytest.mark.anyio
    async def test_stream_query_below_floor_raises(self) -> None:
        """stream_query with max score below floor → RetrievalError."""
        pipeline = self._make_pipeline(floor=-2.0)
        chunk = make_chunk(score=-5.0)
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=(chunk,),
            query="test",
        )
        with pytest.raises(RetrievalError, match="below relevance floor"):
            async for _ in pipeline.stream_query("test query"):
                pass

    @pytest.mark.anyio
    async def test_stream_query_above_floor_produces_tokens(self) -> None:
        """stream_query with max score above floor → generator called, tokens produced."""
        pipeline = self._make_pipeline(floor=-2.0)
        chunk = make_chunk(score=-1.5)
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=(chunk,),
            query="test",
        )

        # Mock the streaming generator
        streaming_gen = MagicMock()
        streaming_gen.stream_generate.return_value = iter(["tok1", "tok2", "tok3"])
        pipeline._generator = streaming_gen

        tokens = []
        async for token in pipeline.stream_query("test query"):
            tokens.append(token)

        assert tokens == ["tok1", "tok2", "tok3"]
        streaming_gen.stream_generate.assert_called_once()

    @pytest.mark.anyio
    async def test_default_floor_is_minus_two(self) -> None:
        """AsyncRAGPipeline with no settings uses default floor of -2.0."""
        pipeline = self._make_pipeline(floor=None)
        chunk = make_chunk(score=-2.5)
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=(chunk,),
            query="test",
        )
        with pytest.raises(RetrievalError, match="below relevance floor"):
            await pipeline.query("test query")
        pipeline._generator.generate.assert_not_called()

    @pytest.mark.anyio
    async def test_chunk_at_exactly_floor_proceeds(self) -> None:
        """Chunk at exactly -2.0 with floor -2.0 → does NOT raise (strict <)."""
        pipeline = self._make_pipeline(floor=-2.0)
        chunk = make_chunk(score=-2.0)
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=(chunk,),
            query="test",
        )
        pipeline._generator.generate.return_value = GenerationResult(
            answer="ok",
            sources_used=("pokeapi",),
            model_name="test",
            num_chunks_used=1,
        )
        result = await pipeline.query("test query")
        assert result.answer == "ok"

    @pytest.mark.anyio
    async def test_multiple_chunks_max_score_below_floor_raises(self) -> None:
        """Multiple chunks with max score below floor → RetrievalError."""
        pipeline = self._make_pipeline(floor=-2.0)
        chunks = (
            make_chunk(score=-5.0),
            make_chunk(score=-3.0),
            make_chunk(score=-4.0),
        )
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=chunks,
            query="test",
        )
        with pytest.raises(RetrievalError, match="below relevance floor"):
            await pipeline.query("test query")
        pipeline._generator.generate.assert_not_called()

    @pytest.mark.anyio
    async def test_multiple_chunks_max_score_above_floor_proceeds(self) -> None:
        """Multiple chunks with max score above floor → generation proceeds."""
        pipeline = self._make_pipeline(floor=-2.0)
        chunks = (
            make_chunk(score=-5.0),
            make_chunk(score=-1.5),
            make_chunk(score=-3.0),
        )
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=chunks,
            query="test",
        )
        pipeline._generator.generate.return_value = GenerationResult(
            answer="ok",
            sources_used=("pokeapi",),
            model_name="test",
            num_chunks_used=3,
        )
        result = await pipeline.query("test query")
        assert result.answer == "ok"
