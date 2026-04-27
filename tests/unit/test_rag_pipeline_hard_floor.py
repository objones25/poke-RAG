"""Unit tests for RAGPipeline hard floor feature."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.generation.protocols import GeneratorProtocol
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.protocols import RetrieverProtocol
from src.types import GenerationResult, RetrievalError, RetrievalResult
from tests.conftest import make_chunk


@pytest.mark.unit
class TestRAGPipelineHardFloor:
    """Test RAGPipeline hard floor guard against low-scoring chunks."""

    def _make_pipeline(self, floor: float | None = None) -> RAGPipeline:
        """Create a RAGPipeline with a given hard floor value.

        If floor is None, uses the default floor (-2.0) by not passing settings.
        """
        retriever = MagicMock(spec=RetrieverProtocol)
        generator = MagicMock(spec=GeneratorProtocol)
        if floor is None:
            return RAGPipeline(retriever=retriever, generator=generator)

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
        return RAGPipeline(retriever=retriever, generator=generator, settings=settings)

    def test_all_chunks_below_floor_raises_retrieval_error(self) -> None:
        """Chunks all scoring -5.0 with floor -2.0 → RetrievalError."""
        pipeline = self._make_pipeline(floor=-2.0)
        chunk = make_chunk(score=-5.0)
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=(chunk,),
            query="test",
        )
        with pytest.raises(RetrievalError, match="below relevance floor"):
            pipeline.query("test query")
        # Generator must never be called when chunks are below floor
        pipeline._generator.generate.assert_not_called()

    def test_chunk_above_floor_proceeds_to_generation(self) -> None:
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
        result = pipeline.query("test query")
        assert result.answer == "ok"
        pipeline._generator.generate.assert_called_once()

    def test_very_negative_floor_never_raises(self) -> None:
        """Floor of -99.0 → even chunks at -5.0 proceed."""
        pipeline = self._make_pipeline(floor=-99.0)
        chunk = make_chunk(score=-5.0)
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
        result = pipeline.query("test query")
        assert result.answer == "ok"

    def test_chunk_at_exactly_floor_proceeds(self) -> None:
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
        result = pipeline.query("test query")
        assert result.answer == "ok"

    def test_multiple_chunks_max_score_below_floor_raises(self) -> None:
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
            pipeline.query("test query")
        pipeline._generator.generate.assert_not_called()

    def test_multiple_chunks_max_score_above_floor_proceeds(self) -> None:
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
        result = pipeline.query("test query")
        assert result.answer == "ok"

    def test_default_floor_is_minus_two(self) -> None:
        """RAGPipeline with no settings uses default floor of -2.0."""
        retriever = MagicMock(spec=RetrieverProtocol)
        generator = MagicMock(spec=GeneratorProtocol)
        pipeline = RAGPipeline(retriever=retriever, generator=generator)
        chunk = make_chunk(score=-2.5)
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=(chunk,),
            query="test",
        )
        with pytest.raises(RetrievalError, match="below relevance floor"):
            pipeline.query("test query")
        pipeline._generator.generate.assert_not_called()

    def test_positive_floor_also_works(self) -> None:
        """Positive floor values are supported."""
        pipeline = self._make_pipeline(floor=0.5)
        chunk = make_chunk(score=0.3)
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=(chunk,),
            query="test",
        )
        with pytest.raises(RetrievalError, match="below relevance floor"):
            pipeline.query("test query")

    def test_floor_check_happens_after_empty_check(self) -> None:
        """If no chunks returned, empty error raised before floor check."""
        pipeline = self._make_pipeline(floor=-2.0)
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=(),
            query="test",
        )
        with pytest.raises(RetrievalError, match="no documents"):
            pipeline.query("test query")
        pipeline._generator.generate.assert_not_called()

    def test_floor_error_message_is_informative(self) -> None:
        """RetrievalError message indicates floor threshold."""
        pipeline = self._make_pipeline(floor=-2.0)
        chunk = make_chunk(score=-5.0)
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            documents=(chunk,),
            query="test",
        )
        with pytest.raises(RetrievalError) as exc_info:
            pipeline.query("test query")
        assert "floor" in str(exc_info.value).lower()
