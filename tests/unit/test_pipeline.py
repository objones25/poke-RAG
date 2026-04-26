"""Unit tests for src/pipeline/rag_pipeline.py — RED until implemented."""

from __future__ import annotations

import pytest

from src.types import GenerationResult, RetrievalError, RetrievalResult, RetrievedChunk, Source
from tests.conftest import make_chunk as _make_chunk


def _make_retrieval_result(
    chunks: tuple[RetrievedChunk, ...] = (),
    query: str = "test query",
) -> RetrievalResult:
    return RetrievalResult(documents=chunks, query=query)


def _make_generation_result(
    answer: str = "Pikachu is Electric-type.",
    sources_used: tuple[Source, ...] = ("pokeapi",),
    model_name: str = "google/gemma-4-E4B-it",
    num_chunks_used: int = 1,
) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        sources_used=sources_used,
        model_name=model_name,
        num_chunks_used=num_chunks_used,
    )


@pytest.mark.unit
class TestRAGPipelineNoFallbackInvariant:
    def test_does_not_call_generator_when_retrieval_fails(self, mocker) -> None:
        """The core invariant: generator must never be called if retrieval raises."""
        retriever = mocker.MagicMock()
        retriever.retrieve.side_effect = RetrievalError("index unavailable")
        generator = mocker.MagicMock()

        from src.pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        with pytest.raises(RetrievalError):
            pipeline.query("What are Mewtwo's base stats?")

        generator.generate.assert_not_called()

    def test_retrieval_error_propagates_unchanged(self, mocker) -> None:
        retriever = mocker.MagicMock()
        retriever.retrieve.side_effect = RetrievalError("timeout")
        generator = mocker.MagicMock()

        from src.pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        with pytest.raises(RetrievalError, match="timeout"):
            pipeline.query("Any query.")


@pytest.mark.unit
class TestRAGPipelineQuery:
    def _make_pipeline(self, mocker, chunks=None, answer="Pikachu is Electric-type."):
        from src.pipeline.rag_pipeline import RAGPipeline

        if chunks is None:
            chunks = (_make_chunk(),)

        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=chunks)

        # Extract unique sources from chunks and sort them
        sources_from_chunks = tuple(sorted({c.source for c in chunks}))

        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result(
            answer=answer,
            sources_used=sources_from_chunks,
            num_chunks_used=len(chunks),
        )

        return RAGPipeline(retriever=retriever, generator=generator), retriever, generator

    def test_returns_pipeline_result(self, mocker) -> None:
        from src.pipeline.types import PipelineResult

        pipeline, _, _ = self._make_pipeline(mocker)
        result = pipeline.query("What type is Pikachu?")
        assert isinstance(result, PipelineResult)

    def test_answer_matches_generator_output(self, mocker) -> None:
        pipeline, _, _ = self._make_pipeline(mocker, answer="Charizard is Fire/Flying.")
        result = pipeline.query("What type is Charizard?")
        assert result.answer == "Charizard is Fire/Flying."

    def test_model_name_comes_from_generation_result(self, mocker) -> None:
        pipeline, _, _ = self._make_pipeline(mocker)
        result = pipeline.query("Any question.")
        assert result.model_name == "google/gemma-4-E4B-it"

    def test_query_stored_in_result(self, mocker) -> None:
        pipeline, _, _ = self._make_pipeline(mocker)
        result = pipeline.query("What are Bulbasaur's types?")
        assert result.query == "What are Bulbasaur's types?"

    def test_num_chunks_used_matches_retrieved_count(self, mocker) -> None:
        chunks = tuple(_make_chunk(original_doc_id=f"doc_{i}", chunk_index=i) for i in range(4))
        pipeline, _, _ = self._make_pipeline(mocker, chunks=chunks)
        result = pipeline.query("Any question.")
        assert result.num_chunks_used == 4

    def test_sources_used_are_deduped(self, mocker) -> None:
        chunks = (
            _make_chunk(source="pokeapi", original_doc_id="doc_0", chunk_index=0),
            _make_chunk(source="pokeapi", original_doc_id="doc_1", chunk_index=1),
            _make_chunk(source="bulbapedia", original_doc_id="doc_2", chunk_index=2),
        )
        pipeline, _, _ = self._make_pipeline(mocker, chunks=chunks)
        result = pipeline.query("Any question.")
        assert set(result.sources_used) == {"pokeapi", "bulbapedia"}
        assert len(result.sources_used) == 2

    def test_sources_used_are_sorted(self, mocker) -> None:
        chunks = (
            _make_chunk(source="smogon", original_doc_id="doc_0", chunk_index=0),
            _make_chunk(source="bulbapedia", original_doc_id="doc_1", chunk_index=1),
            _make_chunk(source="pokeapi", original_doc_id="doc_2", chunk_index=2),
        )
        pipeline, _, _ = self._make_pipeline(mocker, chunks=chunks)
        result = pipeline.query("Any question.")
        assert list(result.sources_used) == sorted(result.sources_used)

    def test_sources_used_is_tuple(self, mocker) -> None:
        pipeline, _, _ = self._make_pipeline(mocker)
        result = pipeline.query("Any question.")
        assert isinstance(result.sources_used, tuple)

    def test_retriever_called_with_query(self, mocker) -> None:
        pipeline, retriever, _ = self._make_pipeline(mocker)
        pipeline.query("How fast is Jolteon?")
        retriever.retrieve.assert_called_once()
        call_kwargs = retriever.retrieve.call_args
        assert call_kwargs[0][0] == "How fast is Jolteon?"

    def test_retriever_receives_sources_kwarg(self, mocker) -> None:
        pipeline, retriever, _ = self._make_pipeline(mocker)
        pipeline.query("Stats only.", sources=["pokeapi"])
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["sources"] == ["pokeapi"]

    def test_retriever_sources_none_by_default(self, mocker) -> None:
        pipeline, retriever, _ = self._make_pipeline(mocker)
        pipeline.query("Any question.")
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["sources"] is None

    def test_generator_called_with_query_and_chunks(self, mocker) -> None:
        chunks = (_make_chunk(original_doc_id="doc_0"),)
        pipeline, _, generator = self._make_pipeline(mocker, chunks=chunks)
        pipeline.query("What type is Pikachu?")
        generator.generate.assert_called_once_with("What type is Pikachu?", chunks)

    def test_generator_error_propagates(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=(_make_chunk(),))
        generator = mocker.MagicMock()
        generator.generate.side_effect = ValueError("Invalid chunks provided")

        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        with pytest.raises(ValueError, match="Invalid chunks"):
            pipeline.query("What type is Pikachu?")


@pytest.mark.unit
class TestRAGPipelineEntityName:
    def test_entity_name_forwarded_to_retriever(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=(_make_chunk(),))
        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result()
        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        pipeline.query("What is Pikachu?", entity_name="Pikachu")
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["entity_name"] == "Pikachu"

    def test_entity_name_none_by_default(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=(_make_chunk(),))
        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result()
        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        pipeline.query("Any question.")
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["entity_name"] is None


@pytest.mark.unit
class TestRAGPipelineConfidenceScore:
    def test_confidence_score_is_in_zero_one_range(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        chunks = (
            _make_chunk(original_doc_id="doc_0", chunk_index=0, score=10.0),
            _make_chunk(original_doc_id="doc_1", chunk_index=1, score=-5.0),
        )
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=chunks)
        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result(num_chunks_used=2)
        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        result = pipeline.query("Any question.")
        assert result.confidence_score is not None
        assert 0.0 <= result.confidence_score <= 1.0

    def test_confidence_score_higher_for_better_chunks(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        def _pipeline_with_score(score: float, mocker):
            chunks = (_make_chunk(original_doc_id="doc_0", chunk_index=0, score=score),)
            retriever = mocker.MagicMock()
            retriever.retrieve.return_value = _make_retrieval_result(chunks=chunks)
            generator = mocker.MagicMock()
            generator.generate.return_value = _make_generation_result()
            return RAGPipeline(retriever=retriever, generator=generator)

        high = _pipeline_with_score(5.0, mocker).query("q.").confidence_score
        low = _pipeline_with_score(-5.0, mocker).query("q.").confidence_score
        assert high is not None and low is not None
        assert high > low

    def test_confidence_score_uses_max_score_descending_order(self, mocker) -> None:
        import math

        from src.pipeline.rag_pipeline import RAGPipeline

        chunks = (
            _make_chunk(original_doc_id="doc_0", chunk_index=0, score=5.0),
            _make_chunk(original_doc_id="doc_1", chunk_index=1, score=2.0),
        )
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=chunks)
        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result(num_chunks_used=2)
        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        result = pipeline.query("Any question.")
        expected_confidence = 1.0 / (1.0 + math.exp(-5.0))
        assert result.confidence_score == pytest.approx(expected_confidence)

    def test_confidence_score_uses_max_score_not_position(self, mocker) -> None:
        """B3: confidence uses max score regardless of chunk ordering."""
        import math

        from src.pipeline.rag_pipeline import RAGPipeline

        # chunks intentionally NOT in descending score order
        chunks = (
            _make_chunk(original_doc_id="doc_0", chunk_index=0, score=2.0),
            _make_chunk(original_doc_id="doc_1", chunk_index=1, score=5.0),
        )
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=chunks)
        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result(num_chunks_used=2)
        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        result = pipeline.query("Any question.")
        expected_confidence = 1.0 / (1.0 + math.exp(-5.0))  # sigmoid of MAX score
        assert result.confidence_score == pytest.approx(expected_confidence)

    def test_confidence_score_stable_for_large_negative_score(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        chunks = (_make_chunk(original_doc_id="doc_0", chunk_index=0, score=-800.0),)
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=chunks)
        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result()
        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        result = pipeline.query("Any question.")
        assert result.confidence_score is not None
        assert 0.0 <= result.confidence_score <= 1.0

    def test_confidence_score_none_when_no_chunks(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        # With our new guard, empty chunks now raise RetrievalError instead
        # This test verifies that the error is raised
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=())
        generator = mocker.MagicMock()
        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        with pytest.raises(RetrievalError, match="no documents"):
            pipeline.query("Any question.")

    def test_confidence_score_is_none_for_nan_score(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        chunks = (_make_chunk(original_doc_id="doc_0", chunk_index=0, score=float("nan")),)
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=chunks)
        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result()
        pipeline = RAGPipeline(retriever=retriever, generator=generator)
        result = pipeline.query("Any question.")
        assert result.confidence_score is None

    def test_confidence_score_is_none_for_inf_score(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        chunks = (_make_chunk(original_doc_id="doc_0", chunk_index=0, score=float("inf")),)
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=chunks)
        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result()
        pipeline = RAGPipeline(retriever=retriever, generator=generator)
        result = pipeline.query("Any question.")
        assert result.confidence_score is None

    def test_confidence_score_is_none_for_negative_inf_score(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        chunks = (_make_chunk(original_doc_id="doc_0", chunk_index=0, score=float("-inf")),)
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=chunks)
        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result()
        pipeline = RAGPipeline(retriever=retriever, generator=generator)
        result = pipeline.query("Any question.")
        assert result.confidence_score is None


@pytest.mark.unit
class TestRAGPipelineValidation:
    def test_raises_value_error_on_empty_query(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            retriever=mocker.MagicMock(),
            generator=mocker.MagicMock(),
        )
        with pytest.raises(ValueError, match="query"):
            pipeline.query("")

    def test_raises_value_error_on_whitespace_query(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            retriever=mocker.MagicMock(),
            generator=mocker.MagicMock(),
        )
        with pytest.raises(ValueError, match="query"):
            pipeline.query("   ")


@pytest.mark.unit
class TestRAGPipelineQueryRouter:
    """Router wired into pipeline: routes when sources=None, explicit sources wins."""

    def _make_pipeline_with_router(self, mocker, router_sources: list[Source]):
        from src.pipeline.rag_pipeline import RAGPipeline

        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=(_make_chunk(),))
        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result()
        router = mocker.MagicMock()
        router.route.return_value = router_sources
        pipeline = RAGPipeline(retriever=retriever, generator=generator, query_router=router)
        return pipeline, retriever, router

    def test_router_called_when_sources_none(self, mocker) -> None:
        pipeline, _, router = self._make_pipeline_with_router(mocker, ["pokeapi"])
        pipeline.query("What are Charizard's base stats?")
        router.route.assert_called_once_with("What are Charizard's base stats?")

    def test_router_result_forwarded_to_retriever(self, mocker) -> None:
        pipeline, retriever, _ = self._make_pipeline_with_router(mocker, ["pokeapi", "smogon"])
        pipeline.query("What EV spread for Garchomp's stats?")
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["sources"] == ["pokeapi", "smogon"]

    def test_explicit_sources_overrides_router(self, mocker) -> None:
        pipeline, retriever, router = self._make_pipeline_with_router(mocker, ["smogon"])
        pipeline.query("Any query.", sources=["bulbapedia"])
        router.route.assert_not_called()
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["sources"] == ["bulbapedia"]

    def test_no_router_preserves_none_sources(self, mocker) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=(_make_chunk(),))
        generator = mocker.MagicMock()
        generator.generate.return_value = _make_generation_result()
        pipeline = RAGPipeline(retriever=retriever, generator=generator)
        pipeline.query("Any query.")
        _, kwargs = retriever.retrieve.call_args
        assert kwargs["sources"] is None


@pytest.mark.unit
class TestRAGPipelineEmptyDocumentsGuard:
    """Test that pipeline raises RetrievalError on empty documents from retrieval."""

    def test_raises_retrieval_error_when_retriever_returns_empty_documents(self, mocker) -> None:
        """When retriever returns empty documents tuple, pipeline raises RetrievalError."""
        from src.pipeline.rag_pipeline import RAGPipeline

        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=())
        generator = mocker.MagicMock()

        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        with pytest.raises(RetrievalError, match="no documents"):
            pipeline.query("Any question.")

        # Generator must never be called when retrieval documents are empty
        generator.generate.assert_not_called()

    def test_error_message_contains_diagnostic_info(self, mocker) -> None:
        """Error message should indicate why retrieval failed (no documents)."""
        from src.pipeline.rag_pipeline import RAGPipeline

        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(chunks=())
        generator = mocker.MagicMock()

        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        with pytest.raises(RetrievalError) as exc_info:
            pipeline.query("test")

        assert "no documents" in str(exc_info.value).lower()
