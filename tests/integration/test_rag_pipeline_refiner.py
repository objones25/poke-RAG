"""Integration tests: KnowledgeRefiner wired into RAGPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.types import RetrievalError, RetrievalResult
from tests.conftest import make_chunk


def _make_retriever(chunks) -> MagicMock:
    mock = MagicMock()
    mock.retrieve.return_value = RetrievalResult(documents=tuple(chunks), query="query")
    return mock


def _make_generator() -> MagicMock:
    from src.types import GenerationResult

    mock = MagicMock()
    mock.generate.return_value = GenerationResult(
        answer="answer",
        sources_used=("pokeapi",),
        model_name="model",
        num_chunks_used=1,
    )
    return mock


def _make_refiner(chunks, gaps=()) -> MagicMock:
    from src.retrieval.types import RefinementResult

    mock = MagicMock()
    mock.refine.return_value = RefinementResult(chunks=tuple(chunks), gaps=tuple(gaps))
    return mock


@pytest.mark.integration
class TestRAGPipelineWithRefiner:
    def test_refiner_none_skips_refinement(self) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        chunk = make_chunk()
        retriever = _make_retriever([chunk])
        generator = _make_generator()
        pipeline = RAGPipeline(retriever=retriever, generator=generator)

        pipeline.query("test query")

        generator.generate.assert_called_once()

    def test_refiner_called_with_query_and_chunks(self) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        chunk = make_chunk()
        retriever = _make_retriever([chunk])
        generator = _make_generator()
        refiner = _make_refiner([chunk])
        pipeline = RAGPipeline(retriever=retriever, generator=generator, knowledge_refiner=refiner)

        pipeline.query("test query")

        refiner.refine.assert_called_once()
        call_args = refiner.refine.call_args
        assert call_args.args[0] == "test query"

    def test_refined_chunks_passed_to_generator(self) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        original = make_chunk(text="original")
        refined = make_chunk(text="refined text")
        retriever = _make_retriever([original])
        generator = _make_generator()
        refiner = _make_refiner([refined])
        pipeline = RAGPipeline(retriever=retriever, generator=generator, knowledge_refiner=refiner)

        pipeline.query("test query")

        chunks_passed = generator.generate.call_args.args[1]
        assert any(c.text == "refined text" for c in chunks_passed)

    def test_all_chunks_dropped_raises_retrieval_error(self) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        chunk = make_chunk()
        retriever = _make_retriever([chunk])
        generator = _make_generator()
        refiner = _make_refiner([])  # drops all chunks
        pipeline = RAGPipeline(retriever=retriever, generator=generator, knowledge_refiner=refiner)

        with pytest.raises(RetrievalError, match="KnowledgeRefiner"):
            pipeline.query("test query")

        generator.generate.assert_not_called()

    def test_gaps_in_pipeline_result(self) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        chunk = make_chunk()
        retriever = _make_retriever([chunk])
        generator = _make_generator()
        refiner = _make_refiner([chunk], gaps=("gen9", "ou"))
        pipeline = RAGPipeline(retriever=retriever, generator=generator, knowledge_refiner=refiner)

        result = pipeline.query("gen9 OU query")

        assert result.knowledge_gaps == ("gen9", "ou")

    def test_no_gaps_pipeline_result_is_none(self) -> None:
        from src.pipeline.rag_pipeline import RAGPipeline

        chunk = make_chunk()
        retriever = _make_retriever([chunk])
        generator = _make_generator()
        refiner = _make_refiner([chunk], gaps=())
        pipeline = RAGPipeline(retriever=retriever, generator=generator, knowledge_refiner=refiner)

        result = pipeline.query("query")

        assert result.knowledge_gaps is None
