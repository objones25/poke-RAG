"""Integration tests for RAGPipeline orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.generation.protocols import GeneratorProtocol
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.types import PipelineResult
from src.retrieval.protocols import RetrieverProtocol
from src.types import GenerationResult, RetrievalError, RetrievalResult


@pytest.mark.integration
class TestRAGPipelineQuery:
    """Test RAGPipeline.query() orchestration and return value."""

    @pytest.fixture
    def mock_retriever(self, make_chunk: callable) -> RetrieverProtocol:
        """Mock retriever returning two chunks from different sources."""
        retriever = MagicMock(spec=RetrieverProtocol)
        retriever.retrieve.return_value = RetrievalResult(
            documents=(
                make_chunk(source="pokeapi", text="Pikachu electric stats"),
                make_chunk(source="bulbapedia", text="Pikachu description"),
            ),
            query="Pikachu",
        )
        return retriever

    @pytest.fixture
    def mock_generator(self) -> GeneratorProtocol:
        """Mock generator returning a generation result based on input chunks."""
        generator = MagicMock(spec=GeneratorProtocol)

        def generate_side_effect(query: str, chunks: tuple) -> GenerationResult:
            """Generate result with sources and count derived from chunks."""
            sources_used = tuple(sorted({c.source for c in chunks}))
            return GenerationResult(
                answer="Pikachu is an Electric-type Pokémon.",
                sources_used=sources_used,
                model_name="google/gemma-4-E4B-it",
                num_chunks_used=len(chunks),
            )

        generator.generate.side_effect = generate_side_effect
        return generator

    def test_returns_pipeline_result(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """query() returns a PipelineResult instance."""
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        result = pipeline.query("Pikachu")
        assert isinstance(result, PipelineResult)

    def test_answer_comes_from_generator(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """result.answer equals generator's answer."""
        expected_answer = "Pikachu is an Electric-type Pokémon."
        mock_generator.generate.return_value = GenerationResult(
            answer=expected_answer,
            sources_used=("pokeapi",),
            model_name="google/gemma-4-E4B-it",
            num_chunks_used=1,
        )
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        result = pipeline.query("Pikachu")
        assert result.answer == expected_answer

    def test_query_preserved_in_result(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """result.query equals the input query."""
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        query_str = "What type is Pikachu?"
        result = pipeline.query(query_str)
        assert result.query == query_str

    def test_num_chunks_used_matches_retrieval_count(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
        make_chunk: callable,
    ) -> None:
        """result.num_chunks_used equals the number of chunks from retrieval."""
        chunks = (
            make_chunk(source="pokeapi"),
            make_chunk(source="pokeapi"),
            make_chunk(source="bulbapedia"),
        )
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=chunks,
            query="test",
        )
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        result = pipeline.query("test")
        assert result.num_chunks_used == 3

    def test_model_name_comes_from_generator(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """result.model_name equals generator's model_name."""
        expected_model = "google/gemma-4-E4B-it"
        mock_generator.generate.return_value = GenerationResult(
            answer="test answer",
            sources_used=("pokeapi",),
            model_name=expected_model,
            num_chunks_used=1,
        )
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        result = pipeline.query("test")
        assert result.model_name == expected_model

    def test_sources_used_deduplicated_and_sorted(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
        make_chunk: callable,
    ) -> None:
        """result.sources_used deduplicates and sorts chunk sources."""
        chunks = (
            make_chunk(source="smogon"),
            make_chunk(source="pokeapi"),
            make_chunk(source="smogon"),
            make_chunk(source="bulbapedia"),
        )
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=chunks,
            query="test",
        )
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        result = pipeline.query("test")
        assert result.sources_used == ("bulbapedia", "pokeapi", "smogon")

    def test_top_k_forwarded_to_retriever(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """top_k parameter is forwarded to retriever.retrieve()."""
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        pipeline.query("test", top_k=3)
        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args[1]
        assert call_kwargs["top_k"] == 3

    def test_sources_forwarded_to_retriever(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """sources parameter is forwarded to retriever.retrieve()."""
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        sources_filter = ["pokeapi", "smogon"]
        pipeline.query("test", sources=sources_filter)
        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args[1]
        assert call_kwargs["sources"] == sources_filter

    def test_retriever_receives_exact_query(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """retriever.retrieve() receives the exact query string."""
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        query_str = "My exact query string"
        pipeline.query(query_str)
        call_args = mock_retriever.retrieve.call_args
        assert call_args[0][0] == query_str

    def test_generator_receives_chunks_from_retrieval(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
        make_chunk: callable,
    ) -> None:
        """generator.generate() receives the exact chunks from retrieval."""
        chunks = (
            make_chunk(source="pokeapi", text="chunk 1"),
            make_chunk(source="bulbapedia", text="chunk 2"),
        )
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=chunks,
            query="test",
        )
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        pipeline.query("test")
        gen_call_args = mock_generator.generate.call_args
        assert gen_call_args[0][1] == chunks

    def test_generator_receives_query(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """generator.generate() receives the query string."""
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        query_str = "test query"
        pipeline.query(query_str)
        gen_call_args = mock_generator.generate.call_args
        assert gen_call_args[0][0] == query_str

    def test_default_top_k_is_five(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """top_k defaults to 5 when not provided."""
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        pipeline.query("test")
        call_kwargs = mock_retriever.retrieve.call_args[1]
        assert call_kwargs["top_k"] == 5

    def test_default_sources_is_none(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """sources defaults to None when not provided."""
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        pipeline.query("test")
        call_kwargs = mock_retriever.retrieve.call_args[1]
        assert call_kwargs["sources"] is None

    def test_empty_chunks_returned_from_retrieval(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """Pipeline handles retrieval returning zero chunks (edge case)."""
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=(),
            query="test",
        )
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        result = pipeline.query("test")
        assert result.num_chunks_used == 0
        assert result.sources_used == ()


@pytest.mark.integration
class TestRAGPipelineErrorPropagation:
    """Test error handling and propagation in RAGPipeline."""

    @pytest.fixture
    def mock_retriever(self) -> RetrieverProtocol:
        """Mock retriever."""
        return MagicMock(spec=RetrieverProtocol)

    @pytest.fixture
    def mock_generator(self) -> GeneratorProtocol:
        """Mock generator."""
        return MagicMock(spec=GeneratorProtocol)

    def test_retrieval_error_propagates_without_calling_generator(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """RetrievalError from retriever propagates; generator is never called."""
        mock_retriever.retrieve.side_effect = RetrievalError("vector index down")
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        with pytest.raises(RetrievalError, match="vector index down"):
            pipeline.query("test")
        mock_generator.generate.assert_not_called()

    def test_empty_query_raises_value_error(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """Empty query string raises ValueError."""
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        with pytest.raises(ValueError, match="query must not be empty or whitespace-only"):
            pipeline.query("")
        mock_retriever.retrieve.assert_not_called()
        mock_generator.generate.assert_not_called()

    def test_whitespace_only_query_raises_value_error(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """Whitespace-only query raises ValueError."""
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        with pytest.raises(ValueError, match="query must not be empty or whitespace-only"):
            pipeline.query("   \t\n   ")
        mock_retriever.retrieve.assert_not_called()
        mock_generator.generate.assert_not_called()

    def test_generator_error_propagates(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
        make_chunk: callable,
    ) -> None:
        """RuntimeError from generator propagates."""
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=(make_chunk(),),
            query="test",
        )
        mock_generator.generate.side_effect = RuntimeError("model inference failed")
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        with pytest.raises(RuntimeError, match="model inference failed"):
            pipeline.query("test")

    def test_query_validation_happens_before_retrieval(
        self,
        mock_retriever: RetrieverProtocol,
        mock_generator: GeneratorProtocol,
    ) -> None:
        """Query validation (empty check) happens before any retrieval call."""
        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        with pytest.raises(ValueError):
            pipeline.query("")
        mock_retriever.retrieve.assert_not_called()


@pytest.mark.integration
class TestRAGPipelineProtocolCompliance:
    """Test that RAGPipeline correctly uses the protocol interfaces."""

    def test_pipeline_uses_retriever_protocol(self, make_chunk: callable) -> None:
        """Pipeline calls retriever with correct protocol signature."""
        retriever = MagicMock(spec=RetrieverProtocol)
        generator = MagicMock(spec=GeneratorProtocol)

        retriever.retrieve.return_value = RetrievalResult(
            documents=(make_chunk(),),
            query="test",
        )
        generator.generate.return_value = GenerationResult(
            answer="test",
            sources_used=("pokeapi",),
            model_name="google/gemma-4-E4B-it",
            num_chunks_used=1,
        )

        pipeline = RAGPipeline(retriever=retriever, generator=generator)
        pipeline.query("test query", top_k=10, sources=["pokeapi"])

        retriever.retrieve.assert_called_once_with(
            "test query",
            top_k=10,
            sources=["pokeapi"],
            entity_name=None,
        )

    def test_pipeline_uses_generator_protocol(self, make_chunk: callable) -> None:
        """Pipeline calls generator with correct protocol signature."""
        retriever = MagicMock(spec=RetrieverProtocol)
        generator = MagicMock(spec=GeneratorProtocol)

        chunks = (
            make_chunk(source="pokeapi", text="stats"),
            make_chunk(source="bulbapedia", text="description"),
        )
        retriever.retrieve.return_value = RetrievalResult(
            documents=chunks,
            query="test",
        )
        generator.generate.return_value = GenerationResult(
            answer="test answer",
            sources_used=("pokeapi", "bulbapedia"),
            model_name="google/gemma-4-E4B-it",
            num_chunks_used=2,
        )

        pipeline = RAGPipeline(retriever=retriever, generator=generator)
        pipeline.query("test query")

        generator.generate.assert_called_once_with("test query", chunks)
