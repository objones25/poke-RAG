"""Unit tests for Generator.stream_generate() — streaming via Inferencer.stream_infer()."""

from __future__ import annotations

import pytest

from src.generation.exceptions import GenerationError
from src.generation.generator import Generator
from src.generation.models import GenerationConfig
from src.types import RetrievedChunk


@pytest.fixture
def gen_config() -> GenerationConfig:
    return GenerationConfig(
        model_id="test-model",
        temperature=0.7,
        max_new_tokens=64,
        top_p=0.9,
        do_sample=True,
    )


@pytest.fixture
def chunk() -> RetrievedChunk:
    return RetrievedChunk(
        text="Pikachu is an Electric-type Pokémon.",
        source="bulbapedia",
        score=0.9,
        entity_name="Pikachu",
        entity_type="pokemon",
        chunk_index=0,
        original_doc_id="doc-0",
    )


@pytest.fixture
def mock_loader(mocker):
    return mocker.MagicMock()


@pytest.fixture
def mock_inferencer(mocker):
    return mocker.MagicMock()


@pytest.fixture
def mock_prompt_builder(mocker):
    builder = mocker.MagicMock()
    builder.return_value = "built prompt"
    return builder


@pytest.fixture
def generator(mock_loader, mock_prompt_builder, mock_inferencer, gen_config) -> Generator:
    return Generator(
        loader=mock_loader,
        prompt_builder=mock_prompt_builder,
        inferencer=mock_inferencer,
        config=gen_config,
    )


class TestStreamGenerateValidation:
    def test_empty_chunks_raises_value_error(self, generator):
        with pytest.raises(ValueError, match="chunks must not be empty"):
            list(generator.stream_generate("what is pikachu?", ()))

    def test_single_chunk_accepted(self, generator, chunk):
        generator._inferencer.stream_infer.return_value = iter(["Pikachu"])
        result = list(generator.stream_generate("what?", (chunk,)))
        assert result == ["Pikachu"]


class TestStreamGenerateYieldsTokens:
    def test_yields_tokens_in_order(self, generator, chunk):
        tokens = ["Pika", "chu", " is", " Electric"]
        generator._inferencer.stream_infer.return_value = iter(tokens)
        result = list(generator.stream_generate("tell me about Pikachu", (chunk,)))
        assert result == tokens

    def test_empty_token_stream_yields_nothing(self, generator, chunk):
        generator._inferencer.stream_infer.return_value = iter([])
        result = list(generator.stream_generate("query", (chunk,)))
        assert result == []

    def test_yields_all_tokens_from_multiple_chunks(self, generator, chunk):
        chunks = (chunk, chunk)
        tokens = ["A", "B", "C"]
        generator._inferencer.stream_infer.return_value = iter(tokens)
        result = list(generator.stream_generate("query", chunks))
        assert result == tokens


class TestStreamGenerateCallsInferencer:
    def test_calls_prompt_builder_with_query_and_chunks(
        self, generator, chunk, mock_prompt_builder
    ):
        generator._inferencer.stream_infer.return_value = iter([])
        list(generator.stream_generate("my query", (chunk,)))
        mock_prompt_builder.assert_called_once_with("my query", (chunk,))

    def test_calls_stream_infer_with_built_prompt(self, generator, chunk, mock_prompt_builder):
        mock_prompt_builder.return_value = "the built prompt"
        generator._inferencer.stream_infer.return_value = iter([])
        list(generator.stream_generate("my query", (chunk,)))
        generator._inferencer.stream_infer.assert_called_once_with("the built prompt")

    def test_stream_infer_called_once(self, generator, chunk):
        generator._inferencer.stream_infer.return_value = iter(["tok"])
        list(generator.stream_generate("query", (chunk,)))
        generator._inferencer.stream_infer.assert_called_once()


class TestStreamGenerateErrorHandling:
    def test_infer_exception_wrapped_as_generation_error(self, generator, chunk):
        generator._inferencer.stream_infer.return_value = _failing_generator(
            RuntimeError("GPU OOM")
        )
        with pytest.raises(GenerationError, match="GPU OOM"):
            list(generator.stream_generate("query", (chunk,)))

    def test_generation_error_chained_from_original(self, generator, chunk):
        original = RuntimeError("GPU OOM")
        generator._inferencer.stream_infer.return_value = _failing_generator(original)
        with pytest.raises(GenerationError) as exc_info:
            list(generator.stream_generate("query", (chunk,)))
        assert exc_info.value.__cause__ is original


def _failing_generator(exc: BaseException):
    """Helper: generator that raises on first iteration."""
    raise exc
    yield  # makes this a generator function
