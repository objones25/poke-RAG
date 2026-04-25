"""Unit tests for Inferencer.stream_infer() — real token streaming via TextIteratorStreamer."""

from __future__ import annotations

import threading

import pytest

from src.generation.inference import Inferencer
from src.generation.models import GenerationConfig


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
def mock_model(mocker):
    model = mocker.MagicMock()
    model.device = "cpu"
    return model


@pytest.fixture
def mock_processor(mocker):
    processor = mocker.MagicMock()
    processor.apply_chat_template.return_value = "formatted prompt"
    mock_inputs = {"input_ids": mocker.MagicMock()}
    mock_be = mocker.MagicMock()
    mock_be.to.return_value = mock_inputs
    processor.return_value = mock_be
    return processor


@pytest.fixture
def inferencer(mock_model, mock_processor, gen_config) -> Inferencer:
    return Inferencer(model=mock_model, processor=mock_processor, config=gen_config)


def _patch_streamer(mocker, tokens: list[str]):
    """Patch TextIteratorStreamer to return a pre-filled iterator of tokens."""
    mock_streamer = iter(tokens)
    mocker.patch("src.generation.inference.TextIteratorStreamer", return_value=mock_streamer)
    return mock_streamer


class TestStreamInferValidation:
    def test_empty_prompt_raises_value_error(self, inferencer):
        with pytest.raises(ValueError, match="prompt must not be empty"):
            list(inferencer.stream_infer(""))

    def test_whitespace_only_prompt_raises_value_error(self, inferencer):
        with pytest.raises(ValueError, match="prompt must not be empty"):
            list(inferencer.stream_infer("   "))

    def test_newline_only_prompt_raises_value_error(self, inferencer):
        with pytest.raises(ValueError, match="prompt must not be empty"):
            list(inferencer.stream_infer("\n\t"))


class TestStreamInferYieldsTokens:
    def test_yields_tokens_in_order(self, inferencer, mocker):
        tokens = ["Pika", "chu", " is", " Electric", "-", "type", "."]
        _patch_streamer(mocker, tokens)
        result = list(inferencer.stream_infer("tell me about Pikachu"))
        assert result == tokens

    def test_skips_empty_string_tokens(self, inferencer, mocker):
        _patch_streamer(mocker, ["hello", "", "world", ""])
        result = list(inferencer.stream_infer("test"))
        assert result == ["hello", "world"]

    def test_preserves_whitespace_tokens(self, inferencer, mocker):
        _patch_streamer(mocker, ["hello", " ", "world"])
        result = list(inferencer.stream_infer("test"))
        assert result == ["hello", " ", "world"]

    def test_empty_streamer_yields_nothing(self, inferencer, mocker):
        _patch_streamer(mocker, [])
        result = list(inferencer.stream_infer("test"))
        assert result == []

    def test_single_token_streamer(self, inferencer, mocker):
        _patch_streamer(mocker, ["hello"])
        result = list(inferencer.stream_infer("test"))
        assert result == ["hello"]


class TestStreamInferThreading:
    def test_model_generate_is_called(self, inferencer, mocker):
        _patch_streamer(mocker, ["ok"])
        list(inferencer.stream_infer("test prompt"))
        inferencer._model.generate.assert_called_once()

    def test_model_generate_called_with_streamer(self, inferencer, mocker):
        _patch_streamer(mocker, ["ok"])
        list(inferencer.stream_infer("test prompt"))
        _, kwargs = inferencer._model.generate.call_args
        assert "streamer" in kwargs

    def test_model_generate_called_with_max_new_tokens(self, inferencer, mocker, gen_config):
        _patch_streamer(mocker, ["ok"])
        list(inferencer.stream_infer("test prompt"))
        _, kwargs = inferencer._model.generate.call_args
        assert kwargs["max_new_tokens"] == gen_config.max_new_tokens

    def test_max_new_tokens_override_is_used(self, inferencer, mocker):
        _patch_streamer(mocker, ["ok"])
        list(inferencer.stream_infer("test prompt", max_new_tokens=42))
        _, kwargs = inferencer._model.generate.call_args
        assert kwargs["max_new_tokens"] == 42

    def test_temperature_passed_to_generate(self, inferencer, mocker, gen_config):
        _patch_streamer(mocker, ["ok"])
        list(inferencer.stream_infer("test"))
        _, kwargs = inferencer._model.generate.call_args
        assert kwargs["temperature"] == gen_config.temperature

    def test_thread_completes_before_return(self, inferencer, mocker):
        """Verify thread.join() is called — generator waits for thread to finish."""
        threads_started: list[threading.Thread] = []
        original_thread_init = threading.Thread.__init__

        def track_thread(self_t, *args, **kwargs):
            original_thread_init(self_t, *args, **kwargs)
            threads_started.append(self_t)

        mocker.patch.object(threading.Thread, "__init__", track_thread)
        _patch_streamer(mocker, ["hello"])
        list(inferencer.stream_infer("test"))
        assert len(threads_started) == 1
        assert not threads_started[0].is_alive()


class TestStreamInferErrorHandling:
    def test_generate_exception_raises_runtime_error(self, inferencer, mocker):
        """Exception in the generation thread is propagated as RuntimeError."""
        _patch_streamer(mocker, [])
        inferencer._model.generate.side_effect = RuntimeError("GPU OOM")
        with pytest.raises(RuntimeError, match="GPU OOM"):
            list(inferencer.stream_infer("test"))

    def test_processor_apply_chat_template_called(self, inferencer, mocker):
        _patch_streamer(mocker, [])
        list(inferencer.stream_infer("hello world"))
        inferencer._processor.apply_chat_template.assert_called_once()

    def test_processor_called_with_chat_formatted_text(self, inferencer, mocker):
        _patch_streamer(mocker, [])
        list(inferencer.stream_infer("hello world"))
        inferencer._processor.assert_called_once()
        call_kwargs = inferencer._processor.call_args.kwargs
        assert "text" in call_kwargs or inferencer._processor.call_args.args
