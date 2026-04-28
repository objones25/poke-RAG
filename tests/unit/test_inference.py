"""Unit tests for src/generation/inference.py — Inferencer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch


class _FakeInputs(dict[str, Any]):
    """Dict-like batch that supports .to(device), mirrors BatchFeature."""


def _make_fake_inputs(input_len: int = 3) -> _FakeInputs:
    input_ids = torch.arange(input_len).unsqueeze(0)
    fi = _FakeInputs({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
    fi.to = MagicMock(return_value=fi)  # type: ignore[attr-defined]
    return fi


def _make_inferencer(
    *,
    prompt_len: int = 3,
    decoded: str = "  parsed answer  ",
    device: str = "cpu",
) -> tuple[Any, ...]:
    from src.generation.inference import Inferencer
    from src.generation.models import GenerationConfig

    fake_model = MagicMock()
    fake_model.device = device
    fake_processor = MagicMock()

    fake_inputs = _make_fake_inputs(prompt_len)
    fake_processor.apply_chat_template.return_value = "formatted text"
    fake_processor.return_value = fake_inputs
    fake_model.generate.return_value = torch.arange(prompt_len + 2).unsqueeze(0)
    fake_processor.decode.return_value = decoded

    config = GenerationConfig(model_id="test/model")
    return Inferencer(fake_model, fake_processor, config), fake_model, fake_processor, fake_inputs


@pytest.mark.unit
class TestInferencerInfer:
    def test_apply_chat_template_called_with_prompt(self) -> None:
        from src.generation.prompt_builder import SYSTEM_PROMPT

        inferencer, _, fake_processor, _ = _make_inferencer()
        inferencer.infer("What is Pikachu?")

        fake_processor.apply_chat_template.assert_called_once()
        args, _ = fake_processor.apply_chat_template.call_args
        messages = args[0]
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert messages[1] == {"role": "user", "content": "What is Pikachu?"}

    def test_apply_chat_template_kwargs(self) -> None:
        inferencer, _, fake_processor, _ = _make_inferencer()
        inferencer.infer("prompt")

        _, kwargs = fake_processor.apply_chat_template.call_args
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False

    def test_processor_called_with_text_and_return_tensors(self) -> None:
        inferencer, _, fake_processor, _ = _make_inferencer()
        inferencer.infer("prompt")

        fake_processor.assert_called_once()
        _, kwargs = fake_processor.call_args
        assert kwargs["text"] == "formatted text"
        assert kwargs["return_tensors"] == "pt"

    def test_moves_inputs_to_model_device(self) -> None:
        inferencer, _, _, fake_inputs = _make_inferencer(device="cuda")
        inferencer.infer("prompt")

        fake_inputs.to.assert_called_once_with("cuda")

    def test_calls_model_generate(self) -> None:
        inferencer, fake_model, _, _ = _make_inferencer()
        inferencer.infer("prompt")

        fake_model.generate.assert_called_once()

    def test_passes_generation_config_to_model_generate(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "text"
        fake_processor.return_value = fake_inputs
        fake_model.generate.return_value = torch.arange(5).unsqueeze(0)
        fake_processor.decode.return_value = "answer"

        config = GenerationConfig(
            model_id="test/model",
            temperature=0.3,
            max_new_tokens=256,
            top_p=0.8,
            do_sample=True,
        )
        inferencer = Inferencer(fake_model, fake_processor, config)
        inferencer.infer("prompt")

        _, kwargs = fake_model.generate.call_args
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_new_tokens"] == 256
        assert kwargs["top_p"] == 0.8
        assert kwargs["do_sample"] is True

    def test_skips_prompt_tokens_in_output(self) -> None:
        prompt_len = 3
        inferencer, _, fake_processor, _ = _make_inferencer(prompt_len=prompt_len)
        inferencer.infer("prompt")

        args, _ = fake_processor.decode.call_args
        decoded_tensor = args[0]
        # output_ids has prompt_len + 2 total tokens; slice skips prompt_len
        assert decoded_tensor.shape[-1] == 2

    def test_decode_called_with_skip_special_tokens_true(self) -> None:
        inferencer, _, fake_processor, _ = _make_inferencer()
        inferencer.infer("prompt")

        _, kwargs = fake_processor.decode.call_args
        assert kwargs.get("skip_special_tokens") is True

    def test_returns_decoded_and_stripped_result(self) -> None:
        inferencer, _, _, _ = _make_inferencer(decoded="  parsed answer  ")
        result = inferencer.infer("prompt")

        assert result == "parsed answer"

    def test_returns_string_type(self) -> None:
        inferencer, _, _, _ = _make_inferencer()
        result = inferencer.infer("prompt")

        assert isinstance(result, str)

    def test_raises_on_empty_prompt(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        inferencer = Inferencer(MagicMock(), MagicMock(), GenerationConfig(model_id="test/model"))
        with pytest.raises(ValueError, match="prompt"):
            inferencer.infer("")

    def test_raises_on_whitespace_only_prompt(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        inferencer = Inferencer(MagicMock(), MagicMock(), GenerationConfig(model_id="test/model"))
        with pytest.raises(ValueError, match="prompt"):
            inferencer.infer("   \n\t  ")

    def test_raises_on_non_string_decode_output(self) -> None:
        inferencer, _, fake_processor, _ = _make_inferencer()
        fake_processor.decode.return_value = 123

        with pytest.raises(TypeError, match="str"):
            inferencer.infer("prompt")

    def test_max_new_tokens_override_passed_to_generate(self) -> None:
        inferencer, fake_model, _, _ = _make_inferencer()
        inferencer.infer("prompt", max_new_tokens=50)

        _, kwargs = fake_model.generate.call_args
        assert kwargs["max_new_tokens"] == 50

    def test_raises_when_model_returns_no_new_tokens(self) -> None:
        inferencer, fake_model, _, _ = _make_inferencer(prompt_len=3)
        # Return a tensor with exactly prompt_len tokens — no new tokens generated
        fake_model.generate.return_value = torch.arange(3).unsqueeze(0)

        with pytest.raises(RuntimeError, match="no new tokens"):
            inferencer.infer("prompt")

    def test_max_new_tokens_none_uses_config_default(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "text"
        fake_processor.return_value = fake_inputs
        fake_model.generate.return_value = torch.arange(5).unsqueeze(0)
        fake_processor.decode.return_value = "answer"

        config = GenerationConfig(model_id="test/model", max_new_tokens=256)
        inferencer = Inferencer(fake_model, fake_processor, config)
        inferencer.infer("prompt")

        _, kwargs = fake_model.generate.call_args
        assert kwargs["max_new_tokens"] == 256

    def test_raises_runtime_error_when_decode_returns_empty_string(self) -> None:
        inferencer, _, fake_processor, _ = _make_inferencer(decoded="   ")
        with pytest.raises(RuntimeError, match="empty"):
            inferencer.infer("prompt")

    def test_raises_runtime_error_when_decode_returns_only_whitespace(self) -> None:
        inferencer, _, fake_processor, _ = _make_inferencer(decoded="\n\t  \r\n")
        with pytest.raises(RuntimeError, match="empty|whitespace"):
            inferencer.infer("prompt")


@pytest.mark.unit
class TestInferencerEdgeCases:
    def test_empty_source_chunks_with_zero_chunks(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "text"
        fake_processor.return_value = fake_inputs
        fake_model.generate.return_value = torch.arange(5).unsqueeze(0)
        fake_processor.decode.return_value = "answer"

        config = GenerationConfig(model_id="test/model")
        inferencer = Inferencer(fake_model, fake_processor, config)
        result = inferencer.infer("test")

        assert isinstance(result, str)
        assert result == "answer"

    def test_model_output_shape_zero_raises_error(self) -> None:
        inferencer, fake_model, _, _ = _make_inferencer()
        fake_model.generate.return_value = torch.zeros((0, 5))

        with pytest.raises(RuntimeError, match="no sequences"):
            inferencer.infer("prompt")

    def test_output_shape_batch_zero_raises_error(self) -> None:
        inferencer, fake_model, _, _ = _make_inferencer(prompt_len=3)
        fake_model.generate.return_value = torch.zeros((0, 10))

        with pytest.raises(RuntimeError, match="no sequences"):
            inferencer.infer("prompt")


@pytest.mark.unit
class TestPrepareInputs:
    def test_returns_inputs_and_input_len(self) -> None:
        inferencer, _, fake_processor, fake_inputs = _make_inferencer(prompt_len=5)
        inputs, input_len = inferencer._prepare_inputs("What is Pikachu?")
        assert inputs is fake_inputs
        assert input_len == 5

    def test_apply_chat_template_called(self) -> None:
        from src.generation.prompt_builder import SYSTEM_PROMPT

        inferencer, _, fake_processor, _ = _make_inferencer()
        inferencer._prepare_inputs("test prompt")
        fake_processor.apply_chat_template.assert_called_once()
        args, _ = fake_processor.apply_chat_template.call_args
        messages = args[0]
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert messages[1] == {"role": "user", "content": "test prompt"}

    def test_processor_called_with_pt_tensors(self) -> None:
        inferencer, _, fake_processor, _ = _make_inferencer()
        inferencer._prepare_inputs("test prompt")
        _, kwargs = fake_processor.call_args
        assert kwargs["return_tensors"] == "pt"

    def test_inputs_moved_to_model_device(self) -> None:
        inferencer, _, _, fake_inputs = _make_inferencer(device="cuda")
        inferencer._prepare_inputs("test prompt")
        fake_inputs.to.assert_called_once_with("cuda")

    def test_prepare_inputs_sends_system_and_user_messages(self) -> None:
        from src.generation.prompt_builder import SYSTEM_PROMPT

        inferencer, _, fake_processor, _ = _make_inferencer()
        inferencer.infer("What is Pikachu?")

        args, _ = fake_processor.apply_chat_template.call_args
        messages = args[0]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert messages[1] == {"role": "user", "content": "What is Pikachu?"}

    def test_prepare_inputs_enable_thinking_false_by_default(self) -> None:
        inferencer, _, fake_processor, _ = _make_inferencer()
        inferencer.infer("prompt")

        _, kwargs = fake_processor.apply_chat_template.call_args
        assert kwargs["enable_thinking"] is False

    def test_prepare_inputs_enable_thinking_true_when_thinking_on(self) -> None:
        inferencer, _, fake_processor, _ = _make_inferencer()
        inferencer._prepare_inputs("prompt", thinking=True)

        _, kwargs = fake_processor.apply_chat_template.call_args
        assert kwargs["enable_thinking"] is True

    def test_thinking_enabled_defaults_to_false(self) -> None:
        inferencer, _, _, _ = _make_inferencer()
        assert inferencer._thinking_enabled is False

    def test_thinking_enabled_true_stored(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        inferencer = Inferencer(
            MagicMock(),
            MagicMock(),
            GenerationConfig(model_id="test/model"),
            thinking_enabled=True,
        )
        assert inferencer._thinking_enabled is True


@pytest.mark.unit
class TestInferencerThinking:
    def _make_thinking_inferencer(
        self,
        *,
        raw_decoded: str = "<|channel>thought\nsome reasoning<channel|>The answer",
        parse_response_return: str = "The answer",
    ) -> tuple[Any, Any, Any]:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs
        fake_model.generate.return_value = torch.arange(5).unsqueeze(0)
        fake_processor.decode.return_value = raw_decoded
        fake_processor.parse_response.return_value = parse_response_return

        inferencer = Inferencer(
            fake_model,
            fake_processor,
            GenerationConfig(model_id="test/model"),
            thinking_enabled=True,
        )
        return inferencer, fake_model, fake_processor

    def test_thinking_path_uses_skip_special_tokens_false(self) -> None:
        inferencer, _, fake_processor = self._make_thinking_inferencer()
        inferencer.infer("question")

        _, kwargs = fake_processor.decode.call_args
        assert kwargs.get("skip_special_tokens") is False

    def test_thinking_path_calls_parse_response(self) -> None:
        inferencer, _, fake_processor = self._make_thinking_inferencer()
        inferencer.infer("question")

        fake_processor.parse_response.assert_called_once()
        raw = fake_processor.decode.return_value
        fake_processor.parse_response.assert_called_with(raw)

    def test_thinking_path_returns_parse_response_str_output(self) -> None:
        resp = "Charizard is Fire."
        inferencer, _, _ = self._make_thinking_inferencer(parse_response_return=resp)
        result = inferencer.infer("question")
        assert result == resp

    def test_thinking_path_returns_content_from_parse_response_dict(self) -> None:
        # parse_response returns a dict on current transformers — content key holds the answer
        resp = {"role": "assistant", "thinking": "some reasoning", "content": "Charizard is Fire."}
        inferencer, _, _ = self._make_thinking_inferencer(parse_response_return=resp)
        result = inferencer.infer("question")
        assert result == "Charizard is Fire."

    def test_thinking_override_false_skips_parse_response(self) -> None:
        """Inferencer with thinking_enabled=True but infer(thinking=False) uses normal path."""
        inferencer, _, fake_processor = self._make_thinking_inferencer()
        fake_processor.decode.return_value = "normal answer"
        result = inferencer.infer("question", thinking=False)

        _, kwargs = fake_processor.decode.call_args
        assert kwargs.get("skip_special_tokens") is True
        fake_processor.parse_response.assert_not_called()
        assert result == "normal answer"

    def test_thinking_override_true_on_non_thinking_instance(self) -> None:
        """infer(thinking=True) overrides instance default of False."""
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs
        fake_model.generate.return_value = torch.arange(5).unsqueeze(0)
        fake_processor.decode.return_value = "<|channel>thought\n<channel|>answer"
        fake_processor.parse_response.return_value = "answer"

        inferencer = Inferencer(
            fake_model,
            fake_processor,
            GenerationConfig(model_id="test/model"),
            thinking_enabled=False,
        )
        result = inferencer.infer("question", thinking=True)

        _, kwargs = fake_processor.decode.call_args
        assert kwargs.get("skip_special_tokens") is False
        fake_processor.parse_response.assert_called_once()
        assert result == "answer"


@pytest.mark.unit
class TestThinkingStreamFilter:
    def _make_filter(self) -> Any:
        from src.generation.inference import _ThinkingStreamFilter

        return _ThinkingStreamFilter()

    def test_buffers_all_tokens_before_close_tag(self) -> None:
        f = self._make_filter()
        assert f.feed("<|channel>thought\n") is None
        assert f.feed("some internal reasoning") is None

    def test_emits_suffix_when_close_tag_received(self) -> None:
        f = self._make_filter()
        f.feed("<|channel>thought\n")
        result = f.feed("<channel|>The actual answer")
        assert result == "The actual answer"

    def test_returns_none_when_close_tag_has_no_suffix(self) -> None:
        f = self._make_filter()
        f.feed("<|channel>thought\n")
        result = f.feed("<channel|>")
        assert result is None

    def test_passthrough_after_close_tag_seen(self) -> None:
        f = self._make_filter()
        f.feed("<|channel>thought\n<channel|>")
        assert f.feed("more answer tokens") == "more answer tokens"
        assert f.feed("and more") == "and more"

    def test_handles_straddled_close_tag(self) -> None:
        f = self._make_filter()
        f.feed("<|channel>thought\n")
        f.feed("reasoning ")
        f.feed("<channel")  # first half of tag
        result = f.feed("|>answer here")  # second half
        assert result == "answer here"

    def test_close_tag_inline_with_thought(self) -> None:
        f = self._make_filter()
        result = f.feed("<|channel>thought\nreasoning<channel|>answer")
        assert result == "answer"


@pytest.mark.unit
class TestInferencerStreamInferThinking:
    def test_stream_infer_thinking_false_uses_skip_special_tokens_true(self) -> None:
        from unittest.mock import patch

        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs

        with patch("src.generation.inference.TextIteratorStreamer") as mock_cls:
            mock_cls.return_value = iter(["hello"])
            inferencer = Inferencer(
                fake_model,
                fake_processor,
                GenerationConfig(model_id="test/model"),
                thinking_enabled=False,
            )
            list(inferencer.stream_infer("prompt"))
            _, kwargs = mock_cls.call_args
            assert kwargs.get("skip_special_tokens") is True

    def test_stream_infer_thinking_true_uses_skip_special_tokens_false(self) -> None:
        from unittest.mock import patch

        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs

        with patch("src.generation.inference.TextIteratorStreamer") as mock_cls:
            mock_cls.return_value = iter(["<|channel>thought\n<channel|>answer"])
            inferencer = Inferencer(
                fake_model,
                fake_processor,
                GenerationConfig(model_id="test/model"),
                thinking_enabled=True,
            )
            list(inferencer.stream_infer("prompt"))
            _, kwargs = mock_cls.call_args
            assert kwargs.get("skip_special_tokens") is False

    def test_stream_infer_thinking_filters_thought_block(self) -> None:
        from unittest.mock import patch

        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs

        tokens = ["<|channel>thought\n", "internal reasoning", "<channel|>", "The answer"]

        with patch("src.generation.inference.TextIteratorStreamer") as mock_cls:
            mock_cls.return_value = iter(tokens)
            inferencer = Inferencer(
                fake_model,
                fake_processor,
                GenerationConfig(model_id="test/model"),
                thinking_enabled=True,
            )
            result = list(inferencer.stream_infer("prompt"))

        assert result == ["The answer"]

    def test_stream_infer_thinking_false_no_filter_applied(self) -> None:
        from unittest.mock import patch

        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cpu"
        fake_processor = MagicMock()
        fake_inputs = _make_fake_inputs(3)
        fake_processor.apply_chat_template.return_value = "formatted"
        fake_processor.return_value = fake_inputs

        tokens = ["hello", " world"]

        with patch("src.generation.inference.TextIteratorStreamer") as mock_cls:
            mock_cls.return_value = iter(tokens)
            inferencer = Inferencer(
                fake_model,
                fake_processor,
                GenerationConfig(model_id="test/model"),
                thinking_enabled=False,
            )
            result = list(inferencer.stream_infer("prompt"))

        assert result == ["hello", " world"]
