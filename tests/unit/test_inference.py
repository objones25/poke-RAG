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
        inferencer, _, fake_processor, _ = _make_inferencer()
        inferencer.infer("What is Pikachu?")

        fake_processor.apply_chat_template.assert_called_once()
        args, _ = fake_processor.apply_chat_template.call_args
        messages = args[0]
        assert messages == [{"role": "user", "content": "What is Pikachu?"}]

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
