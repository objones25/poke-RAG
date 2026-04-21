"""Unit tests for src/generation/inference.py — Inferencer and inference orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch


@pytest.mark.unit
class TestInferencerInfer:
    def test_calls_tokenizer_with_prompt(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_model.device = "cpu"

        encoded = MagicMock()
        encoded.to = MagicMock(return_value=encoded)
        encoded.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        fake_tokenizer.return_value = encoded
        fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        fake_tokenizer.decode.return_value = "answer"

        inferencer = Inferencer(fake_model, fake_tokenizer, GenerationConfig(model_id="test/model"))
        inferencer.infer("What is Pikachu?")

        fake_tokenizer.assert_called_once()
        call_args = fake_tokenizer.call_args
        assert "What is Pikachu?" in call_args[0]

    def test_passes_tokenizer_config_to_tokenizer(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig, TokenizerConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_model.device = "cpu"

        encoded = MagicMock()
        encoded.to = MagicMock(return_value=encoded)
        encoded.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        fake_tokenizer.return_value = encoded
        fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        fake_tokenizer.decode.return_value = "answer"

        tok_config = TokenizerConfig(max_length=512, return_tensors="pt", truncation=True)
        inferencer = Inferencer(
            fake_model,
            fake_tokenizer,
            GenerationConfig(model_id="test/model"),
            tokenizer_config=tok_config,
        )
        inferencer.infer("prompt")

        _, kwargs = fake_tokenizer.call_args
        assert kwargs["max_length"] == 512
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["truncation"] is True

    def test_calls_model_generate_with_correct_args(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_model.device = "cpu"

        encoded = MagicMock()
        encoded.to = MagicMock(return_value=encoded)
        encoded.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        fake_tokenizer.return_value = encoded
        fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        fake_tokenizer.decode.return_value = "answer"

        inferencer = Inferencer(fake_model, fake_tokenizer, GenerationConfig(model_id="test/model"))
        inferencer.infer("prompt")

        fake_model.generate.assert_called_once()

    def test_passes_generation_config_to_model_generate(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_model.device = "cpu"

        encoded = MagicMock()
        encoded.to = MagicMock(return_value=encoded)
        encoded.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        fake_tokenizer.return_value = encoded
        fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        fake_tokenizer.decode.return_value = "answer"

        config = GenerationConfig(
            model_id="test/model",
            temperature=0.3,
            max_new_tokens=256,
            top_p=0.8,
            do_sample=True,
        )
        inferencer = Inferencer(fake_model, fake_tokenizer, config)
        inferencer.infer("prompt")

        _, kwargs = fake_model.generate.call_args
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_new_tokens"] == 256
        assert kwargs["top_p"] == 0.8
        assert kwargs["do_sample"] is True

    def test_skips_prompt_tokens_in_output(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_model.device = "cpu"

        encoded = MagicMock()
        encoded.to = MagicMock(return_value=encoded)
        input_ids = torch.tensor([[10, 11, 12]])  # 3 input tokens
        encoded.__getitem__ = MagicMock(return_value=input_ids)
        fake_tokenizer.return_value = encoded

        output_ids = torch.tensor([[10, 11, 12, 20, 21, 22]])  # 6 total tokens
        fake_model.generate.return_value = output_ids
        fake_tokenizer.decode.return_value = "generated text"

        inferencer = Inferencer(fake_model, fake_tokenizer, GenerationConfig(model_id="test/model"))
        inferencer.infer("prompt")

        call_args = fake_tokenizer.decode.call_args[0]
        decoded_tensor = call_args[0]
        assert decoded_tensor.shape[-1] == 3  # Only new tokens (6 - 3 = 3)

    def test_returns_stripped_string(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_model.device = "cpu"

        encoded = MagicMock()
        encoded.to = MagicMock(return_value=encoded)
        encoded.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        fake_tokenizer.return_value = encoded
        fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        fake_tokenizer.decode.return_value = "  some answer  "

        inferencer = Inferencer(fake_model, fake_tokenizer, GenerationConfig(model_id="test/model"))
        result = inferencer.infer("prompt")

        assert result == "some answer"

    def test_calls_decode_with_skip_special_tokens(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_model.device = "cpu"

        encoded = MagicMock()
        encoded.to = MagicMock(return_value=encoded)
        encoded.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        fake_tokenizer.return_value = encoded
        fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        fake_tokenizer.decode.return_value = "answer"

        inferencer = Inferencer(fake_model, fake_tokenizer, GenerationConfig(model_id="test/model"))
        inferencer.infer("prompt")

        _, kwargs = fake_tokenizer.decode.call_args
        assert kwargs.get("skip_special_tokens") is True

    def test_raises_on_empty_prompt(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()

        inferencer = Inferencer(fake_model, fake_tokenizer, GenerationConfig(model_id="test/model"))
        with pytest.raises(ValueError, match="prompt"):
            inferencer.infer("")

    def test_raises_on_whitespace_only_prompt(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()

        inferencer = Inferencer(fake_model, fake_tokenizer, GenerationConfig(model_id="test/model"))
        with pytest.raises(ValueError, match="prompt"):
            inferencer.infer("   \n\t  ")

    def test_moves_inputs_to_model_device(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_model.device = "cuda"
        fake_tokenizer = MagicMock()

        encoded = MagicMock()
        moved_encoded = MagicMock()
        encoded.to = MagicMock(return_value=moved_encoded)
        moved_encoded.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        fake_tokenizer.return_value = encoded
        fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        fake_tokenizer.decode.return_value = "answer"

        inferencer = Inferencer(fake_model, fake_tokenizer, GenerationConfig(model_id="test/model"))
        inferencer.infer("prompt")

        encoded.to.assert_called_once_with("cuda")

    def test_returns_string_type(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_model.device = "cpu"

        encoded = MagicMock()
        encoded.to = MagicMock(return_value=encoded)
        encoded.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        fake_tokenizer.return_value = encoded
        fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        fake_tokenizer.decode.return_value = "test answer"

        inferencer = Inferencer(fake_model, fake_tokenizer, GenerationConfig(model_id="test/model"))
        result = inferencer.infer("prompt")

        assert isinstance(result, str)

    def test_raises_on_non_string_decode_output(self) -> None:
        from src.generation.inference import Inferencer
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_model.device = "cpu"

        encoded = MagicMock()
        encoded.to = MagicMock(return_value=encoded)
        encoded.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        fake_tokenizer.return_value = encoded
        fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        fake_tokenizer.decode.return_value = 123  # Not a string

        inferencer = Inferencer(fake_model, fake_tokenizer, GenerationConfig(model_id="test/model"))
        with pytest.raises(TypeError, match="str"):
            inferencer.infer("prompt")
