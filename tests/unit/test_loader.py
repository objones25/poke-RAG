"""Unit tests for src/generation/loader.py — ModelLoader and dtype selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.mark.unit
class TestDtypeForDevice:
    def test_returns_float16_for_mps(self) -> None:
        from src.generation.loader import _dtype_for_device

        assert _dtype_for_device("mps") == torch.float16

    def test_returns_bfloat16_for_cuda(self) -> None:
        from src.generation.loader import _dtype_for_device

        assert _dtype_for_device("cuda") == torch.bfloat16

    def test_returns_float32_for_cpu(self) -> None:
        from src.generation.loader import _dtype_for_device

        assert _dtype_for_device("cpu") == torch.float32


@pytest.mark.unit
class TestModelLoaderLoad:
    def test_calls_auto_model_from_pretrained(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ) as mock_model_load,
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu"
            )
            loader.load()

            mock_model_load.assert_called_once()
            call_args = mock_model_load.call_args
            assert call_args[0][0] == "google/gemma-2-2b-it"

    def test_calls_auto_tokenizer_from_pretrained(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ) as mock_tok_load,
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu"
            )
            loader.load()

            mock_tok_load.assert_called_once()
            call_args = mock_tok_load.call_args
            assert call_args[0][0] == "google/gemma-2-2b-it"

    def test_passes_device_to_model_load(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ) as mock_model_load,
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="mps"
            )
            loader.load()

            _, kwargs = mock_model_load.call_args
            assert kwargs["device_map"] == "mps"

    def test_passes_correct_dtype_for_cuda(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ) as mock_model_load,
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cuda"
            )
            loader.load()

            _, kwargs = mock_model_load.call_args
            assert kwargs["dtype"] == torch.bfloat16

    def test_passes_correct_dtype_for_cpu(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ) as mock_model_load,
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu"
            )
            loader.load()

            _, kwargs = mock_model_load.call_args
            assert kwargs["dtype"] == torch.float32

    def test_sets_tokenizer_pad_token_to_eos_token(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu"
            )
            loader.load()

            assert fake_tokenizer.pad_token == "<eos>"

    def test_load_is_idempotent(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ) as mock_model_load,
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ) as mock_tok_load,
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu"
            )
            loader.load()
            loader.load()

            mock_model_load.assert_called_once()
            mock_tok_load.assert_called_once()


@pytest.mark.unit
class TestModelLoaderGetters:
    def test_get_model_raises_before_load(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        loader = ModelLoader(config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu")
        with pytest.raises(RuntimeError, match="load"):
            loader.get_model()

    def test_get_tokenizer_raises_before_load(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        loader = ModelLoader(config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu")
        with pytest.raises(RuntimeError, match="load"):
            loader.get_tokenizer()

    def test_get_model_returns_loaded_model(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu"
            )
            loader.load()

            assert loader.get_model() is fake_model

    def test_get_tokenizer_returns_loaded_tokenizer(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu"
            )
            loader.load()

            assert loader.get_tokenizer() is fake_tokenizer


@pytest.mark.unit
class TestModelLoaderUnload:
    def test_unload_clears_model_and_tokenizer(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu"
            )
            loader.load()
            loader.unload()

            with pytest.raises(RuntimeError, match="load"):
                loader.get_model()
            with pytest.raises(RuntimeError, match="load"):
                loader.get_tokenizer()

    def test_unload_calls_cuda_empty_cache_on_cuda(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
            patch("src.generation.loader.torch.cuda.is_available", return_value=True),
            patch("src.generation.loader.torch.cuda.empty_cache") as mock_empty_cache,
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cuda"
            )
            loader.load()
            loader.unload()

            mock_empty_cache.assert_called_once()

    def test_unload_does_not_empty_cache_on_cpu(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.eos_token = "<eos>"

        with (
            patch(
                "src.generation.loader.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
            patch("src.generation.loader.torch.cuda.is_available", return_value=False),
            patch("src.generation.loader.torch.cuda.empty_cache") as mock_empty_cache,
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu"
            )
            loader.load()
            loader.unload()

            mock_empty_cache.assert_not_called()

    def test_unload_before_load_does_not_raise(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        loader = ModelLoader(config=GenerationConfig(model_id="google/gemma-2-2b-it"), device="cpu")
        loader.unload()
