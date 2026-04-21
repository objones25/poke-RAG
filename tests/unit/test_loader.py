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
        fake_processor = MagicMock()

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ) as mock_model_load,
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cpu"
            )
            loader.load()

            mock_model_load.assert_called_once()
            call_args = mock_model_load.call_args
            assert call_args[0][0] == "google/gemma-4-E4B-it"

    def test_calls_auto_processor_from_pretrained(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ) as mock_proc_load,
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cpu"
            )
            loader.load()

            mock_proc_load.assert_called_once()
            call_args = mock_proc_load.call_args
            assert call_args[0][0] == "google/gemma-4-E4B-it"

    def test_mps_loads_without_device_map_then_moves(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()
        fake_model.to.return_value = fake_model

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ) as mock_model_load,
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="mps"
            )
            loader.load()

            _, kwargs = mock_model_load.call_args
            assert "device_map" not in kwargs
            fake_model.to.assert_called_once_with("mps")

    def test_passes_correct_dtype_for_cuda(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ) as mock_model_load,
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cuda"
            )
            loader.load()

            _, kwargs = mock_model_load.call_args
            assert kwargs["dtype"] == torch.bfloat16

    def test_passes_correct_dtype_for_cpu(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ) as mock_model_load,
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cpu"
            )
            loader.load()

            _, kwargs = mock_model_load.call_args
            assert kwargs["dtype"] == torch.float32

    def test_load_is_idempotent(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ) as mock_model_load,
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ) as mock_proc_load,
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cpu"
            )
            loader.load()
            loader.load()

            mock_model_load.assert_called_once()
            mock_proc_load.assert_called_once()


@pytest.mark.unit
class TestModelLoaderGetters:
    def test_get_model_raises_before_load(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        loader = ModelLoader(
            config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cpu"
        )
        with pytest.raises(RuntimeError, match="load"):
            loader.get_model()

    def test_get_tokenizer_raises_before_load(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        loader = ModelLoader(
            config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cpu"
        )
        with pytest.raises(RuntimeError, match="load"):
            loader.get_tokenizer()

    def test_get_model_returns_loaded_model(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cpu"
            )
            loader.load()

            assert loader.get_model() is fake_model

    def test_get_tokenizer_returns_loaded_processor(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cpu"
            )
            loader.load()

            assert loader.get_tokenizer() is fake_processor


@pytest.mark.unit
class TestModelLoaderUnload:
    def test_unload_clears_model_and_tokenizer(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cpu"
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
        fake_processor = MagicMock()

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
            patch("src.generation.loader.torch.cuda.is_available", return_value=True),
            patch("src.generation.loader.torch.cuda.empty_cache") as mock_empty_cache,
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cuda"
            )
            loader.load()
            loader.unload()

            mock_empty_cache.assert_called_once()

    def test_unload_does_not_empty_cache_on_cpu(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
            patch("src.generation.loader.torch.cuda.is_available", return_value=False),
            patch("src.generation.loader.torch.cuda.empty_cache") as mock_empty_cache,
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cpu"
            )
            loader.load()
            loader.unload()

            mock_empty_cache.assert_not_called()

    def test_unload_before_load_does_not_raise(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        loader = ModelLoader(
            config=GenerationConfig(model_id="google/gemma-4-E4B-it"), device="cpu"
        )
        loader.unload()


@pytest.mark.unit
class TestModelLoaderLoraAdapter:
    def test_applies_lora_from_local_path_when_exists(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()
        fake_peft_model = MagicMock()
        mock_path_cls = MagicMock()
        mock_path_cls.return_value.exists.return_value = True

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
            patch("src.generation.loader.Path", mock_path_cls),
            patch(
                "src.generation.loader.PeftModel.from_pretrained",
                return_value=fake_peft_model,
            ) as mock_peft,
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"),
                device="cpu",
                lora_adapter_path="models/pokesage-lora",
            )
            loader.load()

        mock_peft.assert_called_once()
        assert mock_peft.call_args[0][0] is fake_model
        assert mock_peft.call_args[0][1] == "models/pokesage-lora"

    def test_falls_back_to_hf_hub_when_local_missing(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()
        fake_peft_model = MagicMock()
        mock_path_cls = MagicMock()
        mock_path_cls.return_value.exists.return_value = False

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
            patch("src.generation.loader.Path", mock_path_cls),
            patch(
                "src.generation.loader.PeftModel.from_pretrained",
                return_value=fake_peft_model,
            ) as mock_peft,
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"),
                device="cpu",
                lora_adapter_path="models/pokesage-lora",
            )
            loader.load()

        mock_peft.assert_called_once()
        assert mock_peft.call_args[0][1] == "objones25/pokesage-lora"

    def test_raises_when_adapter_unloadable(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()
        mock_path_cls = MagicMock()
        mock_path_cls.return_value.exists.return_value = False

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
            patch("src.generation.loader.Path", mock_path_cls),
            patch(
                "src.generation.loader.PeftModel.from_pretrained",
                side_effect=OSError("adapter not found"),
            ),
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"),
                device="cpu",
                lora_adapter_path="models/pokesage-lora",
            )
            with pytest.raises(RuntimeError, match="LoRA"):
                loader.load()

    def test_skips_lora_when_path_is_none(self) -> None:
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig

        fake_model = MagicMock()
        fake_processor = MagicMock()

        with (
            patch(
                "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "src.generation.loader.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ),
            patch(
                "src.generation.loader.PeftModel.from_pretrained",
            ) as mock_peft,
        ):
            loader = ModelLoader(
                config=GenerationConfig(model_id="google/gemma-4-E4B-it"),
                device="cpu",
                lora_adapter_path=None,
            )
            loader.load()

        mock_peft.assert_not_called()
