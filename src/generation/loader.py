from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, PreTrainedModel

from src.generation.models import GenerationConfig

_LOG = logging.getLogger(__name__)


def _dtype_for_device(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


_HF_FALLBACK_ADAPTER = "objones25/pokesage-lora"


class ModelLoader:
    def __init__(
        self,
        config: GenerationConfig,
        device: str,
        lora_adapter_path: str | None = None,
    ) -> None:
        self._config = config
        self._device = device
        self._model_id = self._config.model_id
        self._lora_adapter_path = lora_adapter_path
        self._model: PreTrainedModel | None = None
        self._processor: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            _LOG.debug("Model '%s' already loaded — skipping", self._model_id)
            return

        dtype = _dtype_for_device(self._device)
        _LOG.info("Loading '%s' on %s (dtype=%s)", self._model_id, self._device, dtype)

        try:
            self._processor = AutoProcessor.from_pretrained(self._model_id)  # type: ignore[no-untyped-call]
        except Exception as exc:
            raise RuntimeError(f"Failed to load processor for '{self._model_id}': {exc}") from exc
        _LOG.debug("Processor for '%s' ready", self._model_id)

        try:
            if self._device == "mps":
                # transformers 5.5 caching_allocator_warmup has no MPS-specific logic and
                # blindly tries to torch.empty() the full model size (~15 GiB), which MPS
                # rejects. The warmup is skipped when device_map is None, so load on CPU
                # then move to MPS — the documented pattern for non-CUDA devices.
                raw_model: PreTrainedModel = AutoModelForImageTextToText.from_pretrained(
                    self._model_id,
                    dtype=dtype,
                    attn_implementation="sdpa",
                ).to(self._device)  # type: ignore[arg-type]
            else:
                raw_model = AutoModelForImageTextToText.from_pretrained(
                    self._model_id,
                    device_map="auto",
                    dtype=dtype,
                    attn_implementation="sdpa",
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to load model '{self._model_id}': {exc}") from exc
        self._model = self._apply_lora_adapter(raw_model)
        _LOG.info("Model '%s' ready", self._model_id)

    def _apply_lora_adapter(self, model: PreTrainedModel) -> PreTrainedModel:
        if self._lora_adapter_path is None:
            return model
        source = (
            self._lora_adapter_path
            if Path(self._lora_adapter_path).exists()
            else _HF_FALLBACK_ADAPTER
        )
        _LOG.info("Loading LoRA adapter from '%s'", source)
        try:
            return PeftModel.from_pretrained(model, source)  # type: ignore[return-value]
        except Exception as exc:
            raise RuntimeError(f"Failed to load LoRA adapter (tried '{source}'): {exc}") from exc

    def get_model(self) -> PreTrainedModel:
        if self._model is None:
            raise RuntimeError(f"Model '{self._model_id}' not loaded. Call load() first.")
        return self._model

    def get_tokenizer(self) -> Any:
        if self._processor is None:
            raise RuntimeError(f"Processor for '{self._model_id}' not loaded. Call load() first.")
        return self._processor

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()
        _LOG.info("Model '%s' unloaded", self._model_id)
