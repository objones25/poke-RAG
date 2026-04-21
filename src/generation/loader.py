from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, PreTrainedModel

from src.generation.models import GenerationConfig

_LOG = logging.getLogger(__name__)


def _dtype_for_device(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


class ModelLoader:
    def __init__(self, config: GenerationConfig, device: str) -> None:
        self._config = config
        self._device = device
        self._model_id = self._config.model_id
        self._model: PreTrainedModel | None = None
        self._processor: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            _LOG.debug("Model '%s' already loaded — skipping", self._model_id)
            return

        dtype = _dtype_for_device(self._device)
        _LOG.info("Loading '%s' on %s (dtype=%s)", self._model_id, self._device, dtype)

        self._processor = AutoProcessor.from_pretrained(self._model_id)  # type: ignore[no-untyped-call]
        _LOG.debug("Processor for '%s' ready", self._model_id)

        if self._device == "mps":
            # transformers 5.5 caching_allocator_warmup has no MPS-specific logic and
            # blindly tries to torch.empty() the full model size (~15 GiB), which MPS
            # rejects. The warmup is skipped when device_map is None, so load on CPU
            # then move to MPS — the documented pattern for non-CUDA devices.
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_id,
                dtype=dtype,
                attn_implementation="sdpa",
            ).to(self._device)  # type: ignore[arg-type]
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_id,
                device_map="auto",
                dtype=dtype,
                attn_implementation="sdpa",
            )
        _LOG.info("Model '%s' ready", self._model_id)

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
