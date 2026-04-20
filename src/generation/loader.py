from __future__ import annotations

import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

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
        self._tokenizer: PreTrainedTokenizerBase | None = None

    def load(self) -> None:
        if self._model is not None:
            _LOG.debug("Model '%s' already loaded — skipping", self._model_id)
            return

        dtype = _dtype_for_device(self._device)
        _LOG.info("Loading '%s' on %s (dtype=%s)", self._model_id, self._device, dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)  # type: ignore[no-untyped-call]
        self._tokenizer.pad_token = self._tokenizer.eos_token
        _LOG.debug("Tokenizer for '%s' ready", self._model_id)

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            device_map=self._device,
            dtype=dtype,
        )
        _LOG.info("Model '%s' ready", self._model_id)

    def get_model(self) -> PreTrainedModel:
        if self._model is None:
            raise RuntimeError(f"Model '{self._model_id}' not loaded. Call load() first.")
        return self._model

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError(f"Tokenizer for '{self._model_id}' not loaded. Call load() first.")
        return self._tokenizer

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _LOG.info("Model '%s' unloaded", self._model_id)
