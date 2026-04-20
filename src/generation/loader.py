from __future__ import annotations

import logging

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.generation.models import GenerationConfig

_LOG = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, config: GenerationConfig | None = None) -> None:
        self._config = config or GenerationConfig()
        self._model_id = self._config.model_id
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    def load(self) -> None:
        if self._model is not None:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _LOG.info("Loading model %s on %s", self._model_id, device)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_id,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        _LOG.info("Model loaded successfully")

    def get_model(self) -> PreTrainedModel:
        if self._model is None:
            raise RuntimeError(
                f"Model '{self._model_id}' not loaded. Call load() first."
            )
        return self._model

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError(
                f"Tokenizer for '{self._model_id}' not loaded. Call load() first."
            )
        return self._tokenizer

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        _LOG.info("Model '%s' unloaded", self._model_id)
