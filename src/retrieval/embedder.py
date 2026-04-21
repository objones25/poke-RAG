"""BGE-M3 hybrid embedder wrapping FlagEmbedding.BGEM3FlagModel."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import src.retrieval._compat  # noqa: F401
from src.retrieval.types import EmbeddingOutput

_LOG = logging.getLogger(__name__)


class BGEEmbedder:
    """Dense + sparse embedder backed by BAAI/bge-m3 via FlagEmbedding.

    Construct via BGEEmbedder.from_pretrained() in production.
    Pass a mock model directly in tests for full isolation.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def from_pretrained(cls, *, model_name: str, device: str) -> BGEEmbedder:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]

        use_fp16 = device in ("cuda", "mps")
        warnings.filterwarnings(
            "ignore",
            message=".*fast tokenizer.*`__call__`.*",
            category=UserWarning,
        )
        _LOG.info("Loading BGE-M3 embedder '%s' on %s (fp16=%s)", model_name, device, use_fp16)
        instance = cls(
            BGEM3FlagModel(model_name_or_path=model_name, use_fp16=use_fp16, device=device)
        )
        _LOG.info("BGE-M3 embedder '%s' ready", model_name)
        return instance

    def encode(self, texts: list[str]) -> EmbeddingOutput:
        if not texts:
            return EmbeddingOutput(dense=[], sparse=[])

        _LOG.debug("Encoding %d text(s)", len(texts))
        raw = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense: list[list[float]] = [list(map(float, vec)) for vec in raw["dense_vecs"]]
        sparse: list[dict[int, float]] = [
            {int(k): float(v) for k, v in weights.items()} for weights in raw["lexical_weights"]
        ]
        _LOG.debug("Encoded %d texts → dense_dim=%d", len(dense), len(dense[0]) if dense else 0)
        return EmbeddingOutput(dense=dense, sparse=sparse)
