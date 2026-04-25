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

    def __init__(self, model: Any, *, colbert_enabled: bool = False) -> None:
        self._model = model
        self._colbert_enabled = colbert_enabled

    @classmethod
    def from_pretrained(
        cls, *, model_name: str, device: str, colbert_enabled: bool = False
    ) -> BGEEmbedder:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]

        use_fp16 = device in ("cuda", "mps")
        warnings.filterwarnings(
            "ignore",
            message=".*fast tokenizer.*`__call__`.*",
            category=UserWarning,
        )
        _LOG.info(
            "Loading BGE-M3 embedder '%s' on %s (fp16=%s, colbert=%s)",
            model_name,
            device,
            use_fp16,
            colbert_enabled,
        )
        instance = cls(
            BGEM3FlagModel(model_name_or_path=model_name, use_fp16=use_fp16, device=device),
            colbert_enabled=colbert_enabled,
        )
        _LOG.info("BGE-M3 embedder '%s' ready", model_name)
        return instance

    def encode(self, texts: list[str]) -> EmbeddingOutput:
        if not texts:
            empty_colbert: list[list[list[float]]] | None = [] if self._colbert_enabled else None
            return EmbeddingOutput(dense=[], sparse=[], colbert=empty_colbert)

        _LOG.debug("Encoding %d text(s) (colbert=%s)", len(texts), self._colbert_enabled)
        raw = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=self._colbert_enabled,
        )

        dense: list[list[float]] = [list(map(float, vec)) for vec in raw["dense_vecs"]]
        sparse: list[dict[int, float]] = []
        for i, weights in enumerate(raw["lexical_weights"]):
            deduped: dict[int, float] = {}
            for k, v in weights.items():
                k_int = int(k)
                new_val = float(v)
                if k_int in deduped:
                    new_val = max(deduped[k_int], new_val)
                    _LOG.debug(
                        "Duplicate sparse token ID %s in embedding index %d; keeping max", k_int, i
                    )
                deduped[k_int] = new_val
            sparse.append(deduped)

        colbert: list[list[list[float]]] | None = None
        if self._colbert_enabled:
            colbert = [
                [list(map(float, token_vec)) for token_vec in doc_vecs]
                for doc_vecs in raw["colbert_vecs"]
            ]

        _LOG.debug("Encoded %d texts → dense_dim=%d", len(dense), len(dense[0]) if dense else 0)
        return EmbeddingOutput(dense=dense, sparse=sparse, colbert=colbert)
