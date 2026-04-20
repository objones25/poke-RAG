"""BGE-M3 hybrid embedder wrapping FlagEmbedding.BGEM3FlagModel."""
from __future__ import annotations

from typing import Any

from src.retrieval.types import EmbeddingOutput


class BGEEmbedder:
    """Dense + sparse embedder backed by BAAI/bge-m3 via FlagEmbedding.

    Construct via BGEEmbedder.from_pretrained() in production.
    Pass a mock model directly in tests for full isolation.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def from_pretrained(cls, *, model_name: str, use_fp16: bool) -> BGEEmbedder:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]

        return cls(BGEM3FlagModel(model_name_or_path=model_name, use_fp16=use_fp16))

    def encode(self, texts: list[str]) -> EmbeddingOutput:
        if not texts:
            return EmbeddingOutput(dense=[], sparse=[])

        raw = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense: list[list[float]] = [list(map(float, vec)) for vec in raw["dense_vecs"]]
        sparse: list[dict[int, float]] = [
            {int(k): float(v) for k, v in weights.items()}
            for weights in raw["lexical_weights"]
        ]
        return EmbeddingOutput(dense=dense, sparse=sparse)
