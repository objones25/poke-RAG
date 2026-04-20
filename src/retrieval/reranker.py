"""BGE reranker wrapping FlagEmbedding.FlagReranker."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.types import RetrievedChunk


class BGEReranker:
    """Reranker backed by BAAI/bge-reranker-v2-m3.

    Construct via BGEReranker.from_pretrained() in production.
    Pass a mock model directly in tests for full isolation.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def from_pretrained(cls, *, model_name: str, use_fp16: bool) -> BGEReranker:
        from FlagEmbedding import FlagReranker  # type: ignore[import-untyped]

        return cls(FlagReranker(model_name, use_fp16=use_fp16))

    def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not documents:
            return []

        pairs = [[query, doc.text] for doc in documents]
        scores: list[float] = [float(s) for s in self._model.compute_score(pairs)]

        ranked = sorted(
            (replace(doc, score=score) for doc, score in zip(documents, scores, strict=True)),
            key=lambda c: c.score,
            reverse=True,
        )
        return ranked[:top_k]
