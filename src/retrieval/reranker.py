"""BGE reranker wrapping FlagEmbedding.FlagReranker."""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any

import src.retrieval._compat  # noqa: F401
from src.types import RetrievalError, RetrievedChunk

_LOG = logging.getLogger(__name__)

_RERANKER_MAX_LENGTH = 512


class BGEReranker:
    """Reranker backed by BAAI/bge-reranker-v2-m3.

    Construct via BGEReranker.from_pretrained() in production.
    Pass a mock model directly in tests for full isolation.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def from_pretrained(cls, *, model_name: str, device: str) -> BGEReranker:
        from FlagEmbedding import FlagReranker  # type: ignore[import-untyped]

        use_fp16 = device in ("cuda", "mps")
        _LOG.info("Loading BGE reranker '%s' on %s (fp16=%s)", model_name, device, use_fp16)
        instance = cls(FlagReranker(model_name, use_fp16=use_fp16, device=device))
        _LOG.info("BGE reranker '%s' ready", model_name)
        return instance

    def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not documents:
            return []

        _LOG.debug("Reranking %d candidates, top_k=%d", len(documents), top_k)
        pairs = [[query, doc.text] for doc in documents]
        raw_scores = self._model.compute_score(pairs, max_length=_RERANKER_MAX_LENGTH)
        scores: list[float] = [float(s) if math.isfinite(float(s)) else 0.0 for s in raw_scores]
        if any(not math.isfinite(float(s)) for s in raw_scores):
            _LOG.warning("Reranker returned non-finite score(s); replaced with 0.0")

        if len(scores) != len(documents):
            raise RetrievalError(
                f"Reranker returned {len(scores)} score(s) for {len(documents)} document(s)"
            )
        ranked = sorted(
            (replace(doc, score=score) for doc, score in zip(documents, scores, strict=True)),
            key=lambda c: c.score,
            reverse=True,
        )
        result = ranked[:top_k]
        _LOG.debug("Rerank complete: top_score=%.4f", result[0].score if result else 0.0)
        return result
