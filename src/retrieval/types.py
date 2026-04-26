"""Retrieval-internal types. Not shared with generation layer."""

from __future__ import annotations

from dataclasses import dataclass

from src.types import RetrievedChunk


@dataclass(frozen=True)
class EmbeddingOutput:
    """Output of BGEEmbedder.encode(). One entry per input text."""

    dense: list[list[float]]  # shape (n, 1024) — primary semantic vectors
    sparse: list[dict[int, float]]  # token_id → weight per document
    colbert: list[list[list[float]]] | None = None  # (n, seq_len, 1024) — token-level vectors


@dataclass(frozen=True)
class RefinementResult:
    """Output of KnowledgeRefiner.refine(). Immutable."""

    chunks: tuple[RetrievedChunk, ...]
    gaps: tuple[str, ...]
    dropped_chunks: tuple[RetrievedChunk, ...] = ()
