"""Retrieval-internal types. Not shared with generation layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingOutput:
    """Output of BGEEmbedder.encode(). One entry per input text."""

    dense: list[list[float]]  # shape (n, 1024) — primary semantic vectors
    sparse: list[dict[int, float]]  # token_id → weight per document
