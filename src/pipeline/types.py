from __future__ import annotations

from dataclasses import dataclass

from src.types import Source


@dataclass(frozen=True)
class PipelineResult:
    """Immutable output of a single RAG query. Carries the answer and traceability metadata."""

    answer: str
    sources_used: tuple[Source, ...]
    num_chunks_used: int
    model_name: str
    query: str
