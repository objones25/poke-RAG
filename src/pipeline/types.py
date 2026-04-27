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
    # Sigmoid of top chunk's BGE reranker logit — measures retrieval relevance, not answer
    # quality. On-topic queries routinely exceed 0.5; None when the score is non-finite.
    confidence_score: float | None = None
    knowledge_gaps: tuple[str, ...] | None = None
