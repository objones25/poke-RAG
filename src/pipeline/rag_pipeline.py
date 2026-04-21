from __future__ import annotations

import math
import statistics

from src.generation.protocols import GeneratorProtocol
from src.pipeline.types import PipelineResult
from src.retrieval.protocols import RetrieverProtocol
from src.types import RetrievalError, Source


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class RAGPipeline:
    """Orchestrates retrieval → generation. Never calls the generator if retrieval fails."""

    def __init__(
        self,
        *,
        retriever: RetrieverProtocol,
        generator: GeneratorProtocol,
    ) -> None:
        self._retriever = retriever
        self._generator = generator

    def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        sources: list[Source] | None = None,
        entity_name: str | None = None,
    ) -> PipelineResult:
        """Run a RAG query.

        Raises:
            ValueError: If query is empty or whitespace-only.
            RetrievalError: Propagated immediately if retrieval fails — generator is never called.
        """
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        retrieval_result = self._retriever.retrieve(
            query, top_k=top_k, sources=sources, entity_name=entity_name
        )
        chunks = retrieval_result.documents

        if not chunks:
            raise RetrievalError("Retrieval returned no documents for query")

        gen_result = self._generator.generate(query, chunks)

        confidence_score: float | None = (
            statistics.mean(_sigmoid(c.score) for c in chunks) if chunks else None
        )

        return PipelineResult(
            answer=gen_result.answer,
            sources_used=gen_result.sources_used,
            num_chunks_used=gen_result.num_chunks_used,
            model_name=gen_result.model_name,
            query=query,
            confidence_score=confidence_score,
        )
