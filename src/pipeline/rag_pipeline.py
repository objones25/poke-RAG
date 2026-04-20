from __future__ import annotations

from src.generation.protocols import GeneratorProtocol
from src.pipeline.types import PipelineResult
from src.retrieval.protocols import RetrieverProtocol
from src.types import Source


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
    ) -> PipelineResult:
        """Run a RAG query.

        Raises:
            ValueError: If query is empty or whitespace-only.
            RetrievalError: Propagated immediately if retrieval fails — generator is never called.
        """
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        retrieval_result = self._retriever.retrieve(query, top_k=top_k, sources=sources)
        chunks = retrieval_result.documents

        gen_result = self._generator.generate(query, chunks)

        sources_used: tuple[Source, ...] = tuple(sorted({c.source for c in chunks}))

        return PipelineResult(
            answer=gen_result.answer,
            sources_used=sources_used,
            num_chunks_used=len(chunks),
            model_name=gen_result.model_name,
            query=query,
        )
