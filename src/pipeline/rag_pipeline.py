from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import AsyncGenerator
from typing import Any, cast

from src.generation.protocols import GeneratorProtocol, StreamingGeneratorProtocol
from src.pipeline.types import PipelineResult
from src.retrieval.protocols import (
    AsyncRetrieverProtocol,
    KnowledgeRefinerProtocol,
    QueryRouterProtocol,
    RetrieverProtocol,
)
from src.types import RetrievalError, Source
from src.utils.math import sigmoid as _sigmoid

_SENTINEL = object()


class RAGPipeline:
    """Orchestrates retrieval → generation. Never calls the generator if retrieval fails."""

    def __init__(
        self,
        *,
        retriever: RetrieverProtocol,
        generator: GeneratorProtocol,
        query_router: QueryRouterProtocol | None = None,
        knowledge_refiner: KnowledgeRefinerProtocol | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._query_router = query_router
        self._knowledge_refiner = knowledge_refiner

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

        if sources is None and self._query_router is not None:
            sources = self._query_router.route(query)

        retrieval_result = self._retriever.retrieve(
            query, top_k=top_k, sources=sources, entity_name=entity_name
        )
        chunks = retrieval_result.documents

        if not chunks:
            raise RetrievalError("Retrieval returned no documents for query")

        knowledge_gaps: tuple[str, ...] | None = None
        if self._knowledge_refiner is not None:
            refinement = self._knowledge_refiner.refine(query, list(chunks))
            if not refinement.chunks:
                raise RetrievalError("KnowledgeRefiner dropped all chunks for query")
            chunks = refinement.chunks
            knowledge_gaps = refinement.gaps if refinement.gaps else None

        gen_result = self._generator.generate(query, chunks)

        raw_score = max(c.score for c in chunks)
        confidence_score: float | None = _sigmoid(raw_score) if math.isfinite(raw_score) else None

        return PipelineResult(
            answer=gen_result.answer,
            sources_used=gen_result.sources_used,
            num_chunks_used=gen_result.num_chunks_used,
            model_name=gen_result.model_name,
            query=query,
            confidence_score=confidence_score,
            knowledge_gaps=knowledge_gaps,
        )


class AsyncRAGPipeline:
    """Async RAG pipeline: await retrieval, run generation in a thread pool."""

    def __init__(
        self,
        *,
        retriever: AsyncRetrieverProtocol,
        generator: GeneratorProtocol,
        query_router: QueryRouterProtocol | None = None,
        knowledge_refiner: KnowledgeRefinerProtocol | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._query_router = query_router
        self._knowledge_refiner = knowledge_refiner

    async def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        sources: list[Source] | None = None,
        entity_name: str | None = None,
    ) -> PipelineResult:
        """Run an async RAG query.

        Raises:
            ValueError: If query is empty or whitespace-only.
            RetrievalError: Propagated immediately if retrieval fails.
        """
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        if sources is None and self._query_router is not None:
            sources = self._query_router.route(query)

        retrieval_result = await self._retriever.retrieve(
            query, top_k=top_k, sources=sources, entity_name=entity_name
        )
        chunks = retrieval_result.documents

        if not chunks:
            raise RetrievalError("Retrieval returned no documents for query")

        knowledge_gaps: tuple[str, ...] | None = None
        if self._knowledge_refiner is not None:
            refinement = self._knowledge_refiner.refine(query, list(chunks))
            if not refinement.chunks:
                raise RetrievalError("KnowledgeRefiner dropped all chunks for query")
            chunks = refinement.chunks
            knowledge_gaps = refinement.gaps if refinement.gaps else None

        gen_result = await asyncio.to_thread(self._generator.generate, query, chunks)

        raw_score = max(c.score for c in chunks)
        confidence_score: float | None = _sigmoid(raw_score) if math.isfinite(raw_score) else None

        return PipelineResult(
            answer=gen_result.answer,
            sources_used=gen_result.sources_used,
            num_chunks_used=gen_result.num_chunks_used,
            model_name=gen_result.model_name,
            query=query,
            confidence_score=confidence_score,
            knowledge_gaps=knowledge_gaps,
        )

    async def stream_query(
        self,
        query: str,
        *,
        top_k: int = 5,
        sources: list[Source] | None = None,
        entity_name: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens for a RAG query one-at-a-time as the model produces them.

        Raises:
            ValueError: If query is empty or whitespace-only.
            RetrievalError: Propagated immediately if retrieval returns no documents.
        """
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        if sources is None and self._query_router is not None:
            sources = self._query_router.route(query)

        retrieval_result = await self._retriever.retrieve(
            query, top_k=top_k, sources=sources, entity_name=entity_name
        )
        chunks = retrieval_result.documents

        if not chunks:
            raise RetrievalError("Retrieval returned no documents for query")

        if self._knowledge_refiner is not None:
            refinement = self._knowledge_refiner.refine(query, list(chunks))
            if not refinement.chunks:
                raise RetrievalError("KnowledgeRefiner dropped all chunks for query")
            chunks = refinement.chunks

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()

        def _produce() -> None:
            try:
                streaming = cast(StreamingGeneratorProtocol, self._generator)
                for token in streaming.stream_generate(query, chunks):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        thread = loop.run_in_executor(None, _produce)

        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                with contextlib.suppress(Exception):
                    await thread
                raise item
            yield item

        await thread
