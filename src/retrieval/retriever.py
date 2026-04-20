"""RAG retrieval orchestrator: embed → search → rerank."""
from __future__ import annotations

import logging

from src.retrieval.protocols import EmbedderProtocol, RerankerProtocol, VectorStoreProtocol
from src.types import EmbeddingError, RetrievalError, RetrievalResult, Source

_LOG = logging.getLogger(__name__)

_ALL_SOURCES: tuple[Source, ...] = ("bulbapedia", "pokeapi", "smogon")
_DEFAULT_CANDIDATES_PER_SOURCE = 20


class Retriever:
    def __init__(
        self,
        *,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        reranker: RerankerProtocol,
        candidates_per_source: int = _DEFAULT_CANDIDATES_PER_SOURCE,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._candidates_per_source = candidates_per_source

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        sources: list[Source] | None = None,
    ) -> RetrievalResult:
        active_sources = sources if sources is not None else _ALL_SOURCES
        _LOG.info(
            "Retrieving: query_len=%d chars, sources=%s, top_k=%d",
            len(query),
            active_sources,
            top_k,
        )

        try:
            embedding = self._embedder.encode([query])
        except Exception as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc

        if not embedding.dense or not embedding.sparse:
            raise EmbeddingError("Embedder returned empty result for query")
        query_dense = embedding.dense[0]
        query_sparse = embedding.sparse[0]

        candidates = []
        try:
            for source in active_sources:
                chunks = self._vector_store.search(
                    collection=source,
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    top_k=self._candidates_per_source,
                )
                _LOG.debug("Search '%s' → %d candidates", source, len(chunks))
                candidates.extend(chunks)
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        _LOG.info("Total candidates: %d across %d source(s)", len(candidates), len(active_sources))

        if not candidates:
            raise RetrievalError("No candidates found across all sources.")

        try:
            reranked = self._reranker.rerank(query, candidates, top_k=top_k)
        except Exception as exc:
            raise RetrievalError(f"Reranking failed: {exc}") from exc

        if not reranked:
            raise RetrievalError("No documents found for query.")

        _LOG.info("Retrieval complete: %d document(s) returned", len(reranked))
        return RetrievalResult(documents=tuple(reranked), query=query)
