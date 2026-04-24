"""RAG retrieval orchestrator: embed → search → rerank."""

from __future__ import annotations

import asyncio
import builtins
import logging
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import cast

from src.retrieval.protocols import (
    AsyncVectorStoreProtocol,
    EmbedderProtocol,
    QueryTransformerProtocol,
    RerankerProtocol,
    VectorStoreProtocol,
)
from src.retrieval.types import EmbeddingOutput
from src.types import EmbeddingError, RetrievalError, RetrievalResult, RetrievedChunk, Source

_LOG = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


_ALL_SOURCES: tuple[Source, ...] = ("bulbapedia", "pokeapi", "smogon")
_DEFAULT_CANDIDATES_PER_SOURCE = 25
_TARGET_TOTAL_CANDIDATES = 75
_SEARCH_TIMEOUT_SECONDS = 30


class Retriever:
    def __init__(
        self,
        *,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        reranker: RerankerProtocol,
        candidates_per_source: int = _DEFAULT_CANDIDATES_PER_SOURCE,
        query_transformer: QueryTransformerProtocol | None = None,
        hyde_confidence_threshold: float | None = None,
    ) -> None:
        if candidates_per_source <= 0:
            raise ValueError(f"candidates_per_source must be positive, got {candidates_per_source}")
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._candidates_per_source = candidates_per_source
        self._query_transformer = query_transformer
        self._hyde_confidence_threshold = hyde_confidence_threshold

    def _search_one(
        self,
        source: Source,
        query_dense: list[float],
        query_sparse: dict[int, float],
        entity_name: str | None,
        top_k: int,
        query_colbert: list[list[float]] | None = None,
    ) -> tuple[Source, list[RetrievedChunk]]:
        chunks = self._vector_store.search(
            collection=source,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=top_k,
            entity_name=entity_name,
            query_colbert=query_colbert,
        )
        _LOG.debug("Search '%s' → %d candidates", source, len(chunks))
        return source, chunks

    def _embed(self, text: str) -> EmbeddingOutput:
        """Embed a single text string, raising RetrievalError on failure."""
        try:
            return self._embedder.encode([text])
        except EmbeddingError as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc
        except (RuntimeError, ValueError) as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"Embedding failed unexpectedly: {exc}") from exc

    def _embed_for_search(self, query: str, *, use_transformer: bool) -> EmbeddingOutput:
        """Return embedding to use for vector search.

        When use_transformer is True and the transformer supports transform_to_embedding,
        calls that directly (avoiding a separate embedder.encode call). Otherwise falls
        back to transform() + embedder.encode().
        """
        if use_transformer and self._query_transformer is not None:
            if hasattr(self._query_transformer, "transform_to_embedding"):
                embedding = cast(
                    EmbeddingOutput,
                    self._query_transformer.transform_to_embedding(query),
                )
            else:
                embed_text = self._query_transformer.transform(query)
                _LOG.debug(
                    "Query transformer applied: %d-char input → %d-char embed_text",
                    len(query),
                    len(embed_text),
                )
                embedding = self._embed(embed_text)
        else:
            embedding = self._embed(query)

        if len(embedding.dense) != 1 or len(embedding.sparse) != 1:
            raise EmbeddingError(
                f"Expected 1 embedding, got dense={len(embedding.dense)}"
                f" sparse={len(embedding.sparse)}"
            )
        return embedding

    def _run_search(
        self,
        embedding: EmbeddingOutput,
        active_sources: tuple[Source, ...] | list[Source],
        candidates_per_source: int,
        entity_name: str | None,
    ) -> list[RetrievedChunk]:
        query_colbert: list[list[float]] | None = (
            embedding.colbert[0] if embedding.colbert else None
        )
        candidates: list[RetrievedChunk] = []
        with ThreadPoolExecutor(max_workers=len(active_sources)) as executor:
            futures = {
                executor.submit(
                    self._search_one,
                    src,
                    embedding.dense[0],
                    embedding.sparse[0],
                    entity_name,
                    candidates_per_source,
                    query_colbert,
                ): src
                for src in active_sources
            }
            try:
                for future in as_completed(futures, timeout=_SEARCH_TIMEOUT_SECONDS):
                    src = futures[future]
                    try:
                        _, chunks = future.result(timeout=_SEARCH_TIMEOUT_SECONDS)
                        candidates.extend(chunks)
                    except TimeoutError as exc:
                        raise RetrievalError(
                            f"Vector search timed out for '{src}' "
                            f"(timeout={_SEARCH_TIMEOUT_SECONDS}s)"
                        ) from exc
                    except (RuntimeError, ValueError, OSError) as exc:
                        raise RetrievalError(f"Vector search failed for '{src}': {exc}") from exc
                    except Exception as exc:
                        raise RetrievalError(
                            f"Vector search failed unexpectedly for '{src}': {exc}"
                        ) from exc
            except TimeoutError as exc:
                raise RetrievalError(
                    f"Vector search timed out (timeout={_SEARCH_TIMEOUT_SECONDS}s)"
                ) from exc
        return candidates

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        sources: list[Source] | None = None,
        entity_name: str | None = None,
    ) -> RetrievalResult:
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        active_sources = sources if sources is not None else _ALL_SOURCES
        if not active_sources:
            raise RetrievalError("sources must not be empty")
        candidates_per_source = max(
            self._candidates_per_source,
            _TARGET_TOTAL_CANDIDATES // len(active_sources),
        )
        _LOG.info(
            "Retrieving: query_len=%d chars, sources=%s, top_k=%d",
            len(query),
            active_sources,
            top_k,
        )

        use_transformer = self._query_transformer is not None
        threshold = self._hyde_confidence_threshold

        if use_transformer and threshold is not None:
            # Two-pass: raw pass first, HyDE only if confidence is below threshold.
            raw_embedding = self._embed_for_search(query, use_transformer=False)
            raw_candidates = self._run_search(
                raw_embedding, active_sources, candidates_per_source, entity_name
            )
            _LOG.info(
                "Two-pass raw candidates: %d across %d source(s)",
                len(raw_candidates),
                len(active_sources),
            )
            if not raw_candidates:
                raise RetrievalError("No candidates found across all sources.")

            try:
                raw_reranked = self._reranker.rerank(query, raw_candidates, top_k=top_k)
            except (ValueError, RuntimeError) as exc:
                raise RetrievalError(f"Reranking failed: {exc}") from exc
            except Exception as exc:
                raise RetrievalError(f"Reranking failed unexpectedly: {exc}") from exc

            top_confidence = _sigmoid(raw_reranked[0].score) if raw_reranked else 0.0
            if raw_reranked and top_confidence >= threshold:
                _LOG.info(
                    "Raw pass confidence %.3f >= threshold %.3f; skipping HyDE",
                    top_confidence,
                    threshold,
                )
                return RetrievalResult(documents=tuple(raw_reranked), query=query)

            _LOG.info(
                "Raw pass confidence %.3f < threshold %.3f; running HyDE pass",
                top_confidence,
                threshold,
            )
            embedding = self._embed_for_search(query, use_transformer=True)
        else:
            embedding = self._embed_for_search(query, use_transformer=use_transformer)

        candidates = self._run_search(embedding, active_sources, candidates_per_source, entity_name)
        _LOG.info("Total candidates: %d across %d source(s)", len(candidates), len(active_sources))

        if not candidates:
            raise RetrievalError("No candidates found across all sources.")

        try:
            reranked = self._reranker.rerank(query, candidates, top_k=top_k)
        except (ValueError, RuntimeError) as exc:
            raise RetrievalError(f"Reranking failed: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"Reranking failed unexpectedly: {exc}") from exc

        if not reranked:
            raise RetrievalError("No documents found for query.")

        _LOG.info("Retrieval complete: %d document(s) returned", len(reranked))
        return RetrievalResult(documents=tuple(reranked), query=query)


class AsyncRetriever:
    """Async retrieval orchestrator using AsyncQdrantVectorStore and asyncio.TaskGroup.

    Sync operations (embed, rerank) run in asyncio.to_thread to avoid blocking
    the event loop. Parallel vector searches use asyncio.TaskGroup (Python 3.11+).
    """

    def __init__(
        self,
        *,
        embedder: EmbedderProtocol,
        vector_store: AsyncVectorStoreProtocol,
        reranker: RerankerProtocol,
        candidates_per_source: int = _DEFAULT_CANDIDATES_PER_SOURCE,
        query_transformer: QueryTransformerProtocol | None = None,
        hyde_confidence_threshold: float | None = None,
    ) -> None:
        if candidates_per_source <= 0:
            raise ValueError(f"candidates_per_source must be positive, got {candidates_per_source}")
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._candidates_per_source = candidates_per_source
        self._query_transformer = query_transformer
        self._hyde_confidence_threshold = hyde_confidence_threshold

    def _embed_sync(self, text: str) -> EmbeddingOutput:
        """Embed a single text, wrapping errors as RetrievalError. Runs in a thread."""
        try:
            return self._embedder.encode([text])
        except EmbeddingError as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc
        except (RuntimeError, ValueError) as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"Embedding failed unexpectedly: {exc}") from exc

    async def _embed_for_search(self, query: str, *, use_transformer: bool) -> EmbeddingOutput:
        if use_transformer and self._query_transformer is not None:
            if hasattr(self._query_transformer, "transform_to_embedding"):
                embedding = cast(
                    EmbeddingOutput,
                    await asyncio.to_thread(self._query_transformer.transform_to_embedding, query),
                )
            else:
                embed_text: str = await asyncio.to_thread(self._query_transformer.transform, query)
                _LOG.debug(
                    "Query transformer applied: %d-char input → %d-char embed_text",
                    len(query),
                    len(embed_text),
                )
                embedding = await asyncio.to_thread(self._embed_sync, embed_text)
        else:
            embedding = await asyncio.to_thread(self._embed_sync, query)

        if len(embedding.dense) != 1 or len(embedding.sparse) != 1:
            raise EmbeddingError(
                f"Expected 1 embedding, got dense={len(embedding.dense)}"
                f" sparse={len(embedding.sparse)}"
            )
        return embedding

    async def _search_one(
        self,
        source: Source,
        query_dense: list[float],
        query_sparse: dict[int, float],
        entity_name: str | None,
        top_k: int,
        query_colbert: list[list[float]] | None = None,
    ) -> tuple[Source, list[RetrievedChunk]]:
        try:
            chunks = await asyncio.wait_for(
                self._vector_store.search(
                    collection=source,
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    top_k=top_k,
                    entity_name=entity_name,
                    query_colbert=query_colbert,
                ),
                timeout=_SEARCH_TIMEOUT_SECONDS,
            )
        except builtins.TimeoutError as exc:
            raise RetrievalError(
                f"Vector search timed out for '{source}' (timeout={_SEARCH_TIMEOUT_SECONDS}s)"
            ) from exc
        except (RuntimeError, ValueError, OSError) as exc:
            raise RetrievalError(f"Vector search failed for '{source}': {exc}") from exc
        except Exception as exc:
            raise RetrievalError(
                f"Vector search failed unexpectedly for '{source}': {exc}"
            ) from exc
        _LOG.debug("Search '%s' → %d candidates", source, len(chunks))
        return source, chunks

    async def _run_search(
        self,
        embedding: EmbeddingOutput,
        active_sources: tuple[Source, ...] | list[Source],
        candidates_per_source: int,
        entity_name: str | None,
    ) -> list[RetrievedChunk]:
        query_colbert: list[list[float]] | None = (
            embedding.colbert[0] if embedding.colbert else None
        )
        tasks: list[asyncio.Task[tuple[Source, list[RetrievedChunk]]]] = []
        try:
            async with asyncio.TaskGroup() as tg:
                for src in active_sources:
                    tasks.append(
                        tg.create_task(
                            self._search_one(
                                src,
                                embedding.dense[0],
                                embedding.sparse[0],
                                entity_name,
                                candidates_per_source,
                                query_colbert,
                            )
                        )
                    )
        except* RetrievalError as eg:
            raise eg.exceptions[0] from None

        candidates: list[RetrievedChunk] = []
        for task in tasks:
            _, chunks = task.result()
            candidates.extend(chunks)
        return candidates

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        sources: list[Source] | None = None,
        entity_name: str | None = None,
    ) -> RetrievalResult:
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        active_sources: tuple[Source, ...] | list[Source] = (
            sources if sources is not None else _ALL_SOURCES
        )
        if not active_sources:
            raise RetrievalError("sources must not be empty")
        candidates_per_source = max(
            self._candidates_per_source,
            _TARGET_TOTAL_CANDIDATES // len(active_sources),
        )
        _LOG.info(
            "Retrieving: query_len=%d chars, sources=%s, top_k=%d",
            len(query),
            active_sources,
            top_k,
        )

        use_transformer = self._query_transformer is not None
        threshold = self._hyde_confidence_threshold

        if use_transformer and threshold is not None:
            raw_embedding = await self._embed_for_search(query, use_transformer=False)
            raw_candidates = await self._run_search(
                raw_embedding, active_sources, candidates_per_source, entity_name
            )
            _LOG.info(
                "Two-pass raw candidates: %d across %d source(s)",
                len(raw_candidates),
                len(active_sources),
            )
            if not raw_candidates:
                raise RetrievalError("No candidates found across all sources.")

            try:
                raw_reranked = await asyncio.to_thread(
                    self._reranker.rerank, query, raw_candidates, top_k
                )
            except (ValueError, RuntimeError) as exc:
                raise RetrievalError(f"Reranking failed: {exc}") from exc
            except Exception as exc:
                raise RetrievalError(f"Reranking failed unexpectedly: {exc}") from exc

            top_confidence = _sigmoid(raw_reranked[0].score) if raw_reranked else 0.0
            if raw_reranked and top_confidence >= threshold:
                _LOG.info(
                    "Raw pass confidence %.3f >= threshold %.3f; skipping HyDE",
                    top_confidence,
                    threshold,
                )
                return RetrievalResult(documents=tuple(raw_reranked), query=query)

            _LOG.info(
                "Raw pass confidence %.3f < threshold %.3f; running HyDE pass",
                top_confidence,
                threshold,
            )
            embedding = await self._embed_for_search(query, use_transformer=True)
        else:
            embedding = await self._embed_for_search(query, use_transformer=use_transformer)

        candidates = await self._run_search(
            embedding, active_sources, candidates_per_source, entity_name
        )
        _LOG.info("Total candidates: %d across %d source(s)", len(candidates), len(active_sources))

        if not candidates:
            raise RetrievalError("No candidates found across all sources.")

        try:
            reranked = await asyncio.to_thread(self._reranker.rerank, query, candidates, top_k)
        except (ValueError, RuntimeError) as exc:
            raise RetrievalError(f"Reranking failed: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"Reranking failed unexpectedly: {exc}") from exc

        if not reranked:
            raise RetrievalError("No documents found for query.")

        _LOG.info("Retrieval complete: %d document(s) returned", len(reranked))
        return RetrievalResult(documents=tuple(reranked), query=query)
