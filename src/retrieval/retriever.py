"""RAG retrieval orchestrator: embed → search → rerank."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

from src.retrieval.protocols import (
    EmbedderProtocol,
    QueryTransformerProtocol,
    RerankerProtocol,
    VectorStoreProtocol,
)
from src.retrieval.types import EmbeddingOutput
from src.types import EmbeddingError, RetrievalError, RetrievalResult, RetrievedChunk, Source

_LOG = logging.getLogger(__name__)

_ALL_SOURCES: tuple[Source, ...] = ("bulbapedia", "pokeapi", "smogon")
_DEFAULT_CANDIDATES_PER_SOURCE = 25
_TARGET_TOTAL_CANDIDATES = 75


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
    ) -> tuple[Source, list[RetrievedChunk]]:
        chunks = self._vector_store.search(
            collection=source,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=top_k,
            entity_name=entity_name,
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
                ): src
                for src in active_sources
            }
            for future in as_completed(futures):
                src = futures[future]
                try:
                    _, chunks = future.result()
                    candidates.extend(chunks)
                except (RuntimeError, ValueError, OSError) as exc:
                    raise RetrievalError(f"Vector search failed for '{src}': {exc}") from exc
                except Exception as exc:
                    raise RetrievalError(
                        f"Vector search failed unexpectedly for '{src}': {exc}"
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

            if raw_reranked and raw_reranked[0].score >= threshold:
                _LOG.info(
                    "Raw pass confidence %.3f >= threshold %.3f; skipping HyDE",
                    raw_reranked[0].score,
                    threshold,
                )
                return RetrievalResult(documents=tuple(raw_reranked), query=query)

            _LOG.info(
                "Raw pass confidence %.3f < threshold %.3f; running HyDE pass",
                raw_reranked[0].score if raw_reranked else 0.0,
                threshold,
            )
            embedding = self._embed_for_search(query, use_transformer=True)
        else:
            embedding = self._embed_for_search(query, use_transformer=use_transformer)

        candidates = self._run_search(
            embedding, active_sources, candidates_per_source, entity_name
        )
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
