"""Qdrant-backed vector store with hybrid dense+sparse search."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    Prefetch,
    QueryResponse,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.retrieval.types import EmbeddingOutput
from src.types import RetrievedChunk, Source, VectorIndexError

_LOG = logging.getLogger(__name__)

_SOURCES: tuple[Source, ...] = ("bulbapedia", "pokeapi", "smogon")
_DENSE_DIM = 1024
_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"
_COLBERT_VECTOR_NAME = "colbert"
_UPSERT_BATCH_SIZE = 100
_COLBERT_UPSERT_BATCH_SIZE = 2  # ColBERT token matrices are large; keep HTTP payloads under 32 MB
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code in _TRANSIENT_STATUS_CODES
    return True  # timeouts, connection resets, etc.


class QdrantVectorStore:
    """One Qdrant collection per source, hybrid dense+sparse vectors."""

    def __init__(self, client: Any, *, colbert_enabled: bool = False) -> None:
        self._client = client
        self._colbert_enabled = colbert_enabled

    def drop_collections(self) -> None:
        """Delete all source collections. Required before schema change (e.g. enabling ColBERT)."""
        for source in _SOURCES:
            try:
                self._client.delete_collection(collection_name=source)
                _LOG.info("Dropped collection '%s'", source)
            except Exception as exc:
                _LOG.warning("Could not drop collection '%s': %s", source, exc)

    def ensure_collections(self) -> None:
        _LOG.info("Ensuring %d Qdrant collections: %s", len(_SOURCES), _SOURCES)
        vectors_config: dict[str, VectorParams] = {
            _DENSE_VECTOR_NAME: VectorParams(size=_DENSE_DIM, distance=Distance.COSINE),
        }
        if self._colbert_enabled:
            vectors_config[_COLBERT_VECTOR_NAME] = VectorParams(
                size=_DENSE_DIM,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
            )
        for source in _SOURCES:
            # Only suppress 409 Conflict (already exists); propagate all other HTTP errors
            try:
                self._client.create_collection(
                    collection_name=source,
                    vectors_config=vectors_config,
                    sparse_vectors_config={
                        _SPARSE_VECTOR_NAME: SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        ),
                    },
                )
            except UnexpectedResponse as exc:
                if exc.status_code != 409:
                    raise
            _LOG.debug("Collection '%s' ready", source)

    def upsert(
        self,
        collection: Source,
        documents: list[RetrievedChunk],
        embeddings: EmbeddingOutput,
    ) -> None:
        if len(documents) != len(embeddings.dense) or len(documents) != len(embeddings.sparse):
            raise ValueError(
                f"documents/embeddings length mismatch: "
                f"documents={len(documents)}, dense={len(embeddings.dense)}, "
                f"sparse={len(embeddings.sparse)}"
            )
        if self._colbert_enabled and (
            embeddings.colbert is None or len(embeddings.colbert) != len(documents)
        ):
            colbert_len = len(embeddings.colbert) if embeddings.colbert else None
            raise ValueError(
                f"ColBERT enabled but colbert embeddings missing or length mismatch: "
                f"documents={len(documents)}, colbert={colbert_len}"
            )
        _LOG.info("Upserting %d point(s) into '%s'", len(documents), collection)
        points = []
        for i, doc in enumerate(documents):
            vec: dict[str, Any] = {
                _DENSE_VECTOR_NAME: embeddings.dense[i],
                _SPARSE_VECTOR_NAME: SparseVector(
                    indices=list(embeddings.sparse[i].keys()),
                    values=list(embeddings.sparse[i].values()),
                ),
            }
            if self._colbert_enabled and embeddings.colbert is not None:
                vec[_COLBERT_VECTOR_NAME] = embeddings.colbert[i]
            points.append(
                PointStruct(
                    id=str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.original_doc_id}:{doc.chunk_index}")
                    ),
                    vector=vec,
                    payload={
                        "text": doc.text,
                        "source": doc.source,
                        "entity_name": (
                            doc.entity_name.lower().strip() if doc.entity_name is not None else None
                        ),
                        "entity_type": doc.entity_type,
                        "chunk_index": doc.chunk_index,
                        "original_doc_id": doc.original_doc_id,
                    },
                )
            )
        batch_size = _COLBERT_UPSERT_BATCH_SIZE if self._colbert_enabled else _UPSERT_BATCH_SIZE
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            for attempt in range(_MAX_RETRIES):
                try:
                    self._client.upsert(collection_name=collection, points=batch)
                    break
                except Exception as exc:
                    if attempt < _MAX_RETRIES - 1 and _is_transient(exc):
                        delay = _RETRY_BASE_DELAY * (2**attempt)
                        _LOG.warning(
                            "Upsert to '%s' failed (attempt %d/%d): %s — retrying in %.0fs",
                            collection,
                            attempt + 1,
                            _MAX_RETRIES,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        raise VectorIndexError(
                            f"Upsert to '{collection}' failed: {exc}"
                        ) from exc
            _LOG.debug(
                "Upserted points %d–%d into '%s'",
                i,
                min(i + batch_size, len(points)),
                collection,
            )
        _LOG.debug("Upsert to '%s' complete", collection)

    def _query(
        self,
        collection: Source,
        query_dense: list[float],
        query_sparse: dict[int, float],
        top_k: int,
        query_filter: Filter | None,
        query_colbert: list[list[float]] | None = None,
    ) -> list[RetrievedChunk]:
        sparse_query = SparseVector(
            indices=list(query_sparse.keys()),
            values=list(query_sparse.values()),
        )
        prefetch = [
            Prefetch(query=query_dense, using=_DENSE_VECTOR_NAME, limit=top_k * 2),
            Prefetch(query=sparse_query, using=_SPARSE_VECTOR_NAME, limit=top_k * 2),
        ]
        if self._colbert_enabled and query_colbert is not None:
            prefetch.append(
                Prefetch(query=query_colbert, using=_COLBERT_VECTOR_NAME, limit=top_k * 2)
            )

        try:
            response = self._client.query_points(
                collection_name=collection,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                query_filter=query_filter,
            )
        except Exception as exc:
            raise VectorIndexError(f"Query to collection '{collection}' failed: {exc}") from exc

        chunks: list[RetrievedChunk] = []
        skipped_count = 0
        for p in response.points:
            try:
                chunks.append(
                    RetrievedChunk(
                        text=p.payload["text"],
                        score=float(p.score),
                        source=p.payload["source"],
                        entity_name=p.payload.get("entity_name"),
                        entity_type=p.payload.get("entity_type"),
                        chunk_index=int(p.payload["chunk_index"]),
                        original_doc_id=p.payload["original_doc_id"],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                _LOG.warning("Malformed payload for point %s: %s", p.id, exc)
                skipped_count += 1

        if not chunks and response.points:
            raise VectorIndexError(
                f"Failed to parse any valid results from {len(response.points)} point(s)"
            )

        if skipped_count > 0:
            _LOG.warning(
                "Skipped %d malformed points out of %d total", skipped_count, len(response.points)
            )

        return chunks

    def search(
        self,
        collection: Source,
        query_dense: list[float],
        query_sparse: dict[int, float],
        top_k: int,
        entity_name: str | None = None,
        query_colbert: list[list[float]] | None = None,
    ) -> list[RetrievedChunk]:
        _LOG.debug("Searching '%s': top_k=%d, entity_name=%s", collection, top_k, entity_name)

        normalized_name = entity_name.lower().strip() if entity_name is not None else None
        query_filter = (
            Filter(
                must=[FieldCondition(key="entity_name", match=MatchValue(value=normalized_name))]
            )
            if normalized_name is not None
            else None
        )

        chunks = self._query(
            collection, query_dense, query_sparse, top_k, query_filter, query_colbert
        )

        if not chunks and entity_name is not None:
            _LOG.warning(
                "Entity filter for '%s' in '%s' returned 0 results; retrying without filter",
                entity_name,
                collection,
            )
            chunks = self._query(collection, query_dense, query_sparse, top_k, None, query_colbert)

        _LOG.debug("Search '%s' → %d result(s)", collection, len(chunks))
        return chunks


class AsyncQdrantVectorStore:
    """Async Qdrant vector store with hybrid dense+sparse vectors."""

    def __init__(self, client: AsyncQdrantClient | Any, *, colbert_enabled: bool = False) -> None:
        self._client = client
        self._colbert_enabled = colbert_enabled

    async def drop_collections(self) -> None:
        """Delete all source collections. Required before schema change (e.g. enabling ColBERT)."""
        for source in _SOURCES:
            try:
                await self._client.delete_collection(collection_name=source)
                _LOG.info("Dropped collection '%s'", source)
            except Exception as exc:
                _LOG.warning("Could not drop collection '%s': %s", source, exc)

    async def ensure_collections(self) -> None:
        _LOG.info("Ensuring %d Qdrant collections: %s", len(_SOURCES), _SOURCES)
        vectors_config: dict[str, VectorParams] = {
            _DENSE_VECTOR_NAME: VectorParams(size=_DENSE_DIM, distance=Distance.COSINE),
        }
        if self._colbert_enabled:
            vectors_config[_COLBERT_VECTOR_NAME] = VectorParams(
                size=_DENSE_DIM,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
            )
        for source in _SOURCES:
            try:
                exists = await self._client.collection_exists(collection_name=source)
            except Exception as exc:
                raise exc
            if exists:
                _LOG.debug("Collection '%s' already exists", source)
                continue
            try:
                await self._client.create_collection(
                    collection_name=source,
                    vectors_config=vectors_config,
                    sparse_vectors_config={
                        _SPARSE_VECTOR_NAME: SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        ),
                    },
                )
            except Exception as exc:
                raise exc
            _LOG.debug("Collection '%s' created", source)

    async def upsert(
        self,
        collection: Source,
        documents: list[RetrievedChunk],
        embeddings: EmbeddingOutput,
    ) -> None:
        if len(documents) != len(embeddings.dense) or len(documents) != len(embeddings.sparse):
            raise ValueError(
                f"documents/embeddings length mismatch: "
                f"documents={len(documents)}, dense={len(embeddings.dense)}, "
                f"sparse={len(embeddings.sparse)}"
            )
        if self._colbert_enabled and (
            embeddings.colbert is None or len(embeddings.colbert) != len(documents)
        ):
            colbert_len = len(embeddings.colbert) if embeddings.colbert else None
            raise ValueError(
                f"ColBERT enabled but colbert embeddings missing or length mismatch: "
                f"documents={len(documents)}, colbert={colbert_len}"
            )
        _LOG.info("Upserting %d point(s) into '%s'", len(documents), collection)
        points = []
        for i, doc in enumerate(documents):
            vec: dict[str, Any] = {
                _DENSE_VECTOR_NAME: embeddings.dense[i],
                _SPARSE_VECTOR_NAME: SparseVector(
                    indices=list(embeddings.sparse[i].keys()),
                    values=list(embeddings.sparse[i].values()),
                ),
            }
            if self._colbert_enabled and embeddings.colbert is not None:
                vec[_COLBERT_VECTOR_NAME] = embeddings.colbert[i]
            points.append(
                PointStruct(
                    id=str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.original_doc_id}:{doc.chunk_index}")
                    ),
                    vector=vec,
                    payload={
                        "text": doc.text,
                        "source": doc.source,
                        "entity_name": (
                            doc.entity_name.lower().strip() if doc.entity_name is not None else None
                        ),
                        "entity_type": doc.entity_type,
                        "chunk_index": doc.chunk_index,
                        "original_doc_id": doc.original_doc_id,
                    },
                )
            )
        batch_size = _COLBERT_UPSERT_BATCH_SIZE if self._colbert_enabled else _UPSERT_BATCH_SIZE
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            for attempt in range(_MAX_RETRIES):
                try:
                    await self._client.upsert(collection_name=collection, points=batch)
                    break
                except Exception as exc:
                    if attempt < _MAX_RETRIES - 1 and _is_transient(exc):
                        delay = _RETRY_BASE_DELAY * (2**attempt)
                        _LOG.warning(
                            "Upsert to '%s' failed (attempt %d/%d): %s — retrying in %.0fs",
                            collection,
                            attempt + 1,
                            _MAX_RETRIES,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise VectorIndexError(
                            f"Upsert to '{collection}' failed: {exc}"
                        ) from exc
            _LOG.debug(
                "Upserted points %d–%d into '%s'",
                i,
                min(i + batch_size, len(points)),
                collection,
            )
        _LOG.debug("Upsert to '%s' complete", collection)

    async def _query(
        self,
        collection: Source,
        query_dense: list[float],
        query_sparse: dict[int, float],
        top_k: int,
        query_filter: Filter | None,
        query_colbert: list[list[float]] | None = None,
    ) -> list[RetrievedChunk]:
        sparse_query = SparseVector(
            indices=list(query_sparse.keys()),
            values=list(query_sparse.values()),
        )
        prefetch = [
            Prefetch(query=query_dense, using=_DENSE_VECTOR_NAME, limit=top_k * 2),
            Prefetch(query=sparse_query, using=_SPARSE_VECTOR_NAME, limit=top_k * 2),
        ]
        if self._colbert_enabled and query_colbert is not None:
            prefetch.append(
                Prefetch(query=query_colbert, using=_COLBERT_VECTOR_NAME, limit=top_k * 2)
            )

        try:
            response: QueryResponse = await self._client.query_points(
                collection_name=collection,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                query_filter=query_filter,
            )
        except Exception as exc:
            raise VectorIndexError(f"Query to collection '{collection}' failed: {exc}") from exc

        chunks: list[RetrievedChunk] = []
        skipped_count = 0
        for p in response.points:
            try:
                if p.payload is None:
                    raise ValueError("Payload is None")
                chunks.append(
                    RetrievedChunk(
                        text=p.payload["text"],
                        score=float(p.score),
                        source=p.payload["source"],
                        entity_name=p.payload.get("entity_name"),
                        entity_type=p.payload.get("entity_type"),
                        chunk_index=int(p.payload["chunk_index"]),
                        original_doc_id=p.payload["original_doc_id"],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                _LOG.warning("Malformed payload for point %s: %s", p.id, exc)
                skipped_count += 1

        if not chunks and response.points:
            raise VectorIndexError(
                f"Failed to parse any valid results from {len(response.points)} point(s)"
            )

        if skipped_count > 0:
            _LOG.warning(
                "Skipped %d malformed points out of %d total", skipped_count, len(response.points)
            )

        return chunks

    async def search(
        self,
        collection: Source,
        query_dense: list[float],
        query_sparse: dict[int, float],
        top_k: int,
        entity_name: str | None = None,
        query_colbert: list[list[float]] | None = None,
    ) -> list[RetrievedChunk]:
        _LOG.debug("Searching '%s': top_k=%d, entity_name=%s", collection, top_k, entity_name)

        normalized_name = entity_name.lower().strip() if entity_name is not None else None
        query_filter = (
            Filter(
                must=[FieldCondition(key="entity_name", match=MatchValue(value=normalized_name))]
            )
            if normalized_name is not None
            else None
        )

        chunks = await self._query(
            collection, query_dense, query_sparse, top_k, query_filter, query_colbert
        )

        if not chunks and entity_name is not None:
            _LOG.warning(
                "Entity filter for '%s' in '%s' returned 0 results; retrying without filter",
                entity_name,
                collection,
            )
            chunks = await self._query(
                collection, query_dense, query_sparse, top_k, None, query_colbert
            )

        _LOG.debug("Search '%s' → %d result(s)", collection, len(chunks))
        return chunks

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await self._client.close()
