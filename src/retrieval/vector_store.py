"""Qdrant-backed vector store with hybrid dense+sparse search."""

from __future__ import annotations

import contextlib
import logging
import uuid
from typing import Any

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
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
_UPSERT_BATCH_SIZE = 100


class QdrantVectorStore:
    """One Qdrant collection per source, hybrid dense+sparse vectors."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def ensure_collections(self) -> None:
        _LOG.info("Ensuring %d Qdrant collections: %s", len(_SOURCES), _SOURCES)
        for source in _SOURCES:
            # suppress AlreadyExists: Qdrant raises UnexpectedResponse(409) on duplicate create
            with contextlib.suppress(UnexpectedResponse):
                self._client.create_collection(
                    collection_name=source,
                    vectors_config={
                        _DENSE_VECTOR_NAME: VectorParams(size=_DENSE_DIM, distance=Distance.COSINE),
                    },
                    sparse_vectors_config={
                        _SPARSE_VECTOR_NAME: SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        ),
                    },
                )
            _LOG.debug("Collection '%s' ready", source)

    def upsert(
        self,
        collection: Source,
        documents: list[RetrievedChunk],
        embeddings: EmbeddingOutput,
    ) -> None:
        _LOG.info("Upserting %d point(s) into '%s'", len(documents), collection)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.original_doc_id}:{doc.chunk_index}")),
                vector={
                    _DENSE_VECTOR_NAME: embeddings.dense[i],
                    _SPARSE_VECTOR_NAME: SparseVector(
                        indices=list(embeddings.sparse[i].keys()),
                        values=list(embeddings.sparse[i].values()),
                    ),
                },
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
            for i, doc in enumerate(documents)
        ]
        for i in range(0, len(points), _UPSERT_BATCH_SIZE):
            self._client.upsert(
                collection_name=collection, points=points[i : i + _UPSERT_BATCH_SIZE]
            )
            _LOG.debug(
                "Upserted points %d–%d into '%s'",
                i,
                min(i + _UPSERT_BATCH_SIZE, len(points)),
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
    ) -> list[RetrievedChunk]:
        sparse_query = SparseVector(
            indices=list(query_sparse.keys()),
            values=list(query_sparse.values()),
        )

        response = self._client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=query_dense, using=_DENSE_VECTOR_NAME, limit=top_k * 2),
                Prefetch(query=sparse_query, using=_SPARSE_VECTOR_NAME, limit=top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            query_filter=query_filter,
        )

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

        chunks = self._query(collection, query_dense, query_sparse, top_k, query_filter)

        if not chunks and entity_name is not None:
            _LOG.warning(
                "Entity filter for '%s' in '%s' returned 0 results; retrying without filter",
                entity_name,
                collection,
            )
            chunks = self._query(collection, query_dense, query_sparse, top_k, None)

        _LOG.debug("Search '%s' → %d result(s)", collection, len(chunks))
        return chunks
