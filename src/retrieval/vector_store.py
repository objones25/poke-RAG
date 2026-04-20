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


class QdrantVectorStore:
    """One Qdrant collection per source, hybrid dense+sparse vectors."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def ensure_collections(self) -> None:
        _LOG.info("Ensuring %d Qdrant collections: %s", len(_SOURCES), _SOURCES)
        for source in _SOURCES:
            with contextlib.suppress(UnexpectedResponse):
                self._client.create_collection(
                    collection_name=source,
                    vectors_config={
                        _DENSE_VECTOR_NAME: VectorParams(
                            size=_DENSE_DIM, distance=Distance.COSINE
                        ),
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
                    "pokemon_name": doc.pokemon_name,
                    "chunk_index": doc.chunk_index,
                    "original_doc_id": doc.original_doc_id,
                },
            )
            for i, doc in enumerate(documents)
        ]
        self._client.upsert(collection_name=collection, points=points)
        _LOG.debug("Upsert to '%s' complete", collection)

    def search(
        self,
        collection: Source,
        query_dense: list[float],
        query_sparse: dict[int, float],
        top_k: int,
        pokemon_name: str | None = None,
    ) -> list[RetrievedChunk]:
        _LOG.debug("Searching '%s': top_k=%d, pokemon_name=%s", collection, top_k, pokemon_name)

        query_filter = (
            Filter(
                must=[
                    FieldCondition(
                        key="pokemon_name", match=MatchValue(value=pokemon_name)
                    )
                ]
            )
            if pokemon_name is not None
            else None
        )

        sparse_query = SparseVector(
            indices=list(query_sparse.keys()),
            values=list(query_sparse.values()),
        )

        response = self._client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(
                    query=query_dense,
                    using=_DENSE_VECTOR_NAME,
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=sparse_query,
                    using=_SPARSE_VECTOR_NAME,
                    limit=top_k * 2,
                ),
            ],
            query=Fusion.RRF,
            limit=top_k,
            query_filter=query_filter,
        )

        chunks: list[RetrievedChunk] = []
        for p in response.points:
            try:
                chunks.append(
                    RetrievedChunk(
                        text=p.payload["text"],
                        score=float(p.score),
                        source=p.payload["source"],
                        pokemon_name=p.payload.get("pokemon_name"),
                        chunk_index=int(p.payload["chunk_index"]),
                        original_doc_id=p.payload["original_doc_id"],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise VectorIndexError(f"Malformed payload for point {p.id}: {exc}") from exc

        _LOG.debug("Search '%s' → %d result(s)", collection, len(chunks))
        return chunks
