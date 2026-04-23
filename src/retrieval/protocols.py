"""Abstract protocols for all retrieval dependencies.

Every concrete class (BGEEmbedder, QdrantVectorStore, BGEReranker, Retriever)
must satisfy the corresponding protocol. This enables unit tests to inject
mocks without importing heavy ML dependencies.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.retrieval.types import EmbeddingOutput
from src.types import RetrievalResult, RetrievedChunk, Source


@runtime_checkable
class EmbedderProtocol(Protocol):
    def encode(self, texts: list[str]) -> EmbeddingOutput:
        """Embed a batch of texts. Returns dense (n×1024) and sparse (token_id→weight).

        Raises:
            EmbeddingError: If embedding fails.
        """
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    def ensure_collections(self) -> None:
        """Create the three source collections if they don't exist."""
        ...

    def upsert(
        self,
        collection: Source,
        documents: list[RetrievedChunk],
        embeddings: EmbeddingOutput,
    ) -> None:
        """Upsert documents with their precomputed embeddings into a collection."""
        ...

    def search(
        self,
        collection: Source,
        query_dense: list[float],
        query_sparse: dict[int, float],
        top_k: int,
        entity_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid dense+sparse search with optional entity_name payload filter.

        Raises:
            OSError: If vector store is unavailable or connection fails.
        """
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank documents by relevance to query. Returns new frozen chunks with updated scores.

        Raises:
            ValueError: If chunks are invalid or reranking fails.
            RuntimeError: If reranker model inference fails.
        """
        ...


@runtime_checkable
class QueryTransformerProtocol(Protocol):
    def transform(self, query: str) -> str:
        """Transform a query before embedding. Returns original query on failure."""
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        sources: list[Source] | None = None,
        entity_name: str | None = None,
    ) -> RetrievalResult:
        """Retrieve top_k chunks for query across specified sources (None = all three).

        Raises:
            RetrievalError: If retrieval fails or no documents found.
        """
        ...
