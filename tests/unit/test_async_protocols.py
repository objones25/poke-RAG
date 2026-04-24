"""Tests for async protocol structural subtyping.

These tests verify that:
1. Well-formed mock classes satisfy the runtime_checkable isinstance checks.
2. Incomplete classes (missing required methods) do not satisfy them.

Note: runtime_checkable only checks method *presence*, not async-ness — this is
a Python limitation that applies equally to the existing sync protocols.
"""

from __future__ import annotations

from typing import Any

from src.retrieval.protocols import (
    AsyncRetrieverProtocol,
    AsyncVectorStoreProtocol,
    CacheProtocol,
)
from src.retrieval.types import EmbeddingOutput
from src.types import RetrievalResult, RetrievedChunk, Source

# ---------------------------------------------------------------------------
# Minimal concrete mocks that satisfy each protocol
# ---------------------------------------------------------------------------


class _GoodAsyncVectorStore:
    async def ensure_collections(self) -> None: ...

    async def upsert(
        self,
        collection: Source,
        documents: list[RetrievedChunk],
        embeddings: EmbeddingOutput,
    ) -> None: ...

    async def search(
        self,
        collection: Source,
        query_dense: list[float],
        query_sparse: dict[int, float],
        top_k: int,
        entity_name: str | None = None,
    ) -> list[RetrievedChunk]: ...


class _GoodAsyncRetriever:
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        sources: list[Source] | None = None,
        entity_name: str | None = None,
    ) -> RetrievalResult: ...


class _GoodCache:
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...


class _EmptyClass:
    pass


class _MissingSearchVectorStore:
    async def ensure_collections(self) -> None: ...
    async def upsert(
        self, collection: Source, documents: list[RetrievedChunk], embeddings: EmbeddingOutput
    ) -> None: ...

    # search intentionally omitted


class _MissingDeleteCache:
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None: ...

    # delete intentionally omitted


# ---------------------------------------------------------------------------
# AsyncVectorStoreProtocol
# ---------------------------------------------------------------------------


class TestAsyncVectorStoreProtocol:
    def test_full_implementation_satisfies_protocol(self) -> None:
        assert isinstance(_GoodAsyncVectorStore(), AsyncVectorStoreProtocol)

    def test_empty_class_fails_protocol(self) -> None:
        assert not isinstance(_EmptyClass(), AsyncVectorStoreProtocol)

    def test_missing_search_fails_protocol(self) -> None:
        assert not isinstance(_MissingSearchVectorStore(), AsyncVectorStoreProtocol)

    def test_protocol_has_ensure_collections(self) -> None:
        assert hasattr(AsyncVectorStoreProtocol, "ensure_collections")

    def test_protocol_has_upsert(self) -> None:
        assert hasattr(AsyncVectorStoreProtocol, "upsert")

    def test_protocol_has_search(self) -> None:
        assert hasattr(AsyncVectorStoreProtocol, "search")


# ---------------------------------------------------------------------------
# AsyncRetrieverProtocol
# ---------------------------------------------------------------------------


class TestAsyncRetrieverProtocol:
    def test_full_implementation_satisfies_protocol(self) -> None:
        assert isinstance(_GoodAsyncRetriever(), AsyncRetrieverProtocol)

    def test_empty_class_fails_protocol(self) -> None:
        assert not isinstance(_EmptyClass(), AsyncRetrieverProtocol)

    def test_protocol_has_retrieve(self) -> None:
        assert hasattr(AsyncRetrieverProtocol, "retrieve")


# ---------------------------------------------------------------------------
# CacheProtocol
# ---------------------------------------------------------------------------


class TestCacheProtocol:
    def test_full_implementation_satisfies_protocol(self) -> None:
        assert isinstance(_GoodCache(), CacheProtocol)

    def test_empty_class_fails_protocol(self) -> None:
        assert not isinstance(_EmptyClass(), CacheProtocol)

    def test_missing_delete_fails_protocol(self) -> None:
        assert not isinstance(_MissingDeleteCache(), CacheProtocol)

    def test_protocol_has_get(self) -> None:
        assert hasattr(CacheProtocol, "get")

    def test_protocol_has_set(self) -> None:
        assert hasattr(CacheProtocol, "set")

    def test_protocol_has_delete(self) -> None:
        assert hasattr(CacheProtocol, "delete")
