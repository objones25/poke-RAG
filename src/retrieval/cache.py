"""Cache layer implementations: LocalLRUCache and RedisCache."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from urllib.parse import urlparse

from cachetools import LRUCache as _LRUCache

_LOG = logging.getLogger(__name__)


class LocalLRUCache:
    """In-memory LRU cache backed by cachetools.LRUCache.

    TTL is silently ignored (no expiry for local cache).
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._cache: _LRUCache[str, Any] = _LRUCache(maxsize=maxsize)

    async def get(self, key: str) -> Any | None:
        """Return cached value or None if key is missing."""
        return self._cache.get(key)

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store value. TTL is accepted but ignored for local cache."""
        self._cache[key] = value

    async def delete(self, key: str) -> None:
        """Remove key from cache. No-op if key does not exist."""
        self._cache.pop(key, None)


class RedisCache:
    """Async Redis cache implementation.

    Connects to a Redis instance via URL (redis://host:port).
    Supports optional username and password authentication.
    Gracefully handles connection errors by returning None or no-op.
    """

    _redis: Any

    def __init__(
        self,
        redis_url: str,
        username: str = "default",
        password: str | None = None,
    ) -> None:
        """Initialize Redis cache.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379).
            username: Redis username (default: "default").
            password: Redis password (optional).

        Raises:
            ValueError: If redis_url is invalid.
        """
        import redis.asyncio

        parsed = urlparse(redis_url)
        if not parsed.hostname:
            raise ValueError(f"Invalid redis_url: {redis_url}")

        host = parsed.hostname
        port = parsed.port or 6379

        self._redis = redis.asyncio.Redis(
            host=host,
            port=port,
            username=username,
            password=password,
            decode_responses=True,
        )

    async def get(self, key: str) -> Any | None:
        """Return cached value or None if key is missing.

        Returns None on ConnectionError (graceful degradation).
        """
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except ConnectionError:
            _LOG.warning("Redis connection error in get(%s)", key)
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store value in Redis with optional TTL.

        Silently fails on ConnectionError (graceful degradation).
        """
        try:
            await self._redis.set(key, json.dumps(value), ex=ttl_seconds)
        except ConnectionError:
            _LOG.warning("Redis connection error in set(%s)", key)

    async def delete(self, key: str) -> None:
        """Remove key from Redis. No-op if key does not exist.

        Silently fails on ConnectionError (graceful degradation).
        """
        try:
            await self._redis.delete(key)
        except ConnectionError:
            _LOG.warning("Redis connection error in delete(%s)", key)

    async def aclose(self) -> None:
        """Close Redis connection."""
        await self._redis.aclose()


class CacheKey:
    """Helpers for generating cache keys."""

    @staticmethod
    def make_embedding_key(query: str) -> str:
        """Generate a cache key for an embedding query.

        Returns a deterministic key based on the query text.
        """
        digest = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"embed:{digest}"

    @staticmethod
    def make_retrieval_key(query: str, sources: list[str] | None) -> str:
        """Generate a cache key for a retrieval query.

        Source order is normalized (sorted) so that different orders
        produce the same key.

        Args:
            query: The query string.
            sources: List of sources (e.g., ["pokeapi", "smogon"]) or None.

        Returns:
            A deterministic cache key.
        """
        sources_str = "|".join(sorted(sources)) if sources else "all"
        combined = f"{query}:{sources_str}"
        digest = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"retrieval:{digest}"

    @staticmethod
    def make_rag_key(
        query: str,
        sources: list[str] | None,
        entity_name: str | None,
        top_k: int,
    ) -> str:
        """Generate a cache key for a full RAG query result.

        Includes all parameters that affect the pipeline output.
        Source order is normalized so different orderings produce the same key.
        """
        sources_str = "|".join(sorted(sources)) if sources is not None else "all"
        entity_str = entity_name if entity_name is not None else ""
        combined = f"{query}:{sources_str}:{entity_str}:{top_k}"
        digest = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"rag:{digest}"


def to_cache_dict(result: Any) -> dict[str, Any]:
    """Serialize a PipelineResult to a JSON-safe dict for cache storage."""
    return {
        "answer": result.answer,
        "sources_used": list(result.sources_used),
        "num_chunks_used": result.num_chunks_used,
        "model_name": result.model_name,
        "query": result.query,
        "confidence_score": result.confidence_score,
        "knowledge_gaps": (
            list(result.knowledge_gaps) if result.knowledge_gaps is not None else None
        ),
    }


def from_cache_dict(data: dict[str, Any]) -> Any:
    """Deserialize a PipelineResult from a cached dict.

    Returns a PipelineResult with all tuple fields restored correctly.
    Import is local to avoid a circular import between cache and pipeline.
    """
    from src.pipeline.types import PipelineResult

    gaps_raw = data.get("knowledge_gaps")
    return PipelineResult(
        answer=data["answer"],
        sources_used=tuple(data["sources_used"]),
        num_chunks_used=data["num_chunks_used"],
        model_name=data["model_name"],
        query=data["query"],
        confidence_score=data.get("confidence_score"),
        knowledge_gaps=tuple(gaps_raw) if gaps_raw is not None else None,
    )
