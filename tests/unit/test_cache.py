"""Unit tests for src/retrieval/cache.py — LocalLRUCache, RedisCache, and CacheKey."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.retrieval.cache import CacheKey, LocalLRUCache, RedisCache


@pytest.mark.unit
class TestLocalLRUCache:
    @pytest.mark.anyio
    async def test_get_returns_none_for_missing_key(self) -> None:
        cache = LocalLRUCache(maxsize=10)
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.anyio
    async def test_get_returns_stored_value_after_set(self) -> None:
        cache = LocalLRUCache(maxsize=10)
        value = {"query": "test", "results": [1, 2, 3]}
        await cache.set("key1", value)
        result = await cache.get("key1")
        assert result == value

    @pytest.mark.anyio
    async def test_delete_removes_key(self) -> None:
        cache = LocalLRUCache(maxsize=10)
        await cache.set("key1", {"data": "value"})
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.anyio
    async def test_eviction_removes_oldest_when_maxsize_exceeded(self) -> None:
        cache = LocalLRUCache(maxsize=2)
        await cache.set("key1", {"data": 1})
        await cache.set("key2", {"data": 2})
        await cache.set("key3", {"data": 3})
        result1 = await cache.get("key1")
        assert result1 is None

    @pytest.mark.anyio
    async def test_set_with_ttl_seconds_accepted(self) -> None:
        cache = LocalLRUCache(maxsize=10)
        value = {"query": "test"}
        await cache.set("key1", value, ttl_seconds=3600)
        result = await cache.get("key1")
        assert result == value

    @pytest.mark.anyio
    async def test_set_overwrites_existing_key(self) -> None:
        cache = LocalLRUCache(maxsize=10)
        await cache.set("key1", {"data": "old"})
        await cache.set("key1", {"data": "new"})
        result = await cache.get("key1")
        assert result == {"data": "new"}


@pytest.mark.unit
class TestRedisCache:
    @pytest.mark.anyio
    async def test_get_returns_none_when_key_missing(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        cache = RedisCache.__new__(RedisCache)
        cache._redis = mock_redis
        result = await cache.get("nonexistent")
        assert result is None
        mock_redis.get.assert_called_once_with("nonexistent")

    @pytest.mark.anyio
    async def test_get_returns_deserialized_dict(self) -> None:
        mock_redis = AsyncMock()
        test_dict = {"query": "test", "results": [1, 2]}
        mock_redis.get.return_value = json.dumps(test_dict)
        cache = RedisCache.__new__(RedisCache)
        cache._redis = mock_redis
        result = await cache.get("key1")
        assert result == test_dict
        mock_redis.get.assert_called_once_with("key1")

    @pytest.mark.anyio
    async def test_set_calls_redis_with_ttl(self) -> None:
        mock_redis = AsyncMock()
        cache = RedisCache.__new__(RedisCache)
        cache._redis = mock_redis
        value = {"data": "test"}
        await cache.set("key1", value, ttl_seconds=7200)
        mock_redis.set.assert_called_once_with("key1", json.dumps(value), ex=7200)

    @pytest.mark.anyio
    async def test_set_calls_redis_with_none_ttl(self) -> None:
        mock_redis = AsyncMock()
        cache = RedisCache.__new__(RedisCache)
        cache._redis = mock_redis
        value = {"data": "test"}
        await cache.set("key1", value, ttl_seconds=None)
        mock_redis.set.assert_called_once_with("key1", json.dumps(value), ex=None)

    @pytest.mark.anyio
    async def test_delete_calls_redis_delete(self) -> None:
        mock_redis = AsyncMock()
        cache = RedisCache.__new__(RedisCache)
        cache._redis = mock_redis
        await cache.delete("key1")
        mock_redis.delete.assert_called_once_with("key1")

    @pytest.mark.anyio
    async def test_get_returns_none_on_connection_error(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = ConnectionError("Connection failed")
        cache = RedisCache.__new__(RedisCache)
        cache._redis = mock_redis
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.anyio
    async def test_set_gracefully_handles_connection_error(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.set.side_effect = ConnectionError("Connection failed")
        cache = RedisCache.__new__(RedisCache)
        cache._redis = mock_redis
        await cache.set("key1", {"data": "test"})

    @pytest.mark.anyio
    async def test_delete_gracefully_handles_connection_error(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.delete.side_effect = ConnectionError("Connection failed")
        cache = RedisCache.__new__(RedisCache)
        cache._redis = mock_redis
        await cache.delete("key1")

    @pytest.mark.anyio
    async def test_aclose_calls_redis_aclose(self) -> None:
        mock_redis = AsyncMock()
        cache = RedisCache.__new__(RedisCache)
        cache._redis = mock_redis
        await cache.aclose()
        mock_redis.aclose.assert_called_once()


@pytest.mark.unit
class TestCacheKey:
    def test_same_query_produces_same_key(self) -> None:
        key1 = CacheKey.make_embedding_key("What is Pikachu?")
        key2 = CacheKey.make_embedding_key("What is Pikachu?")
        assert key1 == key2

    def test_different_queries_produce_different_keys(self) -> None:
        key1 = CacheKey.make_embedding_key("What is Pikachu?")
        key2 = CacheKey.make_embedding_key("What is Charizard?")
        assert key1 != key2

    def test_sources_order_does_not_matter(self) -> None:
        sources1 = ["pokeapi", "smogon", "bulbapedia"]
        sources2 = ["bulbapedia", "smogon", "pokeapi"]
        key1 = CacheKey.make_retrieval_key("test query", sources1)
        key2 = CacheKey.make_retrieval_key("test query", sources2)
        assert key1 == key2

    def test_embedding_key_starts_with_embed_prefix(self) -> None:
        key = CacheKey.make_embedding_key("test")
        assert key.startswith("embed:")

    def test_retrieval_key_starts_with_retrieval_prefix(self) -> None:
        key = CacheKey.make_retrieval_key("test", ["pokeapi"])
        assert key.startswith("retrieval:")

    def test_retrieval_key_with_none_sources(self) -> None:
        key = CacheKey.make_retrieval_key("test query", None)
        assert key.startswith("retrieval:")


@pytest.mark.unit
class TestSettingsCacheConfig:
    def test_cache_enabled_true_from_env(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("CACHE_ENABLED", "true")
        settings = Settings.from_env()
        assert settings.cache_enabled is True

    def test_cache_enabled_false_by_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("CACHE_ENABLED", raising=False)
        settings = Settings.from_env()
        assert settings.cache_enabled is False

    def test_cache_ttl_seconds_from_env(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "7200")
        settings = Settings.from_env()
        assert settings.cache_ttl_seconds == 7200

    def test_cache_ttl_seconds_defaults_to_3600(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("CACHE_TTL_SECONDS", raising=False)
        settings = Settings.from_env()
        assert settings.cache_ttl_seconds == 3600

    def test_cache_max_size_from_env(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("CACHE_MAX_SIZE", "5000")
        settings = Settings.from_env()
        assert settings.cache_max_size == 5000

    def test_cache_max_size_defaults_to_1000(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("CACHE_MAX_SIZE", raising=False)
        settings = Settings.from_env()
        assert settings.cache_max_size == 1000

    def test_redis_url_from_env(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        settings = Settings.from_env()
        assert settings.redis_url == "redis://localhost:6379"

    def test_redis_url_defaults_to_none(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("REDIS_URL", raising=False)
        settings = Settings.from_env()
        assert settings.redis_url is None

    def test_redis_username_from_env(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("REDIS_USERNAME", "myuser")
        settings = Settings.from_env()
        assert settings.redis_username == "myuser"

    def test_redis_username_defaults_to_default(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("REDIS_USERNAME", raising=False)
        settings = Settings.from_env()
        assert settings.redis_username == "default"

    def test_redis_password_from_env(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("REDIS_PASSWORD", "secret_password")
        settings = Settings.from_env()
        assert settings.redis_password is not None
        assert settings.redis_password.get_secret_value() == "secret_password"

    def test_redis_password_defaults_to_none(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("REDIS_PASSWORD", raising=False)
        settings = Settings.from_env()
        assert settings.redis_password is None

    def test_async_pipeline_enabled_from_env(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("ASYNC_PIPELINE_ENABLED", "true")
        settings = Settings.from_env()
        assert settings.async_pipeline_enabled is True

    def test_async_pipeline_enabled_defaults_to_false(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("ASYNC_PIPELINE_ENABLED", raising=False)
        settings = Settings.from_env()
        assert settings.async_pipeline_enabled is False
