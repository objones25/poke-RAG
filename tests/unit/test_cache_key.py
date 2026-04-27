"""Unit tests for CacheKey.make_rag_key, to_cache_dict, and from_cache_dict."""

from __future__ import annotations

import pytest

from src.pipeline.types import PipelineResult
from src.retrieval.cache import CacheKey, from_cache_dict, to_cache_dict

# ---------------------------------------------------------------------------
# CacheKey.make_rag_key
# ---------------------------------------------------------------------------


class TestMakeRagKey:
    def _make_result(
        self,
        query: str = "test query",
        sources: list[str] | None = None,
        entity_name: str | None = None,
        top_k: int = 5,
    ) -> str:
        return CacheKey.make_rag_key(query, sources, entity_name, top_k)

    def test_returns_string_with_rag_prefix(self) -> None:
        key = self._make_result()
        assert key.startswith("rag:")

    def test_deterministic_same_args(self) -> None:
        k1 = CacheKey.make_rag_key("who is pikachu", ["pokeapi"], "pikachu", 5)
        k2 = CacheKey.make_rag_key("who is pikachu", ["pokeapi"], "pikachu", 5)
        assert k1 == k2

    def test_source_order_normalized(self) -> None:
        k1 = CacheKey.make_rag_key("q", ["smogon", "pokeapi"], None, 5)
        k2 = CacheKey.make_rag_key("q", ["pokeapi", "smogon"], None, 5)
        assert k1 == k2

    def test_none_sources_differs_from_empty(self) -> None:
        k1 = CacheKey.make_rag_key("q", None, None, 5)
        k2 = CacheKey.make_rag_key("q", [], None, 5)
        assert k1 != k2

    def test_different_queries_differ(self) -> None:
        k1 = CacheKey.make_rag_key("query a", None, None, 5)
        k2 = CacheKey.make_rag_key("query b", None, None, 5)
        assert k1 != k2

    def test_different_entity_names_differ(self) -> None:
        k1 = CacheKey.make_rag_key("q", None, "pikachu", 5)
        k2 = CacheKey.make_rag_key("q", None, "charizard", 5)
        assert k1 != k2

    def test_different_top_k_differ(self) -> None:
        k1 = CacheKey.make_rag_key("q", None, None, 5)
        k2 = CacheKey.make_rag_key("q", None, None, 10)
        assert k1 != k2

    def test_none_entity_differs_from_named(self) -> None:
        k1 = CacheKey.make_rag_key("q", None, None, 5)
        k2 = CacheKey.make_rag_key("q", None, "pikachu", 5)
        assert k1 != k2


# ---------------------------------------------------------------------------
# to_cache_dict / from_cache_dict
# ---------------------------------------------------------------------------


def _minimal_result() -> PipelineResult:
    return PipelineResult(
        answer="It's a fire-type.",
        sources_used=("pokeapi",),
        num_chunks_used=3,
        model_name="gemma-4-E4B-it",
        query="What type is Charizard?",
    )


def _full_result() -> PipelineResult:
    return PipelineResult(
        answer="Charizard is a Fire/Flying type.",
        sources_used=("pokeapi", "bulbapedia"),
        num_chunks_used=5,
        model_name="gemma-4-E4B-it",
        query="What type is Charizard?",
        confidence_score=0.92,
        knowledge_gaps=("gen6", "ou"),
    )


class TestToCacheDict:
    def test_returns_dict(self) -> None:
        assert isinstance(to_cache_dict(_minimal_result()), dict)

    def test_all_keys_present(self) -> None:
        d = to_cache_dict(_minimal_result())
        assert set(d.keys()) == {
            "answer",
            "sources_used",
            "num_chunks_used",
            "model_name",
            "query",
            "confidence_score",
            "knowledge_gaps",
        }

    def test_sources_used_serialized_as_list(self) -> None:
        d = to_cache_dict(_minimal_result())
        assert isinstance(d["sources_used"], list)

    def test_none_optional_fields(self) -> None:
        d = to_cache_dict(_minimal_result())
        assert d["confidence_score"] is None
        assert d["knowledge_gaps"] is None

    def test_knowledge_gaps_serialized_as_list(self) -> None:
        d = to_cache_dict(_full_result())
        assert isinstance(d["knowledge_gaps"], list)
        assert d["knowledge_gaps"] == ["gen6", "ou"]

    def test_confidence_score_preserved(self) -> None:
        d = to_cache_dict(_full_result())
        assert d["confidence_score"] == pytest.approx(0.92)


class TestFromCacheDict:
    def test_roundtrip_minimal(self) -> None:
        original = _minimal_result()
        restored = from_cache_dict(to_cache_dict(original))
        assert restored == original

    def test_roundtrip_full(self) -> None:
        original = _full_result()
        restored = from_cache_dict(to_cache_dict(original))
        assert restored == original

    def test_sources_used_restored_as_tuple(self) -> None:
        restored = from_cache_dict(to_cache_dict(_minimal_result()))
        assert isinstance(restored.sources_used, tuple)

    def test_knowledge_gaps_restored_as_tuple(self) -> None:
        restored = from_cache_dict(to_cache_dict(_full_result()))
        assert isinstance(restored.knowledge_gaps, tuple)

    def test_none_knowledge_gaps_stays_none(self) -> None:
        restored = from_cache_dict(to_cache_dict(_minimal_result()))
        assert restored.knowledge_gaps is None

    def test_none_confidence_score_stays_none(self) -> None:
        restored = from_cache_dict(to_cache_dict(_minimal_result()))
        assert restored.confidence_score is None
