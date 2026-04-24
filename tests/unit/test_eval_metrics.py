"""Unit tests for eval metric functions in scripts/eval/run_eval.py."""

from __future__ import annotations

import pytest


def _import_metrics():
    from scripts.eval.run_eval import hit_at_k, is_relevant, mrr_at_k

    return hit_at_k, is_relevant, mrr_at_k


@pytest.mark.unit
class TestIsRelevant:
    def test_true_when_keyword_present(self) -> None:
        _, is_relevant, _ = _import_metrics()
        assert is_relevant("Pikachu has 90 base Speed.", ["Speed", "Electric"])

    def test_case_insensitive(self) -> None:
        _, is_relevant, _ = _import_metrics()
        assert is_relevant("pikachu has 90 base speed.", ["Speed"])

    def test_false_when_no_keyword(self) -> None:
        _, is_relevant, _ = _import_metrics()
        assert not is_relevant("Bulbasaur is Grass/Poison.", ["Fire", "Water"])

    def test_empty_keywords_is_false(self) -> None:
        _, is_relevant, _ = _import_metrics()
        assert not is_relevant("Some text.", [])

    def test_empty_text_is_false(self) -> None:
        _, is_relevant, _ = _import_metrics()
        assert not is_relevant("", ["keyword"])

    def test_substring_match(self) -> None:
        _, is_relevant, _ = _import_metrics()
        assert is_relevant("Thunderbolt is a powerful move.", ["Thunder"])


@pytest.mark.unit
class TestHitAtK:
    def _make_chunks(self, texts: list[str]):
        from dataclasses import dataclass

        @dataclass
        class FakeChunk:
            text: str

        return [FakeChunk(text=t) for t in texts]

    def test_true_when_relevant_in_top_k(self) -> None:
        hit_at_k, _, _ = _import_metrics()
        chunks = self._make_chunks(["irrelevant", "Pikachu has 90 Speed", "other"])
        assert hit_at_k(chunks, keywords=["Speed"], k=5)

    def test_false_when_relevant_below_k(self) -> None:
        hit_at_k, _, _ = _import_metrics()
        texts = ["no", "no", "no", "Pikachu has 90 Speed", "other"]
        chunks = self._make_chunks(texts)
        assert not hit_at_k(chunks, keywords=["Speed"], k=3)

    def test_true_at_exact_k_boundary(self) -> None:
        hit_at_k, _, _ = _import_metrics()
        texts = ["no", "no", "Pikachu has 90 Speed"]
        chunks = self._make_chunks(texts)
        assert hit_at_k(chunks, keywords=["Speed"], k=3)

    def test_false_when_no_relevant(self) -> None:
        hit_at_k, _, _ = _import_metrics()
        chunks = self._make_chunks(["no", "also no", "nope"])
        assert not hit_at_k(chunks, keywords=["Speed"], k=5)

    def test_empty_chunks_is_false(self) -> None:
        hit_at_k, _, _ = _import_metrics()
        assert not hit_at_k([], keywords=["Speed"], k=5)

    def test_k_larger_than_list(self) -> None:
        hit_at_k, _, _ = _import_metrics()
        chunks = self._make_chunks(["Pikachu has 90 Speed"])
        assert hit_at_k(chunks, keywords=["Speed"], k=100)


@pytest.mark.unit
class TestMrrAtK:
    def _make_chunks(self, texts: list[str]):
        from dataclasses import dataclass

        @dataclass
        class FakeChunk:
            text: str

        return [FakeChunk(text=t) for t in texts]

    def test_perfect_rank(self) -> None:
        _, _, mrr_at_k = _import_metrics()
        chunks = self._make_chunks(["Pikachu has 90 Speed", "other", "more"])
        assert mrr_at_k(chunks, keywords=["Speed"], k=10) == pytest.approx(1.0)

    def test_second_rank(self) -> None:
        _, _, mrr_at_k = _import_metrics()
        chunks = self._make_chunks(["no", "Pikachu has 90 Speed", "other"])
        assert mrr_at_k(chunks, keywords=["Speed"], k=10) == pytest.approx(0.5)

    def test_third_rank(self) -> None:
        _, _, mrr_at_k = _import_metrics()
        chunks = self._make_chunks(["no", "no", "Pikachu has 90 Speed"])
        assert mrr_at_k(chunks, keywords=["Speed"], k=10) == pytest.approx(1 / 3)

    def test_zero_when_none_relevant(self) -> None:
        _, _, mrr_at_k = _import_metrics()
        chunks = self._make_chunks(["no", "also no", "nope"])
        assert mrr_at_k(chunks, keywords=["Speed"], k=10) == 0.0

    def test_zero_when_relevant_below_k(self) -> None:
        _, _, mrr_at_k = _import_metrics()
        texts = ["no", "no", "no", "Pikachu has 90 Speed"]
        chunks = self._make_chunks(texts)
        assert mrr_at_k(chunks, keywords=["Speed"], k=3) == 0.0

    def test_empty_chunks_is_zero(self) -> None:
        _, _, mrr_at_k = _import_metrics()
        assert mrr_at_k([], keywords=["Speed"], k=10) == 0.0
