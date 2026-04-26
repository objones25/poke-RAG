"""Unit tests for eval metric functions in scripts/eval/run_eval.py (v3)."""

from __future__ import annotations

import pytest


def _import_metrics():
    from scripts.eval.run_eval import (
        hard_negative_leak_at_k,
        hit_at_k,
        mrr_at_k,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        resolve_chunk_id,
    )

    return (
        hit_at_k,
        mrr_at_k,
        recall_at_k,
        precision_at_k,
        ndcg_at_k,
        hard_negative_leak_at_k,
        resolve_chunk_id,
    )


# ---------------------------------------------------------------------------
# resolve_chunk_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveChunkId:
    def _resolve(self, **kwargs):
        from dataclasses import dataclass

        from scripts.eval.run_eval import resolve_chunk_id

        @dataclass
        class FakeChunk:
            entity_name: str | None = None
            original_doc_id: str | None = None
            text: str = ""

        return resolve_chunk_id(FakeChunk(**kwargs))

    def test_canonical_id_from_metadata(self) -> None:
        cid = self._resolve(entity_name="venusaur", original_doc_id="pokemon_species_42")
        assert cid == "pokemon_species:venusaur"

    def test_lowercases_entity_name(self) -> None:
        cid = self._resolve(entity_name="Charizard", original_doc_id="pokemon_species_0")
        assert cid == "pokemon_species:charizard"

    def test_strips_trailing_index_from_doc_id(self) -> None:
        cid = self._resolve(entity_name="earthquake", original_doc_id="move_1234")
        assert cid == "move:earthquake"

    def test_returns_none_when_no_metadata_no_text(self) -> None:
        from dataclasses import dataclass

        from scripts.eval.run_eval import resolve_chunk_id

        @dataclass
        class Empty:
            pass

        assert resolve_chunk_id(Empty()) is None

    def test_fallback_to_text_for_pokemon_species(self) -> None:
        cid = self._resolve(text="Bulbasaur is a Grass Poison Pokémon.")
        assert cid == "pokemon_species:bulbasaur"

    def test_fallback_to_text_for_move(self) -> None:
        cid = self._resolve(text="Earthquake is a Pokémon move.")
        assert cid == "move:earthquake"

    def test_fallback_to_text_for_ability(self) -> None:
        cid = self._resolve(text="Intimidate is a Pokémon ability.")
        assert cid == "ability:intimidate"

    def test_fallback_to_text_for_item(self) -> None:
        cid = self._resolve(text="Choice Band is a Pokémon item.")
        assert cid == "item:choice band"


# ---------------------------------------------------------------------------
# hit_at_k — takes list[str | None] IDs + gold set
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHitAtK:
    def test_true_when_gold_in_top_k(self) -> None:
        hit_at_k, *_ = _import_metrics()
        ids = ["other:foo", "pokemon:venusaur", "other:bar"]
        assert hit_at_k(ids, gold={"pokemon:venusaur"}, k=5)

    def test_false_when_gold_below_k(self) -> None:
        hit_at_k, *_ = _import_metrics()
        ids = ["x:a", "x:b", "x:c", "pokemon:venusaur", "x:e"]
        assert not hit_at_k(ids, gold={"pokemon:venusaur"}, k=3)

    def test_true_at_exact_k_boundary(self) -> None:
        hit_at_k, *_ = _import_metrics()
        ids = ["x:a", "x:b", "pokemon:venusaur"]
        assert hit_at_k(ids, gold={"pokemon:venusaur"}, k=3)

    def test_false_when_no_relevant(self) -> None:
        hit_at_k, *_ = _import_metrics()
        ids = ["x:a", "x:b", "x:c"]
        assert not hit_at_k(ids, gold={"pokemon:venusaur"}, k=5)

    def test_empty_ids_is_false(self) -> None:
        hit_at_k, *_ = _import_metrics()
        assert not hit_at_k([], gold={"pokemon:venusaur"}, k=5)

    def test_k_larger_than_list(self) -> None:
        hit_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur"]
        assert hit_at_k(ids, gold={"pokemon:venusaur"}, k=100)

    def test_case_insensitive(self) -> None:
        hit_at_k, *_ = _import_metrics()
        ids = ["Pokemon:Venusaur"]
        assert hit_at_k(ids, gold={"pokemon:venusaur"}, k=5)

    def test_none_ids_are_skipped(self) -> None:
        hit_at_k, *_ = _import_metrics()
        ids = [None, None, "pokemon:venusaur"]
        assert hit_at_k(ids, gold={"pokemon:venusaur"}, k=5)

    def test_empty_gold_is_false(self) -> None:
        hit_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur"]
        assert not hit_at_k(ids, gold=set(), k=5)


# ---------------------------------------------------------------------------
# mrr_at_k
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMrrAtK:
    def test_perfect_rank(self) -> None:
        _, mrr_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur", "x:b", "x:c"]
        assert mrr_at_k(ids, gold={"pokemon:venusaur"}, k=10) == pytest.approx(1.0)

    def test_second_rank(self) -> None:
        _, mrr_at_k, *_ = _import_metrics()
        ids = ["x:a", "pokemon:venusaur", "x:c"]
        assert mrr_at_k(ids, gold={"pokemon:venusaur"}, k=10) == pytest.approx(0.5)

    def test_third_rank(self) -> None:
        _, mrr_at_k, *_ = _import_metrics()
        ids = ["x:a", "x:b", "pokemon:venusaur"]
        assert mrr_at_k(ids, gold={"pokemon:venusaur"}, k=10) == pytest.approx(1 / 3)

    def test_zero_when_none_relevant(self) -> None:
        _, mrr_at_k, *_ = _import_metrics()
        ids = ["x:a", "x:b", "x:c"]
        assert mrr_at_k(ids, gold={"pokemon:venusaur"}, k=10) == 0.0

    def test_zero_when_relevant_below_k(self) -> None:
        _, mrr_at_k, *_ = _import_metrics()
        ids = ["x:a", "x:b", "x:c", "pokemon:venusaur"]
        assert mrr_at_k(ids, gold={"pokemon:venusaur"}, k=3) == 0.0

    def test_empty_ids_is_zero(self) -> None:
        _, mrr_at_k, *_ = _import_metrics()
        assert mrr_at_k([], gold={"pokemon:venusaur"}, k=10) == 0.0

    def test_case_insensitive(self) -> None:
        _, mrr_at_k, *_ = _import_metrics()
        ids = ["Pokemon:Venusaur"]
        assert mrr_at_k(ids, gold={"pokemon:venusaur"}, k=10) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecallAtK:
    def test_full_recall_all_gold_found(self) -> None:
        _, _, recall_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur", "move:earthquake"]
        gold = {"pokemon:venusaur", "move:earthquake"}
        assert recall_at_k(ids, gold=gold, k=10) == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        _, _, recall_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur", "x:b"]
        gold = {"pokemon:venusaur", "move:earthquake"}
        assert recall_at_k(ids, gold=gold, k=10) == pytest.approx(0.5)

    def test_zero_recall_none_found(self) -> None:
        _, _, recall_at_k, *_ = _import_metrics()
        ids = ["x:a", "x:b"]
        assert recall_at_k(ids, gold={"pokemon:venusaur"}, k=10) == 0.0

    def test_empty_gold_is_zero(self) -> None:
        _, _, recall_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur"]
        assert recall_at_k(ids, gold=set(), k=10) == 0.0

    def test_deduplicates_repeated_ids(self) -> None:
        _, _, recall_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur", "pokemon:venusaur", "pokemon:venusaur"]
        gold = {"pokemon:venusaur", "move:earthquake"}
        assert recall_at_k(ids, gold=gold, k=10) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        _, _, _, precision_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur", "move:earthquake"]
        gold = {"pokemon:venusaur", "move:earthquake"}
        assert precision_at_k(ids, gold=gold, k=10) == pytest.approx(1.0)

    def test_half_precision(self) -> None:
        _, _, _, precision_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur", "x:irrelevant"]
        assert precision_at_k(ids, gold={"pokemon:venusaur"}, k=10) == pytest.approx(0.5)

    def test_zero_precision_none_relevant(self) -> None:
        _, _, _, precision_at_k, *_ = _import_metrics()
        ids = ["x:a", "x:b"]
        assert precision_at_k(ids, gold={"pokemon:venusaur"}, k=10) == 0.0

    def test_deduplicates_by_id(self) -> None:
        _, _, _, precision_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur", "pokemon:venusaur"]
        assert precision_at_k(ids, gold={"pokemon:venusaur"}, k=10) == pytest.approx(1.0)

    def test_zero_k_is_zero(self) -> None:
        _, _, _, precision_at_k, *_ = _import_metrics()
        assert precision_at_k(["pokemon:venusaur"], gold={"pokemon:venusaur"}, k=0) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNdcgAtK:
    def test_perfect_ndcg(self) -> None:
        _, _, _, _, ndcg_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur"]
        assert ndcg_at_k(ids, gold={"pokemon:venusaur"}, k=10) == pytest.approx(1.0)

    def test_bounded_between_zero_and_one(self) -> None:
        _, _, _, _, ndcg_at_k, *_ = _import_metrics()
        ids = ["x:a", "pokemon:venusaur", "move:earthquake", "x:b"]
        gold = {"pokemon:venusaur", "move:earthquake"}
        result = ndcg_at_k(ids, gold=gold, k=10)
        assert 0.0 <= result <= 1.0

    def test_zero_when_none_in_gold(self) -> None:
        _, _, _, _, ndcg_at_k, *_ = _import_metrics()
        ids = ["x:a", "x:b"]
        assert ndcg_at_k(ids, gold={"pokemon:venusaur"}, k=10) == 0.0

    def test_empty_gold_is_zero(self) -> None:
        _, _, _, _, ndcg_at_k, *_ = _import_metrics()
        assert ndcg_at_k(["pokemon:venusaur"], gold=set(), k=10) == 0.0

    def test_later_rank_gives_lower_score(self) -> None:
        _, _, _, _, ndcg_at_k, *_ = _import_metrics()
        ids_rank1 = ["pokemon:venusaur", "x:b", "x:c"]
        ids_rank3 = ["x:a", "x:b", "pokemon:venusaur"]
        gold = {"pokemon:venusaur"}
        assert ndcg_at_k(ids_rank1, gold=gold, k=10) > ndcg_at_k(ids_rank3, gold=gold, k=10)

    def test_deduplicates_repeated_gold_hits(self) -> None:
        _, _, _, _, ndcg_at_k, *_ = _import_metrics()
        ids = ["pokemon:venusaur", "pokemon:venusaur", "pokemon:venusaur"]
        assert ndcg_at_k(ids, gold={"pokemon:venusaur"}, k=10) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# hard_negative_leak_at_k
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHardNegativeLeakAtK:
    def test_full_leak_all_hard_negs(self) -> None:
        *_, hard_negative_leak_at_k, _ = _import_metrics()
        ids = ["pokemon:charizard", "pokemon:blastoise"]
        hard_negs = {"pokemon:charizard", "pokemon:blastoise"}
        result = hard_negative_leak_at_k(ids, hard_negs=hard_negs, k=10)
        assert result == pytest.approx(1.0)

    def test_half_leak(self) -> None:
        *_, hard_negative_leak_at_k, _ = _import_metrics()
        ids = ["pokemon:charizard", "x:irrelevant"]
        result = hard_negative_leak_at_k(ids, hard_negs={"pokemon:charizard"}, k=10)
        assert result == pytest.approx(0.5)

    def test_zero_leak_none_match(self) -> None:
        *_, hard_negative_leak_at_k, _ = _import_metrics()
        ids = ["pokemon:venusaur", "move:earthquake"]
        assert hard_negative_leak_at_k(ids, hard_negs={"pokemon:charizard"}, k=10) == 0.0

    def test_empty_hard_negs_is_zero(self) -> None:
        *_, hard_negative_leak_at_k, _ = _import_metrics()
        assert hard_negative_leak_at_k(["pokemon:venusaur"], hard_negs=set(), k=10) == 0.0

    def test_zero_k_is_zero(self) -> None:
        *_, hard_negative_leak_at_k, _ = _import_metrics()
        ids = ["pokemon:charizard"]
        assert hard_negative_leak_at_k(ids, hard_negs={"pokemon:charizard"}, k=0) == 0.0

    def test_deduplicates_repeated_ids(self) -> None:
        *_, hard_negative_leak_at_k, _ = _import_metrics()
        ids = ["pokemon:charizard", "pokemon:charizard"]
        result = hard_negative_leak_at_k(ids, hard_negs={"pokemon:charizard"}, k=10)
        assert result == pytest.approx(1.0)
