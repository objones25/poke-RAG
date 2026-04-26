"""Unit tests for KnowledgeRefiner — written RED before implementation."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from tests.conftest import make_chunk


def _make_reranker(scores: list[float]) -> MagicMock:
    """Reranker mock that assigns scores to strips in input order, then sorts descending."""
    mock = MagicMock()

    def _rerank(query: str, docs: list, top_k: int) -> list:
        if len(scores) < len(docs):
            raise AssertionError(
                f"_make_reranker needs at least {len(docs)} scores, got {len(scores)}. "
                "Pass the right number of scores to the mock."
            )
        scored = [replace(d, score=s) for d, s in zip(docs, scores, strict=False)]
        return sorted(scored, key=lambda c: c.score, reverse=True)[:top_k]

    mock.rerank.side_effect = _rerank
    return mock


@pytest.mark.unit
class TestTriage:
    def _refiner(self, **kwargs):
        from src.retrieval.knowledge_refiner import KnowledgeRefiner

        return KnowledgeRefiner(_make_reranker([]), **kwargs)

    def test_high_score_accepted(self) -> None:
        r = self._refiner()
        chunk = make_chunk(score=1.0)
        accepted, uncertain, dropped = r._triage([chunk])
        assert chunk in accepted
        assert not uncertain
        assert not dropped

    def test_score_at_upper_threshold_accepted(self) -> None:
        r = self._refiner(upper_threshold=0.0)
        chunk = make_chunk(score=0.0)
        accepted, uncertain, dropped = r._triage([chunk])
        assert chunk in accepted

    def test_low_score_dropped(self) -> None:
        r = self._refiner()
        chunk = make_chunk(score=-5.0)
        accepted, uncertain, dropped = r._triage([chunk])
        assert not accepted
        assert not uncertain
        assert chunk in dropped

    def test_score_just_below_lower_dropped(self) -> None:
        r = self._refiner(lower_threshold=-3.0)
        chunk = make_chunk(score=-3.1)
        _, _, dropped = r._triage([chunk])
        assert chunk in dropped

    def test_uncertain_middle_band(self) -> None:
        r = self._refiner(upper_threshold=0.0, lower_threshold=-3.0)
        chunk = make_chunk(score=-1.5)
        accepted, uncertain, dropped = r._triage([chunk])
        assert not accepted
        assert not dropped
        assert len(uncertain) == 1

    def test_uncertain_chunk_marked_in_metadata(self) -> None:
        r = self._refiner(upper_threshold=0.0, lower_threshold=-3.0)
        chunk = make_chunk(score=-1.5)
        _, uncertain, _ = r._triage([chunk])
        assert uncertain[0].metadata["uncertain"] is True

    def test_uncertain_preserves_existing_metadata(self) -> None:
        r = self._refiner(upper_threshold=0.0, lower_threshold=-3.0)
        chunk = make_chunk(score=-1.5, metadata={"topics": ["species_info"]})
        _, uncertain, _ = r._triage([chunk])
        assert uncertain[0].metadata["topics"] == ["species_info"]
        assert uncertain[0].metadata["uncertain"] is True

    def test_score_at_lower_threshold_is_uncertain(self) -> None:
        r = self._refiner(upper_threshold=0.0, lower_threshold=-3.0)
        chunk = make_chunk(score=-3.0)
        accepted, uncertain, dropped = r._triage([chunk])
        assert not accepted
        assert not dropped
        assert len(uncertain) == 1

    def test_empty_list(self) -> None:
        r = self._refiner()
        accepted, uncertain, dropped = r._triage([])
        assert accepted == uncertain == dropped == []

    def test_invalid_thresholds_raise(self) -> None:
        from src.retrieval.knowledge_refiner import KnowledgeRefiner

        with pytest.raises(ValueError, match="lower_threshold"):
            KnowledgeRefiner(_make_reranker([]), lower_threshold=1.0, upper_threshold=0.0)


@pytest.mark.unit
class TestSplitSentences:
    def _split(self, text: str) -> list[str]:
        from src.retrieval.knowledge_refiner import KnowledgeRefiner

        return KnowledgeRefiner._split_sentences(text)

    def test_single_sentence(self) -> None:
        assert self._split("Hello world.") == ["Hello world."]

    def test_two_sentences(self) -> None:
        result = self._split("First sentence. Second sentence.")
        assert result == ["First sentence.", "Second sentence."]

    def test_question_mark_split(self) -> None:
        result = self._split("Is Pikachu electric? Yes it is.")
        assert result == ["Is Pikachu electric?", "Yes it is."]

    def test_exclamation_mark_split(self) -> None:
        result = self._split("Wow! Great move.")
        assert result == ["Wow!", "Great move."]

    def test_empty_string(self) -> None:
        assert self._split("") == []

    def test_strips_whitespace(self) -> None:
        result = self._split("  First.  Second.  ")
        assert result == ["First.", "Second."]


@pytest.mark.unit
class TestFilterStrips:
    def _refiner(self, scores: list[float], *, strip_threshold: float = -1.0) -> object:
        from src.retrieval.knowledge_refiner import KnowledgeRefiner

        return KnowledgeRefiner(
            _make_reranker(scores),
            strip_threshold=strip_threshold,
        )

    def test_single_sentence_passthrough(self) -> None:
        r = self._refiner(scores=[])
        chunk = make_chunk(text="Pikachu is an electric mouse.")
        result = r._filter_strips("query", chunk)
        assert result is chunk
        r._reranker.rerank.assert_not_called()

    def test_all_strips_pass_text_recomposed(self) -> None:
        r = self._refiner(scores=[0.5, 0.5], strip_threshold=-1.0)
        chunk = make_chunk(text="First sentence. Second sentence.")
        result = r._filter_strips("query", chunk)
        assert result is not None
        assert "First sentence." in result.text
        assert "Second sentence." in result.text

    def test_low_strip_dropped(self) -> None:
        r = self._refiner(scores=[0.5, -2.0], strip_threshold=-1.0)
        chunk = make_chunk(text="Good strip. Bad strip.")
        result = r._filter_strips("query", chunk)
        assert result is not None
        assert "Good strip." in result.text
        assert "Bad strip." not in result.text

    def test_all_strips_dropped_returns_none(self) -> None:
        r = self._refiner(scores=[-5.0, -5.0], strip_threshold=-1.0)
        chunk = make_chunk(text="Bad first. Bad second.")
        result = r._filter_strips("query", chunk)
        assert result is None

    def test_original_order_preserved(self) -> None:
        # strip 0 gets score -0.5 (passes), strip 1 gets score 0.8 (passes)
        # reranker returns [strip1, strip0] (sorted descending by score)
        # we must recompose in original order: strip0 first, strip1 second
        r = self._refiner(scores=[-0.5, 0.8], strip_threshold=-1.0)
        chunk = make_chunk(text="First strip. Second strip.")
        result = r._filter_strips("query", chunk)
        assert result is not None
        assert result.text.index("First strip.") < result.text.index("Second strip.")

    def test_non_text_fields_preserved(self) -> None:
        r = self._refiner(scores=[0.5, 0.5], strip_threshold=-1.0)
        chunk = make_chunk(
            text="Strip one. Strip two.", source="bulbapedia", entity_name="Charizard"
        )
        result = r._filter_strips("query", chunk)
        assert result is not None
        assert result.source == "bulbapedia"
        assert result.entity_name == "Charizard"


@pytest.mark.unit
class TestExtractConstraintKeywords:
    def _extract(self, query: str) -> list[str]:
        from src.retrieval.knowledge_refiner import KnowledgeRefiner

        return KnowledgeRefiner._extract_constraint_keywords(query)

    def test_gen_number_extracted(self) -> None:
        assert "gen9" in self._extract("Best Garchomp sets in gen9 OU")

    def test_gen_with_space_normalised(self) -> None:
        assert "gen9" in self._extract("Garchomp in gen 9")

    def test_tier_extracted(self) -> None:
        assert "ou" in self._extract("Gen 9 OU sets")

    def test_no_constraints_empty(self) -> None:
        assert self._extract("What is Pikachu's base speed?") == []

    def test_multiple_constraints(self) -> None:
        result = self._extract("gen9 OU Garchomp moveset")
        assert "gen9" in result
        assert "ou" in result

    def test_vgc_detected(self) -> None:
        assert "vgc" in self._extract("Best Miraidon in VGC 2024")


@pytest.mark.unit
class TestCheckSufficiency:
    def _check(self, query: str, texts: list[str]) -> list[str]:
        from src.retrieval.knowledge_refiner import KnowledgeRefiner

        chunks = [make_chunk(text=t) for t in texts]
        return KnowledgeRefiner._check_sufficiency(query, chunks)

    def test_keyword_present_not_in_gaps(self) -> None:
        gaps = self._check("gen9 OU Garchomp", ["gen9ou moveset for Garchomp"])
        assert "gen9" not in gaps

    def test_keyword_absent_in_gaps(self) -> None:
        gaps = self._check("gen9 OU Garchomp", ["gen6 moveset for Garchomp"])
        assert "gen9" in gaps

    def test_no_constraints_empty_gaps(self) -> None:
        gaps = self._check("What is Pikachu's speed?", ["Pikachu has 90 speed."])
        assert gaps == []

    def test_empty_chunks_all_gaps(self) -> None:
        gaps = self._check("gen9 OU Garchomp", [])
        assert "gen9" in gaps

    def test_tier_absent_is_gap(self) -> None:
        gaps = self._check("OU Garchomp sets", ["Gen 6 Garchomp sets"])
        assert "ou" in gaps


@pytest.mark.unit
class TestRefine:
    def _refiner(self, scores: list[float], **kwargs) -> object:
        from src.retrieval.knowledge_refiner import KnowledgeRefiner

        return KnowledgeRefiner(_make_reranker(scores), **kwargs)

    def test_empty_chunks_returns_empty_result(self) -> None:
        from src.retrieval.types import RefinementResult

        r = self._refiner([])
        result = r.refine("query", [])
        assert result == RefinementResult(chunks=(), gaps=())

    def test_accepted_chunks_included(self) -> None:
        # chunks have 1 sentence → passthrough strip filter
        chunk = make_chunk(text="Pikachu is electric.", score=1.0)
        r = self._refiner([])
        result = r.refine("query", [chunk])
        assert len(result.chunks) == 1

    def test_dropped_chunks_excluded(self) -> None:
        chunk = make_chunk(text="Some text.", score=-5.0)
        r = self._refiner([])
        result = r.refine("query", [chunk])
        assert len(result.chunks) == 0

    def test_uncertain_chunks_pass_through_without_strip_filter(self) -> None:
        chunk = make_chunk(text="First. Second.", score=-1.5)
        r = self._refiner([], upper_threshold=0.0, lower_threshold=-3.0)
        result = r.refine("query", [chunk])
        assert len(result.chunks) == 1
        assert result.chunks[0].metadata["uncertain"] is True
        r._reranker.rerank.assert_not_called()

    def test_gaps_returned_when_constraint_absent(self) -> None:
        chunk = make_chunk(text="Old format Garchomp set.", score=1.0)
        r = self._refiner([])
        result = r.refine("gen9 OU Garchomp sets", [chunk])
        assert "gen9" in result.gaps

    def test_no_gaps_when_constraint_present(self) -> None:
        chunk = make_chunk(text="gen9ou moveset for Garchomp.", score=1.0)
        r = self._refiner([])
        result = r.refine("gen9 OU Garchomp sets", [chunk])
        assert "gen9" not in result.gaps

    def test_all_dropped_by_triage_returns_empty_chunks(self) -> None:
        chunk = make_chunk(text="Some text.", score=-5.0)
        r = self._refiner([])
        result = r.refine("query", [chunk])
        assert result.chunks == ()

    def test_all_chunks_uncertain_all_pass_through(self) -> None:
        c1 = make_chunk(text="First.", score=-1.0)
        c2 = make_chunk(text="Second.", score=-2.0)
        r = self._refiner([], upper_threshold=0.0, lower_threshold=-3.0)
        result = r.refine("query", [c1, c2])
        assert len(result.chunks) == 2
        assert all(c.metadata["uncertain"] is True for c in result.chunks)
        r._reranker.rerank.assert_not_called()

    def test_mixed_batch_accepted_before_uncertain(self) -> None:
        accepted = make_chunk(text="Good.", score=1.0)
        uncertain = make_chunk(text="Maybe.", score=-1.5)
        dropped = make_chunk(text="Bad.", score=-5.0)
        r = self._refiner([], upper_threshold=0.0, lower_threshold=-3.0)
        result = r.refine("query", [accepted, uncertain, dropped])
        assert len(result.chunks) == 2
        texts = [c.text for c in result.chunks]
        assert "Good." in texts
        assert "Maybe." in texts
        assert "Bad." not in texts

    def test_result_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        from src.retrieval.types import RefinementResult

        result = RefinementResult(chunks=(), gaps=())
        with pytest.raises(FrozenInstanceError):
            result.chunks = ()  # type: ignore[misc]
