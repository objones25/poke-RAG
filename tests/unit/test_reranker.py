"""Unit tests for src/retrieval/reranker.py — RED until reranker.py is implemented."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.reranker import BGEReranker
from src.types import RetrievedChunk


def _make_chunk(text: str = "some text", score: float = 0.5, idx: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        source="pokeapi",
        entity_name="Bulbasaur",
        entity_type="pokemon",
        chunk_index=idx,
        original_doc_id=f"doc_{idx}",
    )


def _make_mock_reranker(scores: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.compute_score.return_value = scores
    return mock


@pytest.mark.unit
class TestBGERerankerRerank:
    def test_returns_list_of_retrieved_chunks(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.9, 0.4]))
        docs = [_make_chunk("text a", idx=0), _make_chunk("text b", idx=1)]
        results = reranker.rerank("query", docs, top_k=2)
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_returns_top_k_chunks(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.9, 0.5, 0.2]))
        docs = [_make_chunk(idx=i) for i in range(3)]
        results = reranker.rerank("query", docs, top_k=2)
        assert len(results) == 2

    def test_top_k_larger_than_docs_returns_all(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.8, 0.3]))
        docs = [_make_chunk(idx=i) for i in range(2)]
        results = reranker.rerank("query", docs, top_k=10)
        assert len(results) == 2

    def test_sorted_by_rerank_score_descending(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.3, 0.9, 0.6]))
        docs = [_make_chunk(f"text {i}", idx=i) for i in range(3)]
        results = reranker.rerank("query", docs, top_k=3)
        assert results[0].score >= results[1].score >= results[2].score

    def test_scores_updated_with_rerank_scores(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.7, 0.2]))
        docs = [_make_chunk("a", score=0.0, idx=0), _make_chunk("b", score=0.0, idx=1)]
        results = reranker.rerank("query", docs, top_k=2)
        scores = sorted([r.score for r in results], reverse=True)
        assert scores[0] == pytest.approx(0.7)
        assert scores[1] == pytest.approx(0.2)

    def test_original_nonzero_scores_replaced_by_rerank_scores(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.4, 0.9]))
        docs = [_make_chunk("a", score=0.95, idx=0), _make_chunk("b", score=0.80, idx=1)]
        results = reranker.rerank("query", docs, top_k=2)
        result_scores = sorted([r.score for r in results], reverse=True)
        assert result_scores[0] == pytest.approx(0.9)
        assert result_scores[1] == pytest.approx(0.4)

    def test_chunks_are_frozen(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.5]))
        results = reranker.rerank("query", [_make_chunk()], top_k=1)
        with pytest.raises((AttributeError, TypeError)):
            results[0].score = 0.0  # type: ignore[misc]

    def test_passes_query_text_pairs_to_model(self) -> None:
        mock_model = _make_mock_reranker([0.5, 0.3])
        reranker = BGEReranker(mock_model)
        docs = [_make_chunk("doc text a", idx=0), _make_chunk("doc text b", idx=1)]
        reranker.rerank("my query", docs, top_k=2)
        pairs = mock_model.compute_score.call_args[0][0]
        assert ["my query", "doc text a"] in pairs
        assert ["my query", "doc text b"] in pairs

    def test_empty_documents_returns_empty(self) -> None:
        mock_model = MagicMock()
        mock_model.compute_score.return_value = []
        reranker = BGEReranker(mock_model)
        results = reranker.rerank("query", [], top_k=5)
        assert results == []

    def test_text_preserved_in_results(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.9, 0.1]))
        docs = [_make_chunk("grass type facts", idx=0), _make_chunk("fire type info", idx=1)]
        results = reranker.rerank("query", docs, top_k=2)
        texts = {r.text for r in results}
        assert "grass type facts" in texts
        assert "fire type info" in texts

    def test_metadata_preserved_in_results(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.8]))
        chunk = RetrievedChunk(
            text="some text",
            score=0.0,
            source="smogon",
            entity_name="Gyarados",
            entity_type="pokemon",
            chunk_index=3,
            original_doc_id="smogon_42",
        )
        results = reranker.rerank("query", [chunk], top_k=1)
        assert results[0].source == "smogon"
        assert results[0].entity_name == "Gyarados"
        assert results[0].chunk_index == 3
        assert results[0].original_doc_id == "smogon_42"


@pytest.mark.unit
class TestBGERerankerProtocolCompliance:
    def test_satisfies_reranker_protocol(self) -> None:
        from src.retrieval.protocols import RerankerProtocol

        reranker = BGEReranker(_make_mock_reranker([]))
        assert isinstance(reranker, RerankerProtocol)
