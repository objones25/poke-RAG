"""Unit tests for src/retrieval/reranker.py — RED until reranker.py is implemented."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.reranker import BGEReranker
from src.types import RetrievedChunk
from tests.conftest import make_chunk


def _make_mock_reranker(scores: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.compute_score.return_value = scores
    return mock


@pytest.mark.unit
class TestBGERerankerRerank:
    def test_returns_list_of_retrieved_chunks(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.9, 0.4]))
        docs = [make_chunk(text="text a", chunk_index=0), make_chunk(text="text b", chunk_index=1)]
        results = reranker.rerank("query", docs, top_k=2)
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_returns_top_k_chunks(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.9, 0.5, 0.2]))
        docs = [make_chunk(chunk_index=i) for i in range(3)]
        results = reranker.rerank("query", docs, top_k=2)
        assert len(results) == 2

    def test_top_k_larger_than_docs_returns_all(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.8, 0.3]))
        docs = [make_chunk(chunk_index=i) for i in range(2)]
        results = reranker.rerank("query", docs, top_k=10)
        assert len(results) == 2

    def test_sorted_by_rerank_score_descending(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.3, 0.9, 0.6]))
        docs = [make_chunk(text=f"text {i}", chunk_index=i) for i in range(3)]
        results = reranker.rerank("query", docs, top_k=3)
        assert results[0].score >= results[1].score >= results[2].score

    def test_scores_updated_with_rerank_scores(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.7, 0.2]))
        docs = [
            make_chunk(text="a", score=0.0, chunk_index=0),
            make_chunk(text="b", score=0.0, chunk_index=1),
        ]
        results = reranker.rerank("query", docs, top_k=2)
        scores = sorted([r.score for r in results], reverse=True)
        assert scores[0] == pytest.approx(0.7)
        assert scores[1] == pytest.approx(0.2)

    def test_original_nonzero_scores_replaced_by_rerank_scores(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.4, 0.9]))
        docs = [
            make_chunk(text="a", score=0.95, chunk_index=0),
            make_chunk(text="b", score=0.80, chunk_index=1),
        ]
        results = reranker.rerank("query", docs, top_k=2)
        result_scores = sorted([r.score for r in results], reverse=True)
        assert result_scores[0] == pytest.approx(0.9)
        assert result_scores[1] == pytest.approx(0.4)

    def test_chunks_are_frozen(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.5]))
        results = reranker.rerank("query", [make_chunk()], top_k=1)
        with pytest.raises((AttributeError, TypeError)):
            results[0].score = 0.0  # type: ignore[misc]

    def test_passes_query_text_pairs_to_model(self) -> None:
        mock_model = _make_mock_reranker([0.5, 0.3])
        reranker = BGEReranker(mock_model)
        docs = [
            make_chunk(text="doc text a", chunk_index=0),
            make_chunk(text="doc text b", chunk_index=1),
        ]
        reranker.rerank("my query", docs, top_k=2)
        pairs = mock_model.compute_score.call_args[0][0]
        assert ["my query", "doc text a"] in pairs
        assert ["my query", "doc text b"] in pairs

    def test_passes_max_length_512_to_compute_score(self) -> None:
        mock_model = _make_mock_reranker([0.5])
        reranker = BGEReranker(mock_model)
        docs = [make_chunk(text="doc text", chunk_index=0)]
        reranker.rerank("my query", docs, top_k=1)
        call_kwargs = mock_model.compute_score.call_args[1]
        assert call_kwargs.get("max_length") == 512

    def test_uses_reranker_max_length_constant(self) -> None:
        """Verify that max_length is defined as a module constant, not hardcoded."""
        from src.retrieval import reranker as reranker_module

        assert hasattr(reranker_module, "_RERANKER_MAX_LENGTH")
        assert reranker_module._RERANKER_MAX_LENGTH == 512

    def test_empty_documents_returns_empty(self) -> None:
        mock_model = MagicMock()
        mock_model.compute_score.return_value = []
        reranker = BGEReranker(mock_model)
        results = reranker.rerank("query", [], top_k=5)
        assert results == []

    def test_text_preserved_in_results(self) -> None:
        reranker = BGEReranker(_make_mock_reranker([0.9, 0.1]))
        docs = [
            make_chunk(text="grass type facts", chunk_index=0),
            make_chunk(text="fire type info", chunk_index=1),
        ]
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

    def test_replaces_nan_scores_with_zero(self) -> None:
        """NaN scores from model should be replaced with 0.0."""
        reranker = BGEReranker(_make_mock_reranker([float("nan"), 0.9]))
        docs = [make_chunk(text="bad", chunk_index=0), make_chunk(text="good", chunk_index=1)]
        results = reranker.rerank("query", docs, top_k=2)
        assert len(results) == 2
        # The NaN should have been replaced with 0.0, so the 0.9 score should be top
        assert results[0].score == pytest.approx(0.9)
        assert results[1].score == pytest.approx(0.0)

    def test_replaces_inf_scores_with_zero(self) -> None:
        """Inf scores from model should be replaced with 0.0."""
        reranker = BGEReranker(_make_mock_reranker([float("inf"), 0.5]))
        docs = [make_chunk(text="bad", chunk_index=0), make_chunk(text="ok", chunk_index=1)]
        results = reranker.rerank("query", docs, top_k=2)
        assert len(results) == 2
        # The inf should have been replaced with 0.0, so the 0.5 score should be top
        assert results[0].score == pytest.approx(0.5)
        assert results[1].score == pytest.approx(0.0)

    def test_replaces_negative_inf_scores_with_zero(self) -> None:
        """Negative inf scores should also be replaced with 0.0."""
        reranker = BGEReranker(_make_mock_reranker([float("-inf"), 0.7]))
        docs = [make_chunk(text="bad", chunk_index=0), make_chunk(text="good", chunk_index=1)]
        results = reranker.rerank("query", docs, top_k=2)
        assert len(results) == 2
        assert results[0].score == pytest.approx(0.7)
        assert results[1].score == pytest.approx(0.0)


@pytest.mark.unit
class TestBGERerankerFromPretrained:
    def test_passes_model_name_to_flag_reranker(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.FlagReranker") as mock_model_class:
            BGEReranker.from_pretrained(model_name="BAAI/bge-reranker-v2-m3", device="cpu")
            mock_model_class.assert_called_once()
            assert mock_model_class.call_args[0][0] == "BAAI/bge-reranker-v2-m3"

    def test_passes_device_to_flag_reranker(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.FlagReranker") as mock_model_class:
            BGEReranker.from_pretrained(model_name="BAAI/bge-reranker-v2-m3", device="mps")
            call_kwargs = mock_model_class.call_args[1]
            assert call_kwargs["device"] == "mps"

    def test_use_fp16_true_for_cuda(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.FlagReranker") as mock_model_class:
            BGEReranker.from_pretrained(model_name="BAAI/bge-reranker-v2-m3", device="cuda")
            call_kwargs = mock_model_class.call_args[1]
            assert call_kwargs["use_fp16"] is True

    def test_use_fp16_true_for_mps(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.FlagReranker") as mock_model_class:
            BGEReranker.from_pretrained(model_name="BAAI/bge-reranker-v2-m3", device="mps")
            call_kwargs = mock_model_class.call_args[1]
            assert call_kwargs["use_fp16"] is True

    def test_use_fp16_false_for_cpu(self) -> None:
        from unittest.mock import patch

        with patch("FlagEmbedding.FlagReranker") as mock_model_class:
            BGEReranker.from_pretrained(model_name="BAAI/bge-reranker-v2-m3", device="cpu")
            call_kwargs = mock_model_class.call_args[1]
            assert call_kwargs["use_fp16"] is False


@pytest.mark.unit
class TestBGERerankerProtocolCompliance:
    def test_satisfies_reranker_protocol(self) -> None:
        from src.retrieval.protocols import RerankerProtocol

        reranker = BGEReranker(_make_mock_reranker([]))
        assert isinstance(reranker, RerankerProtocol)
