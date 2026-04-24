"""Unit tests for src/retrieval/query_transformer.py — RED until implemented."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.query_transformer import (
    HyDETransformer,
    MultiDraftHyDETransformer,
    PassthroughTransformer,
)
from src.retrieval.types import EmbeddingOutput


@pytest.mark.unit
class TestPassthroughTransformer:
    def test_returns_query_unchanged(self) -> None:
        t = PassthroughTransformer()
        assert t.transform("what type is Bulbasaur?") == "what type is Bulbasaur?"

    def test_returns_empty_string_unchanged(self) -> None:
        t = PassthroughTransformer()
        assert t.transform("") == ""

    def test_satisfies_protocol(self) -> None:
        from src.retrieval.protocols import QueryTransformerProtocol

        assert isinstance(PassthroughTransformer(), QueryTransformerProtocol)


@pytest.mark.unit
class TestHyDETransformer:
    def _make_inferencer(self, return_value: str = "A hypothetical doc") -> MagicMock:
        mock = MagicMock()
        mock.infer.return_value = return_value
        return mock

    def test_returns_string(self) -> None:
        t = HyDETransformer(self._make_inferencer())
        result = t.transform("what type is Bulbasaur?")
        assert isinstance(result, str)

    def test_calls_inferencer_infer(self) -> None:
        mock_inf = self._make_inferencer()
        HyDETransformer(mock_inf).transform("any query")
        mock_inf.infer.assert_called_once()

    def test_prompt_includes_query(self) -> None:
        mock_inf = self._make_inferencer()
        HyDETransformer(mock_inf).transform("what is Charizard's type?")
        prompt_passed = mock_inf.infer.call_args[0][0]
        assert "what is Charizard's type?" in prompt_passed

    def test_returns_inferencer_output(self) -> None:
        mock_inf = self._make_inferencer("Charizard is a Fire/Flying type Pokémon.")
        result = HyDETransformer(mock_inf).transform("what is Charizard's type?")
        assert result == "Charizard is a Fire/Flying type Pokémon."

    def test_returns_original_query_on_inference_error(self) -> None:
        mock_inf = MagicMock()
        mock_inf.infer.side_effect = RuntimeError("model exploded")
        result = HyDETransformer(mock_inf).transform("original query")
        assert result == "original query"

    def test_returns_original_query_on_value_error(self) -> None:
        mock_inf = MagicMock()
        mock_inf.infer.side_effect = ValueError("empty prompt")
        result = HyDETransformer(mock_inf).transform("original query")
        assert result == "original query"

    def test_returns_original_query_on_any_exception(self) -> None:
        mock_inf = MagicMock()
        mock_inf.infer.side_effect = Exception("unexpected failure")
        result = HyDETransformer(mock_inf).transform("original query")
        assert result == "original query"

    def test_returns_original_on_empty_output(self) -> None:
        result = HyDETransformer(self._make_inferencer("")).transform("my query")
        assert result == "my query"

    def test_returns_original_on_whitespace_only_output(self) -> None:
        result = HyDETransformer(self._make_inferencer("   \n  ")).transform("my query")
        assert result == "my query"

    def test_satisfies_protocol(self) -> None:
        from src.retrieval.protocols import QueryTransformerProtocol

        assert isinstance(HyDETransformer(self._make_inferencer()), QueryTransformerProtocol)

    def test_max_tokens_passed_to_infer_prompt_config(self) -> None:
        """HyDETransformer forwards its max_new_tokens to infer()."""
        mock_inf = self._make_inferencer()
        t = HyDETransformer(mock_inf, max_new_tokens=50)
        t.transform("query")
        mock_inf.infer.assert_called_once()
        _, kwargs = mock_inf.infer.call_args
        assert kwargs.get("max_new_tokens") == 50

    def test_exception_logs_traceback_on_inference_error(self, caplog) -> None:
        """HyDE error logging should include traceback (exc_info=True)."""
        import logging

        caplog.set_level(logging.WARNING)
        mock_inf = MagicMock()
        mock_inf.infer.side_effect = RuntimeError("model crashed: cuda OOM")
        t = HyDETransformer(mock_inf)
        result = t.transform("query")
        assert result == "query"
        assert len(caplog.records) > 0
        warning_record = caplog.records[0]
        # exc_info=True means the record should have an exception attached
        assert warning_record.exc_info is not None


@pytest.mark.unit
class TestMultiDraftHyDETransformer:
    def _make_inferencer(self, return_value: str = "A hypothetical doc") -> MagicMock:
        mock = MagicMock()
        mock.infer.return_value = return_value
        return mock

    def _make_embedder(self, dense_dim: int = 4) -> MagicMock:
        mock = MagicMock()
        mock.encode.return_value = EmbeddingOutput(
            dense=[[0.1] * dense_dim],
            sparse=[{1: 0.5}],
        )
        return mock

    def test_transform_returns_string(self) -> None:
        t = MultiDraftHyDETransformer(self._make_inferencer(), self._make_embedder(), num_drafts=2)
        result = t.transform("what type is Bulbasaur?")
        assert isinstance(result, str)

    def test_transform_to_embedding_calls_inferencer_num_drafts_times(self) -> None:
        mock_inf = self._make_inferencer("draft text")
        embedder = MagicMock()
        embedder.encode.return_value = EmbeddingOutput(
            dense=[[0.1] * 4, [0.1] * 4, [0.1] * 4], sparse=[{}, {}, {}]
        )
        t = MultiDraftHyDETransformer(mock_inf, embedder, num_drafts=3)
        t.transform_to_embedding("query")
        assert mock_inf.infer.call_count == 3

    def test_transform_to_embedding_returns_embedding_output(self) -> None:
        t = MultiDraftHyDETransformer(self._make_inferencer(), self._make_embedder(), num_drafts=1)
        result = t.transform_to_embedding("query")
        assert isinstance(result, EmbeddingOutput)

    def test_transform_to_embedding_returns_single_dense_vector(self) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = EmbeddingOutput(
            dense=[[0.1] * 4, [0.2] * 4, [0.3] * 4], sparse=[{}, {}, {}]
        )
        t = MultiDraftHyDETransformer(self._make_inferencer(), embedder, num_drafts=3)
        result = t.transform_to_embedding("query")
        assert len(result.dense) == 1

    def test_transform_to_embedding_returns_single_sparse_dict(self) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = EmbeddingOutput(
            dense=[[0.1] * 4, [0.2] * 4], sparse=[{1: 0.3}, {2: 0.4}]
        )
        t = MultiDraftHyDETransformer(self._make_inferencer(), embedder, num_drafts=2)
        result = t.transform_to_embedding("query")
        assert len(result.sparse) == 1

    def test_dense_vector_is_element_wise_mean(self) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = EmbeddingOutput(
            dense=[[0.2, 0.4], [0.6, 0.8]], sparse=[{}, {}]
        )
        t = MultiDraftHyDETransformer(self._make_inferencer("d"), embedder, num_drafts=2)
        result = t.transform_to_embedding("query")
        assert abs(result.dense[0][0] - 0.4) < 1e-6
        assert abs(result.dense[0][1] - 0.6) < 1e-6

    def test_sparse_uses_max_weight_per_token(self) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = EmbeddingOutput(
            dense=[[0.1], [0.1]],
            sparse=[{1: 0.3, 2: 0.8}, {1: 0.9, 3: 0.5}],
        )
        t = MultiDraftHyDETransformer(self._make_inferencer("d"), embedder, num_drafts=2)
        result = t.transform_to_embedding("query")
        assert result.sparse[0] == {1: 0.9, 2: 0.8, 3: 0.5}

    def test_all_drafts_fail_falls_back_to_raw_embedding(self) -> None:
        mock_inf = MagicMock()
        mock_inf.infer.side_effect = RuntimeError("model failed")
        embedder = self._make_embedder()
        t = MultiDraftHyDETransformer(mock_inf, embedder, num_drafts=3)
        result = t.transform_to_embedding("my query")
        embedder.encode.assert_called_once_with(["my query"])
        assert isinstance(result, EmbeddingOutput)

    def test_partial_failure_averages_successful_drafts_only(self) -> None:
        call_count = 0

        def infer(prompt: str, **kwargs: object) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("draft 2 failed")
            return "draft text"

        mock_inf = MagicMock()
        mock_inf.infer.side_effect = infer

        embedder = MagicMock()
        embedder.encode.return_value = EmbeddingOutput(
            dense=[[0.5, 0.5], [0.5, 0.5]], sparse=[{1: 0.4}, {1: 0.4}]
        )
        t = MultiDraftHyDETransformer(mock_inf, embedder, num_drafts=3)
        result = t.transform_to_embedding("query")
        encode_arg = embedder.encode.call_args[0][0]
        assert len(encode_arg) == 2
        assert len(result.dense) == 1

    def test_zero_drafts_falls_back_to_raw_query(self) -> None:
        embedder = self._make_embedder()
        transformer = MultiDraftHyDETransformer(self._make_inferencer(), embedder, num_drafts=0)
        result = transformer.transform_to_embedding("how does Intimidate work?")
        assert embedder.encode.called
        assert len(result.dense) == 1
        assert len(result.sparse) == 1

    def test_satisfies_protocol(self) -> None:
        from src.retrieval.protocols import QueryTransformerProtocol

        t = MultiDraftHyDETransformer(self._make_inferencer(), self._make_embedder(), num_drafts=2)
        assert isinstance(t, QueryTransformerProtocol)
