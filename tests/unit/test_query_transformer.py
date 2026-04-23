"""Unit tests for src/retrieval/query_transformer.py — RED until implemented."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.query_transformer import HyDETransformer, PassthroughTransformer


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
        """HyDETransformer respects max_new_tokens from its config."""
        mock_inf = self._make_inferencer()
        t = HyDETransformer(mock_inf, max_new_tokens=50)
        t.transform("query")
        mock_inf.infer.assert_called_once()
