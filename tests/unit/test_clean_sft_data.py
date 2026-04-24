from __future__ import annotations

import pytest

from scripts.training.clean_sft_data import _normalize_messages


@pytest.mark.unit
class TestNormalizeMessages:
    """Test _normalize_messages type annotation and behavior."""

    def test_normalize_system_user_assistant(self) -> None:
        """GREEN: system, user, assistant structure returns user and assistant only."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Pikachu?"},
            {"role": "assistant", "content": "Pikachu is an Electric-type."},
        ]
        result = _normalize_messages(messages)
        assert result is not None
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_normalize_user_assistant_passthrough(self) -> None:
        """GREEN: user, assistant structure passes through unchanged."""
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "What is Pikachu?"},
            {"role": "assistant", "content": "Pikachu is an Electric-type."},
        ]
        result = _normalize_messages(messages)
        assert result == messages

    def test_normalize_unexpected_structure(self) -> None:
        """GREEN: Unexpected role structure returns None."""
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "Question"},
        ]
        result = _normalize_messages(messages)
        assert result is None

    def test_normalize_empty_list(self) -> None:
        """GREEN: Empty messages returns None."""
        messages: list[dict[str, str]] = []
        result = _normalize_messages(messages)
        assert result is None
