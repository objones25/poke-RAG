"""Unit tests for query string normalisation."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestParseQuery:
    def test_strips_leading_whitespace(self) -> None:
        from src.api.query_parser import parse_query

        assert parse_query("  Pikachu") == "Pikachu"

    def test_strips_trailing_whitespace(self) -> None:
        from src.api.query_parser import parse_query

        assert parse_query("Pikachu  ") == "Pikachu"

    def test_strips_both_ends(self) -> None:
        from src.api.query_parser import parse_query

        assert parse_query("  What type is Pikachu?  ") == "What type is Pikachu?"

    def test_raises_on_empty_string(self) -> None:
        from src.api.query_parser import parse_query

        with pytest.raises(ValueError, match="empty"):
            parse_query("")

    def test_raises_on_whitespace_only(self) -> None:
        from src.api.query_parser import parse_query

        with pytest.raises(ValueError, match="empty"):
            parse_query("   ")

    def test_preserves_internal_whitespace(self) -> None:
        from src.api.query_parser import parse_query

        assert parse_query("What type is Pikachu?") == "What type is Pikachu?"

    def test_returns_str_type(self) -> None:
        from src.api.query_parser import parse_query

        result = parse_query("Pikachu")
        assert isinstance(result, str)

    def test_tabs_and_newlines_stripped(self) -> None:
        from src.api.query_parser import parse_query

        assert parse_query("\tPikachu\n") == "Pikachu"
