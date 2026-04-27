"""Unit tests for API models and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.models import QueryRequest, QueryResponse


class TestQueryRequestValidation:
    """Test QueryRequest field validation."""

    def test_valid_entity_name_accepted(self) -> None:
        req = QueryRequest(query="What type is Pikachu?", entity_name="Pikachu")
        assert req.entity_name == "Pikachu"

    def test_entity_name_with_hyphen_accepted(self) -> None:
        req = QueryRequest(query="test", entity_name="Ho-Oh")
        assert req.entity_name == "Ho-Oh"

    def test_entity_name_with_apostrophe_accepted(self) -> None:
        req = QueryRequest(query="test", entity_name="Farfetch'd")
        assert req.entity_name == "Farfetch'd"

    def test_entity_name_with_underscore_accepted(self) -> None:
        req = QueryRequest(query="test", entity_name="Nidoran_M")
        assert req.entity_name == "Nidoran_M"

    def test_entity_name_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="test", entity_name="A" * 51)

    def test_entity_name_with_newline_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="test", entity_name="pikachu\nmalicious")

    def test_entity_name_with_angle_bracket_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="test", entity_name="<script>")

    def test_entity_name_with_semicolon_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="test", entity_name="pikachu; DROP TABLE")

    def test_entity_name_with_quotes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="test", entity_name='pikachu"test')

    def test_entity_name_with_equals_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="test", entity_name="x=1")

    def test_entity_name_none_accepted(self) -> None:
        req = QueryRequest(query="test")
        assert req.entity_name is None

    def test_query_min_length_enforced(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_query_max_length_enforced(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 501)

    def test_query_at_max_length_accepted(self) -> None:
        req = QueryRequest(query="x" * 500)
        assert len(req.query) == 500

    def test_query_required_field(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest()  # type: ignore[call-arg]

    def test_valid_sources_accepted(self) -> None:
        req = QueryRequest(query="test", sources=["bulbapedia", "pokeapi"])
        assert req.sources == ["bulbapedia", "pokeapi"]

    def test_sources_empty_list_accepted(self) -> None:
        req = QueryRequest(query="test", sources=[])
        assert req.sources == []

    def test_sources_none_accepted(self) -> None:
        req = QueryRequest(query="test", sources=None)
        assert req.sources is None


class TestEntityNameNFKCNormalization:
    """S7: entity_name must be NFKC-normalized before regex validation."""

    def test_fullwidth_chars_normalized_and_accepted(self) -> None:
        # U+FF30…U+FF49 are fullwidth Latin letters; NFKC folds them to ASCII
        req = QueryRequest(query="test", entity_name="Ｐikachu")  # 'Ｐikachu'
        assert req.entity_name == "Pikachu"

    def test_validator_returns_normalized_form(self) -> None:
        # Confirm the stored value is the normalized string, not the original
        req = QueryRequest(query="test", entity_name="０")  # FULLWIDTH DIGIT ZERO → '0'
        assert req.entity_name == "0"

    def test_cyrillic_lookalike_rejected(self) -> None:
        # Cyrillic 'а' (U+0430) does not normalize to ASCII 'a' — must be rejected
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QueryRequest(query="test", entity_name="Pіkachu")  # 'і' is Cyrillic


class TestQueryResponseValidation:
    """Test QueryResponse field validation."""

    def test_query_response_all_fields(self) -> None:
        resp = QueryResponse(
            answer="Pikachu is Electric-type.",
            sources_used=["pokeapi"],
            num_chunks_used=3,
            model_name="google/gemma-4-E4B-it",
            query="What type is Pikachu?",
            confidence_score=0.95,
        )
        assert resp.answer == "Pikachu is Electric-type."
        assert resp.sources_used == ["pokeapi"]
        assert resp.num_chunks_used == 3
        assert resp.model_name == "google/gemma-4-E4B-it"
        assert resp.query == "What type is Pikachu?"
        assert resp.confidence_score == 0.95

    def test_query_response_confidence_score_value(self) -> None:
        resp = QueryResponse(
            answer="Pikachu is Electric-type.",
            sources_used=["pokeapi"],
            num_chunks_used=3,
            model_name="google/gemma-4-E4B-it",
            query="What type is Pikachu?",
            confidence_score=0.88,
        )
        assert resp.confidence_score == 0.88
