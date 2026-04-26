"""Unit tests for _extract_bulbapedia_metadata and chunk_bulbapedia_doc topic lookup."""

from __future__ import annotations

import pytest

from src.retrieval.chunker import _extract_bulbapedia_metadata, chunk_bulbapedia_doc


@pytest.mark.unit
class TestExtractBulbapediaMetadata:
    def test_none_entry_returns_empty(self) -> None:
        meta = _extract_bulbapedia_metadata(topic_entry=None)
        assert meta == {}

    def test_topics_stored(self) -> None:
        entry = {"topics": ["species_info", "lore"], "entity_type_hint": "pokemon"}
        meta = _extract_bulbapedia_metadata(topic_entry=entry)
        assert meta["topics"] == ["species_info", "lore"]

    def test_entity_type_hint_stored(self) -> None:
        entry = {"topics": ["ability_mechanics"], "entity_type_hint": "ability"}
        meta = _extract_bulbapedia_metadata(topic_entry=entry)
        assert meta["entity_type_hint"] == "ability"

    def test_null_entity_type_hint_excluded(self) -> None:
        entry = {"topics": ["lore"], "entity_type_hint": None}
        meta = _extract_bulbapedia_metadata(topic_entry=entry)
        assert "entity_type_hint" not in meta

    def test_empty_topics_stored(self) -> None:
        entry = {"topics": [], "entity_type_hint": None}
        meta = _extract_bulbapedia_metadata(topic_entry=entry)
        assert meta["topics"] == []

    def test_no_args_returns_empty(self) -> None:
        meta = _extract_bulbapedia_metadata()
        assert meta == {}


@pytest.mark.unit
class TestChunkBulbapediaDocTopicLookup:
    _DOC = "Title: Pikachu (Pokémon)\nPikachu is an Electric-type Pokémon."

    def test_topics_from_lookup(self) -> None:
        lookup = {"bulbapedia_0": {"topics": ["species_info"], "entity_type_hint": "pokemon"}}
        chunks = chunk_bulbapedia_doc(self._DOC, doc_id="bulbapedia_0", topic_lookup=lookup)
        assert chunks[0].metadata["topics"] == ["species_info"]

    def test_entity_type_hint_from_lookup(self) -> None:
        lookup = {"bulbapedia_0": {"topics": ["species_info"], "entity_type_hint": "pokemon"}}
        chunks = chunk_bulbapedia_doc(self._DOC, doc_id="bulbapedia_0", topic_lookup=lookup)
        assert chunks[0].metadata["entity_type_hint"] == "pokemon"

    def test_missing_doc_id_returns_empty_meta(self) -> None:
        lookup = {"other_doc": {"topics": ["lore"], "entity_type_hint": None}}
        chunks = chunk_bulbapedia_doc(self._DOC, doc_id="bulbapedia_0", topic_lookup=lookup)
        assert chunks[0].metadata == {}

    def test_no_lookup_returns_empty_meta(self) -> None:
        chunks = chunk_bulbapedia_doc(self._DOC, doc_id="bulbapedia_0")
        assert chunks[0].metadata == {}

    def test_all_subchunks_share_same_topics(self) -> None:
        body = " ".join(["Pikachu is a great electric mouse pokemon."] * 100)
        doc = f"Title: Pikachu (Pokémon)\n{body}"
        lookup = {"bulbapedia_0": {"topics": ["species_info"], "entity_type_hint": "pokemon"}}
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulbapedia_0", topic_lookup=lookup)
        assert len(chunks) > 1
        assert all(c.metadata["topics"] == ["species_info"] for c in chunks)
