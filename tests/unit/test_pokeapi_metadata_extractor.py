"""Unit tests for _extract_pokeapi_metadata — RED until implemented in chunker.py."""

from __future__ import annotations

import pytest

from src.retrieval.chunker import _extract_pokeapi_metadata


@pytest.mark.unit
class TestExtractPokeapiMetadata:
    def test_pokemon_species(self) -> None:
        meta = _extract_pokeapi_metadata(doc_id="pokemon_species_42")
        assert meta["entity_subtype"] == "species"

    def test_pokemon_moves(self) -> None:
        meta = _extract_pokeapi_metadata(doc_id="pokemon_moves_5")
        assert meta["entity_subtype"] == "moves"

    def test_pokemon_encounters(self) -> None:
        meta = _extract_pokeapi_metadata(doc_id="pokemon_encounters_12")
        assert meta["entity_subtype"] == "encounters"

    def test_ability(self) -> None:
        meta = _extract_pokeapi_metadata(doc_id="ability_3")
        assert meta["entity_subtype"] == "ability"

    def test_item(self) -> None:
        meta = _extract_pokeapi_metadata(doc_id="item_7")
        assert meta["entity_subtype"] == "item"

    def test_move(self) -> None:
        meta = _extract_pokeapi_metadata(doc_id="move_42")
        assert meta["entity_subtype"] == "move"

    def test_index_zero(self) -> None:
        meta = _extract_pokeapi_metadata(doc_id="pokemon_species_0")
        assert meta["entity_subtype"] == "species"

    def test_large_index(self) -> None:
        meta = _extract_pokeapi_metadata(doc_id="ability_999")
        assert meta["entity_subtype"] == "ability"

    def test_unknown_stem_returns_empty(self) -> None:
        meta = _extract_pokeapi_metadata(doc_id="unknown_stem_5")
        assert meta == {}

    def test_doc_id_without_index_returns_empty(self) -> None:
        meta = _extract_pokeapi_metadata(doc_id="pokemon_species")
        assert meta == {}

    def test_aug_variant_pokemon_species(self) -> None:
        meta = _extract_pokeapi_metadata(doc_id="pokemon_species_aug_10")
        assert meta["entity_subtype"] == "species"
