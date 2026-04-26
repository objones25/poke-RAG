"""Unit tests verifying chunk functions populate .metadata — RED until chunker.py is updated."""

from __future__ import annotations

import pytest

from src.retrieval.chunker import (
    chunk_bulbapedia_doc,
    chunk_pokeapi_line,
    chunk_smogon_data_file,
    chunk_smogon_line,
)

_SMOGON_DATA_SAMPLE = """\
================================================================================
Venusaur
Smogon form: Venusaur
================================================================================

----------------------------------------
 Format: gen9ou
----------------------------------------

[ Overview ]
Venusaur is a solid defensive pivot in Gen 9 OU. It has good typing and recovery.

[ Set: Swords Dance ]
  Tera Type: Fire
  Item: Life Orb
  Ability: Chlorophyll
  Nature: Jolly
  Moves:
    - Swords Dance
    - Leaf Blade
  Description:
  A setup sweeper set that leverages Swords Dance.
"""


@pytest.mark.unit
class TestChunkPokeapiMetadata:
    def test_metadata_not_none(self) -> None:
        chunks = chunk_pokeapi_line("Bulbasaur is a Seed Pokémon.", doc_id="pokemon_species_0")
        assert chunks[0].metadata is not None

    def test_entity_subtype_present(self) -> None:
        chunks = chunk_pokeapi_line("Bulbasaur is a Seed Pokémon.", doc_id="pokemon_species_0")
        assert chunks[0].metadata["entity_subtype"] == "species"

    def test_moves_subtype(self) -> None:
        chunks = chunk_pokeapi_line("Bulbasaur learns Tackle.", doc_id="pokemon_moves_0")
        assert chunks[0].metadata["entity_subtype"] == "moves"

    def test_ability_subtype(self) -> None:
        chunks = chunk_pokeapi_line("Overgrow is an ability.", doc_id="ability_0")
        assert chunks[0].metadata["entity_subtype"] == "ability"

    def test_unknown_doc_id_metadata_empty_dict(self) -> None:
        chunks = chunk_pokeapi_line("Something.", doc_id="unknown_0")
        assert chunks[0].metadata == {}


@pytest.mark.unit
class TestChunkSmogonLineMetadata:
    def test_metadata_not_none(self) -> None:
        chunks = chunk_smogon_line("Garchomp (OU): Garchomp is a fast attacker.", doc_id="smogon_0")
        assert chunks[0].metadata is not None

    def test_metadata_is_dict(self) -> None:
        chunks = chunk_smogon_line("Garchomp (OU): Garchomp is a fast attacker.", doc_id="smogon_0")
        assert isinstance(chunks[0].metadata, dict)


@pytest.mark.unit
class TestChunkBulbapediaMetadata:
    def test_metadata_not_none(self) -> None:
        doc = "Title: Pikachu (Pokémon)\nPikachu is an Electric-type Pokémon."
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulbapedia_0")
        assert chunks[0].metadata is not None

    def test_metadata_is_dict(self) -> None:
        doc = "Title: Pikachu (Pokémon)\nPikachu is an Electric-type Pokémon."
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulbapedia_0")
        assert isinstance(chunks[0].metadata, dict)


@pytest.mark.unit
class TestChunkSmogonDataFileMetadata:
    def test_overview_chunk_has_metadata(self) -> None:
        chunks = chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)
        overview = [c for c in chunks if "Overview" in c.text]
        assert overview, "expected at least one overview chunk"
        assert overview[0].metadata is not None

    def test_overview_chunk_kind(self) -> None:
        chunks = chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)
        overview = [c for c in chunks if "Overview" in c.text]
        assert overview[0].metadata["chunk_kind"] == "overview"

    def test_overview_format_name(self) -> None:
        chunks = chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)
        overview = [c for c in chunks if "Overview" in c.text]
        assert overview[0].metadata["format_name"] == "gen9ou"

    def test_overview_generation(self) -> None:
        chunks = chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)
        overview = [c for c in chunks if "Overview" in c.text]
        assert overview[0].metadata["generation"] == 9

    def test_overview_tier(self) -> None:
        chunks = chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)
        overview = [c for c in chunks if "Overview" in c.text]
        assert overview[0].metadata["tier"] == "ou"

    def test_set_chunk_kind(self) -> None:
        chunks = chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)
        sets = [c for c in chunks if "Swords Dance" in c.text]
        assert sets, "expected at least one set chunk"
        assert sets[0].metadata["chunk_kind"] == "set"

    def test_set_name(self) -> None:
        chunks = chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)
        sets = [c for c in chunks if "Swords Dance" in c.text]
        assert sets[0].metadata["set_name"] == "Swords Dance"

    def test_set_item(self) -> None:
        chunks = chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)
        sets = [c for c in chunks if "Swords Dance" in c.text]
        assert sets[0].metadata["item"] == "Life Orb"

    def test_set_ability(self) -> None:
        chunks = chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)
        sets = [c for c in chunks if "Swords Dance" in c.text]
        assert sets[0].metadata["ability"] == "Chlorophyll"

    def test_set_tera_type(self) -> None:
        chunks = chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)
        sets = [c for c in chunks if "Swords Dance" in c.text]
        assert sets[0].metadata["tera_type"] == "Fire"

    def test_set_nature(self) -> None:
        chunks = chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)
        sets = [c for c in chunks if "Swords Dance" in c.text]
        assert sets[0].metadata["nature"] == "Jolly"
