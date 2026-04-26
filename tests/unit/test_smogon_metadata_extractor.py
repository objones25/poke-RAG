"""Unit tests for _extract_smogon_metadata — RED until implemented in chunker.py."""

from __future__ import annotations

import pytest

from src.retrieval.chunker import _extract_smogon_metadata


@pytest.mark.unit
class TestExtractSmogonMetadata:
    def test_standard_ou_format(self) -> None:
        meta = _extract_smogon_metadata(format_name="gen9ou")
        assert meta["generation"] == 9
        assert meta["tier"] == "ou"

    def test_uu_format(self) -> None:
        meta = _extract_smogon_metadata(format_name="gen9uu")
        assert meta["generation"] == 9
        assert meta["tier"] == "uu"

    def test_nu_format(self) -> None:
        meta = _extract_smogon_metadata(format_name="gen8nu")
        assert meta["generation"] == 8
        assert meta["tier"] == "nu"

    def test_multiword_tier(self) -> None:
        meta = _extract_smogon_metadata(format_name="gen9vgc2024")
        assert meta["generation"] == 9
        assert meta["tier"] == "vgc2024"

    def test_ubers_format(self) -> None:
        meta = _extract_smogon_metadata(format_name="gen9ubers")
        assert meta["generation"] == 9
        assert meta["tier"] == "ubers"

    def test_gen6_format(self) -> None:
        meta = _extract_smogon_metadata(format_name="gen6ou")
        assert meta["generation"] == 6
        assert meta["tier"] == "ou"

    def test_format_name_stored(self) -> None:
        meta = _extract_smogon_metadata(format_name="gen9ou")
        assert meta["format_name"] == "gen9ou"

    def test_none_format_name_returns_empty(self) -> None:
        meta = _extract_smogon_metadata(format_name=None)
        assert meta == {}

    def test_overview_chunk_kind(self) -> None:
        meta = _extract_smogon_metadata(format_name="gen9ou", chunk_kind="overview")
        assert meta["chunk_kind"] == "overview"

    def test_set_chunk_kind(self) -> None:
        meta = _extract_smogon_metadata(
            format_name="gen9ou", chunk_kind="set", set_name="Swords Dance"
        )
        assert meta["chunk_kind"] == "set"
        assert meta["set_name"] == "Swords Dance"

    def test_set_attrs_stored(self) -> None:
        attrs = {
            "Tera Type": "Fire",
            "Item": "Leftovers",
            "Ability": "Rough Skin",
            "Nature": "Jolly",
        }
        meta = _extract_smogon_metadata(
            format_name="gen9ou", chunk_kind="set", set_name="SD", attrs=attrs
        )
        assert meta["tera_type"] == "Fire"
        assert meta["item"] == "Leftovers"
        assert meta["ability"] == "Rough Skin"
        assert meta["nature"] == "Jolly"

    def test_partial_attrs_stored(self) -> None:
        attrs = {"Item": "Choice Scarf"}
        meta = _extract_smogon_metadata(
            format_name="gen9ou", chunk_kind="set", set_name="Scarf", attrs=attrs
        )
        assert meta["item"] == "Choice Scarf"
        assert "tera_type" not in meta

    def test_no_set_name_when_overview(self) -> None:
        meta = _extract_smogon_metadata(format_name="gen9ou", chunk_kind="overview")
        assert "set_name" not in meta

    def test_unparseable_format_returns_name_only(self) -> None:
        meta = _extract_smogon_metadata(format_name="randombattle")
        assert meta["format_name"] == "randombattle"
        assert "generation" not in meta
        assert "tier" not in meta
