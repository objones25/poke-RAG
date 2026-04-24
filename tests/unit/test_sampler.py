from __future__ import annotations

from pathlib import Path

import pytest

from scripts.training.sampler import ChunkSampler, extract_entity_name


@pytest.mark.unit
class TestExtractEntityName:
    def test_pokeapi_is_a_pattern(self) -> None:
        assert extract_entity_name("Pikachu is a Pokémon species.", "pokeapi") == "Pikachu"

    def test_pokeapi_multiword(self) -> None:
        assert extract_entity_name("Master Ball is a Pokémon item.", "pokeapi") == "Master Ball"

    def test_smogon_paren_pattern(self) -> None:
        assert extract_entity_name("Garchomp (OU): runs Earthquake...", "smogon") == "Garchomp"

    def test_bulbapedia_paren_pattern(self) -> None:
        assert (
            extract_entity_name(
                "Charizard (Japanese: リザードン) is a Fire/Flying-type.", "bulbapedia"
            )
            == "Charizard"
        )

    def test_returns_none_for_no_match(self) -> None:
        assert (
            extract_entity_name("some random text with no patterns here at all", "pokeapi") is None
        )

    def test_ignores_separator_too_far_in(self) -> None:
        long_prefix = "A" * 61
        assert extract_entity_name(f"{long_prefix} is a Pokémon.", "pokeapi") is None


@pytest.mark.unit
class TestChunkSampler:
    def test_samples_from_all_sources(self, tmp_path: Path) -> None:
        for source in ("bulbapedia", "pokeapi", "smogon"):
            d = tmp_path / source
            d.mkdir()
            (d / "pokemon.txt").write_text(
                "\n".join(f"{source}_entity_{i} is a test." for i in range(50))
            )
        sampler = ChunkSampler(
            tmp_path, {"bulbapedia": 0.4, "pokeapi": 0.4, "smogon": 0.2}, seed=42
        )
        counts: dict[str, int] = {"bulbapedia": 0, "pokeapi": 0, "smogon": 0}
        for _ in range(120):
            result = sampler.sample()
            if result is not None:
                _, src = result
                counts[src] += 1
        assert all(c > 0 for c in counts.values())
        assert counts["smogon"] < counts["bulbapedia"]
        assert counts["smogon"] < counts["pokeapi"]

    def test_excludes_aug_files_by_default(self, tmp_path: Path) -> None:
        d = tmp_path / "pokeapi"
        d.mkdir()
        (d / "pokemon_species.txt").write_text("Pikachu is a real entity.\n")
        (d / "pokemon_species_aug.txt").write_text("AugLine is an augmented entity.\n")
        sampler = ChunkSampler(tmp_path, {"pokeapi": 1.0}, seed=42)
        sampled = [sampler.sample() for _ in range(5)]
        lines = [result[0] for result in sampled if result is not None]
        assert all("real entity" in line for line in lines)

    def test_returns_none_when_exhausted(self, tmp_path: Path) -> None:
        d = tmp_path / "pokeapi"
        d.mkdir()
        (d / "x.txt").write_text("OnlyOneLine is a singleton.\n")
        sampler = ChunkSampler(tmp_path, {"pokeapi": 1.0}, seed=0)
        assert sampler.sample() is not None
        assert sampler.sample() is None

    def test_zero_weight_all_sources_returns_none(self, tmp_path: Path) -> None:
        """RED: All-zero weights should return None, not raise ZeroDivisionError."""
        for source in ("bulbapedia", "pokeapi", "smogon"):
            d = tmp_path / source
            d.mkdir()
            (d / "pokemon.txt").write_text("SomeEntity is a test.\n")
        sampler = ChunkSampler(
            tmp_path, {"bulbapedia": 0.0, "pokeapi": 0.0, "smogon": 0.0}, seed=42
        )
        assert sampler.sample() is None

    def test_single_source_valid_weight(self, tmp_path: Path) -> None:
        """GREEN: Single source with valid weight samples correctly."""
        d = tmp_path / "pokeapi"
        d.mkdir()
        (d / "data.txt").write_text("Pikachu is a species.\nCharizard is a species.\n")
        sampler = ChunkSampler(tmp_path, {"pokeapi": 1.0}, seed=42)
        result = sampler.sample()
        assert result is not None
        line, source = result
        assert source == "pokeapi"
        assert "is a species" in line

    def test_normal_weights_distribution(self, tmp_path: Path) -> None:
        """GREEN: Normal positive weights distribute samples correctly."""
        for source in ("bulbapedia", "pokeapi", "smogon"):
            d = tmp_path / source
            d.mkdir()
            (d / "data.txt").write_text("\n".join(f"{source}_{i}" for i in range(100)))
        sampler = ChunkSampler(
            tmp_path, {"bulbapedia": 1.0, "pokeapi": 1.0, "smogon": 0.5}, seed=0
        )
        samples = [sampler.sample() for _ in range(50)]
        assert all(s is not None for s in samples)
        sources_seen = [s for _, s in samples]
        assert "bulbapedia" in sources_seen
        assert "pokeapi" in sources_seen
        assert "smogon" in sources_seen
