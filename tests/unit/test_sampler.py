from __future__ import annotations

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
    def test_samples_from_all_sources(self, tmp_path: pytest.TempPathFactory) -> None:
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

    def test_excludes_aug_files_by_default(self, tmp_path: pytest.TempPathFactory) -> None:
        d = tmp_path / "pokeapi"
        d.mkdir()
        (d / "pokemon_species.txt").write_text("Pikachu is a real entity.\n")
        (d / "pokemon_species_aug.txt").write_text("AugLine is an augmented entity.\n")
        sampler = ChunkSampler(tmp_path, {"pokeapi": 1.0}, seed=42)
        sampled = [sampler.sample() for _ in range(5)]
        lines = [result[0] for result in sampled if result is not None]
        assert all("real entity" in line for line in lines)

    def test_returns_none_when_exhausted(self, tmp_path: pytest.TempPathFactory) -> None:
        d = tmp_path / "pokeapi"
        d.mkdir()
        (d / "x.txt").write_text("OnlyOneLine is a singleton.\n")
        sampler = ChunkSampler(tmp_path, {"pokeapi": 1.0}, seed=0)
        assert sampler.sample() is not None
        assert sampler.sample() is None
