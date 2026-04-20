"""Unit tests for src/retrieval/chunker.py — RED until chunker.py is implemented."""

from __future__ import annotations

import pytest

from src.retrieval.chunker import (
    chunk_bulbapedia_doc,
    chunk_file,
    chunk_pokeapi_line,
    chunk_smogon_line,
)


@pytest.mark.unit
class TestChunkPokeapiLine:
    def test_single_line_becomes_one_chunk(self) -> None:
        line = "Bulbasaur is a Seed Pokémon. Type: Grass/Poison. Base stats: HP 45."
        chunks = chunk_pokeapi_line(line, doc_id="pokeapi_0")
        assert len(chunks) == 1

    def test_text_preserved(self) -> None:
        line = "Bulbasaur is a Seed Pokémon. Type: Grass/Poison."
        chunks = chunk_pokeapi_line(line, doc_id="pokeapi_0")
        assert chunks[0].text == line

    def test_source_is_pokeapi(self) -> None:
        chunks = chunk_pokeapi_line("Bulbasaur is a Seed Pokémon.", doc_id="pokeapi_0")
        assert chunks[0].source == "pokeapi"

    def test_entity_name_extracted(self) -> None:
        chunks = chunk_pokeapi_line("Bulbasaur is a Seed Pokémon.", doc_id="pokeapi_0")
        assert chunks[0].entity_name == "Bulbasaur"

    def test_hyphenated_name_extracted(self) -> None:
        chunks = chunk_pokeapi_line("Ho-Oh is a Rainbow Pokémon.", doc_id="pokeapi_1")
        assert chunks[0].entity_name == "Ho-Oh"

    def test_score_is_zero(self) -> None:
        chunks = chunk_pokeapi_line("Bulbasaur is a Seed Pokémon.", doc_id="pokeapi_0")
        assert chunks[0].score == 0.0

    def test_chunk_index_is_zero(self) -> None:
        chunks = chunk_pokeapi_line("Bulbasaur is a Seed Pokémon.", doc_id="pokeapi_0")
        assert chunks[0].chunk_index == 0

    def test_doc_id_stored(self) -> None:
        chunks = chunk_pokeapi_line("Bulbasaur is a Seed Pokémon.", doc_id="my_doc")
        assert chunks[0].original_doc_id == "my_doc"

    def test_empty_line_returns_empty(self) -> None:
        assert chunk_pokeapi_line("", doc_id="pokeapi_0") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert chunk_pokeapi_line("   \t  ", doc_id="pokeapi_0") == []

    def test_leading_trailing_whitespace_stripped(self) -> None:
        line = "  Bulbasaur is a Seed Pokémon.  "
        chunks = chunk_pokeapi_line(line, doc_id="pokeapi_0")
        assert chunks[0].text == line.strip()


@pytest.mark.unit
class TestChunkSmogonLine:
    def test_short_entry_is_one_chunk(self) -> None:
        line = "Pikachu (OU): Pikachu is a fast attacker."
        chunks = chunk_smogon_line(line, doc_id="smogon_0")
        assert len(chunks) == 1

    def test_source_is_smogon(self) -> None:
        line = "Garbodor (NU): Garbodor is a spiker."
        chunks = chunk_smogon_line(line, doc_id="smogon_0")
        assert all(c.source == "smogon" for c in chunks)

    def test_entity_name_extracted(self) -> None:
        line = "Garbodor (NU): Garbodor is a consistent Spiker."
        chunks = chunk_smogon_line(line, doc_id="smogon_0")
        assert chunks[0].entity_name == "Garbodor"

    def test_score_is_zero(self) -> None:
        chunks = chunk_smogon_line("Pikachu (OU): Fast attacker.", doc_id="smogon_0")
        assert all(c.score == 0.0 for c in chunks)

    def test_doc_id_stored(self) -> None:
        chunks = chunk_smogon_line("Pikachu (OU): Fast attacker.", doc_id="smogon_99")
        assert all(c.original_doc_id == "smogon_99" for c in chunks)

    def test_chunk_indices_sequential(self) -> None:
        sentences = [f"Sentence {i} with extra content for padding." for i in range(60)]
        line = "Gyarados (OU): " + " ".join(sentences)
        chunks = chunk_smogon_line(line, doc_id="smogon_0")
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_long_entry_splits_into_multiple_chunks(self) -> None:
        sentences = [f"Sentence number {i} with plenty of content here." for i in range(60)]
        line = "Garbodor (NU): " + " ".join(sentences)
        chunks = chunk_smogon_line(line, doc_id="smogon_0")
        assert len(chunks) > 1

    def test_all_text_covered_across_chunks(self) -> None:
        line = (
            "Garbodor (NU): Garbodor is a great Spiker. It has Aftermath. Rocky Helmet is useful."
        )
        chunks = chunk_smogon_line(line, doc_id="smogon_0")
        combined = " ".join(c.text for c in chunks)
        assert "Aftermath" in combined
        assert "Rocky Helmet" in combined

    def test_empty_line_returns_empty(self) -> None:
        assert chunk_smogon_line("", doc_id="smogon_0") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert chunk_smogon_line("   ", doc_id="smogon_0") == []

    def test_entity_name_none_when_unparseable(self) -> None:
        line = "just some text without a name pattern"
        chunks = chunk_smogon_line(line, doc_id="smogon_0")
        # Should not crash; entity_name may be None
        assert chunks[0].entity_name is None


@pytest.mark.unit
class TestChunkBulbapediaDoc:
    def test_entity_name_extracted_from_title(self) -> None:
        doc = "Title: Abomasnow (Pokémon)\n\nAbomasnow is a large, bipedal Pokémon."
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulba_0")
        assert chunks[0].entity_name == "Abomasnow"

    def test_source_is_bulbapedia(self) -> None:
        doc = "Title: Pikachu (Pokémon)\n\nPikachu is electric."
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulba_0")
        assert all(c.source == "bulbapedia" for c in chunks)

    def test_score_is_zero(self) -> None:
        doc = "Title: Pikachu (Pokémon)\n\nPikachu is electric."
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulba_0")
        assert all(c.score == 0.0 for c in chunks)

    def test_doc_id_stored(self) -> None:
        doc = "Title: Pikachu (Pokémon)\n\nPikachu is electric."
        chunks = chunk_bulbapedia_doc(doc, doc_id="my_doc_123")
        assert all(c.original_doc_id == "my_doc_123" for c in chunks)

    def test_short_doc_is_one_chunk(self) -> None:
        doc = "Title: Pikachu (Pokémon)\n\nPikachu is an Electric-type Pokémon."
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulba_0")
        assert len(chunks) == 1

    def test_body_text_in_chunk(self) -> None:
        doc = "Title: Pikachu (Pokémon)\n\nPikachu is an Electric-type Pokémon."
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulba_0")
        assert "Electric-type" in chunks[0].text

    def test_long_doc_splits_into_multiple_chunks(self) -> None:
        sentences = [f"Sentence {i} about this Pokémon with enough words." for i in range(80)]
        doc = "Title: Abomasnow (Pokémon)\n\n" + " ".join(sentences)
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulba_0")
        assert len(chunks) > 1

    def test_chunk_indices_sequential(self) -> None:
        sentences = [f"Sentence {i} about this Pokémon." for i in range(80)]
        doc = "Title: Bulbasaur (Pokémon)\n\n" + " ".join(sentences)
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulba_0")
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_non_pokemon_title_extracts_name_prefix(self) -> None:
        doc = "Title: Pokémon (species)\n\nPokémon are fictional creatures."
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulba_0")
        assert chunks[0].entity_name == "Pokémon"

    def test_all_text_covered_across_chunks(self) -> None:
        doc = (
            "Title: Pikachu (Pokémon)\n\nPikachu uses Thunderbolt. It loves ketchup. Ash owns one."
        )
        chunks = chunk_bulbapedia_doc(doc, doc_id="bulba_0")
        combined = " ".join(c.text for c in chunks)
        assert "ketchup" in combined
        assert "Thunderbolt" in combined

    def test_empty_doc_returns_empty(self) -> None:
        assert chunk_bulbapedia_doc("", doc_id="bulba_0") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert chunk_bulbapedia_doc("   \n\n  ", doc_id="bulba_0") == []


@pytest.mark.unit
class TestChunkFile:
    def test_pokeapi_one_chunk_per_non_empty_line(self, tmp_path: pytest.TempPathFactory) -> None:
        p = tmp_path / "pokemon_species.txt"  # type: ignore[operator]
        p.write_text(
            "Bulbasaur is a Seed Pokémon.\nIvysaur is a Seed Pokémon.\n",
            encoding="utf-8",
        )
        chunks = chunk_file(p, source="pokeapi")
        assert len(chunks) == 2

    def test_pokeapi_correct_source(self, tmp_path: pytest.TempPathFactory) -> None:
        p = tmp_path / "pokemon.txt"  # type: ignore[operator]
        p.write_text("Bulbasaur is a Seed Pokémon.\n", encoding="utf-8")
        chunks = chunk_file(p, source="pokeapi")
        assert all(c.source == "pokeapi" for c in chunks)

    def test_smogon_one_entry_per_line(self, tmp_path: pytest.TempPathFactory) -> None:
        p = tmp_path / "pokemon.txt"  # type: ignore[operator]
        p.write_text(
            "Garbodor (NU): Great spiker with bulk.\nGyarados (OU): Powerful sweeper.\n",
            encoding="utf-8",
        )
        chunks = chunk_file(p, source="smogon")
        assert len(chunks) >= 2

    def test_smogon_correct_source(self, tmp_path: pytest.TempPathFactory) -> None:
        p = tmp_path / "pokemon.txt"  # type: ignore[operator]
        p.write_text("Garbodor (NU): Great spiker.\n", encoding="utf-8")
        chunks = chunk_file(p, source="smogon")
        assert all(c.source == "smogon" for c in chunks)

    def test_bulbapedia_splits_on_title_boundary(self, tmp_path: pytest.TempPathFactory) -> None:
        p = tmp_path / "pokemon.txt"  # type: ignore[operator]
        content = (
            "Title: Pikachu (Pokémon)\n\nPikachu is electric.\n\n"
            "Title: Raichu (Pokémon)\n\nRaichu is the evolved form.\n"
        )
        p.write_text(content, encoding="utf-8")
        chunks = chunk_file(p, source="bulbapedia")
        names = {c.entity_name for c in chunks}
        assert "Pikachu" in names
        assert "Raichu" in names

    def test_bulbapedia_correct_source(self, tmp_path: pytest.TempPathFactory) -> None:
        p = tmp_path / "pokemon.txt"  # type: ignore[operator]
        p.write_text("Title: Pikachu (Pokémon)\n\nPikachu is electric.\n", encoding="utf-8")
        chunks = chunk_file(p, source="bulbapedia")
        assert all(c.source == "bulbapedia" for c in chunks)

    def test_empty_file_returns_empty(self, tmp_path: pytest.TempPathFactory) -> None:
        p = tmp_path / "empty.txt"  # type: ignore[operator]
        p.write_text("", encoding="utf-8")
        for source in ("pokeapi", "smogon", "bulbapedia"):
            chunks = chunk_file(p, source=source)  # type: ignore[arg-type]
            assert chunks == [], f"expected [] for source={source}"
