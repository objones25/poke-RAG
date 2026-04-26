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

    def test_single_chunk_text_starts_with_entity_header(self) -> None:
        line = "Pikachu (OU): Pikachu is a fast attacker with high speed."
        chunks = chunk_smogon_line(line, doc_id="smogon_0")
        assert chunks[0].text.startswith("Pikachu: ")

    def test_multi_chunk_all_contain_entity_header(self) -> None:
        sentences = [f"Sentence {i} with extra content for padding." for i in range(60)]
        line = "Gyarados (OU): " + " ".join(sentences)
        chunks = chunk_smogon_line(line, doc_id="smogon_0")
        assert len(chunks) > 1
        for c in chunks:
            assert c.text.startswith("Gyarados: "), f"chunk missing header: {c.text[:40]!r}"

    def test_no_header_when_entity_name_none(self) -> None:
        line = "just some text without a name pattern"
        chunks = chunk_smogon_line(line, doc_id="smogon_0")
        assert not chunks[0].text.startswith(": ")


@pytest.mark.unit
class TestChunkSmogonLineTokenizeFn:
    def test_chunk_file_smogon_accepts_tokenize_fn(self, tmp_path) -> None:
        from src.retrieval.chunker import chunk_file

        p = tmp_path / "pokemon.txt"
        sentences = [f"Sentence number {i} with plenty of content here." for i in range(30)]
        line = "Garbodor (NU): " + " ".join(sentences)
        p.write_text(line + "\n", encoding="utf-8")

        # Aggressive tokenizer: 2× word count → forces splits on shorter text
        def aggressive_fn(text: str) -> int:
            return len(text.split()) * 2

        default_chunks = chunk_file(p, source="smogon")
        aggressive_chunks = chunk_file(p, source="smogon", tokenize_fn=aggressive_fn)

        assert len(aggressive_chunks) >= len(default_chunks), (
            "aggressive tokenizer should produce at least as many chunks"
        )

    def test_chunk_smogon_line_tokenize_fn_threads_through(self) -> None:
        from src.retrieval.chunker import chunk_smogon_line

        sentences = [f"Sentence {i} with enough words for the test." for i in range(30)]
        line = "Gyarados (OU): " + " ".join(sentences)

        # tokenize_fn that inflates counts 10×: should produce many more chunks
        def inflating_fn(text: str) -> int:
            return len(text.split()) * 10

        chunks = chunk_smogon_line(line, doc_id="s0", tokenize_fn=inflating_fn)
        assert len(chunks) > 3


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
            chunks = chunk_file(p, source=source)
            assert chunks == [], f"expected [] for source={source}"


@pytest.mark.unit
class TestChunkerConstants:
    def test_bulbapedia_target_tokens_is_400(self) -> None:
        from src.retrieval.chunker import _BULBA_TARGET_TOKENS

        assert _BULBA_TARGET_TOKENS == 400


_EQ_SEP = "=" * 80
_DASH_SEP = "-" * 40

_SMOGON_DATA_SAMPLE = (
    _EQ_SEP + "\n"
    "VENUSAUR  (Grass/Poison)  —  introduced Gen 1\n"
    "Smogon form: Venusaur\n"
    + _EQ_SEP + "\n"
    "\n"
    + _DASH_SEP + "\n"
    " Format: gen1ou\n"
    + _DASH_SEP + "\n"
    "\n"
    "[ Overview ]\n"
    "Venusaur is a strong sleeper in gen1ou. It can threaten Water-types.\n"
    "\n"
    "[ Set: Swords Dance ]\n"
    "  Moves:\n"
    "    - Sleep Powder\n"
    "    - Razor Leaf\n"
    "    - Swords Dance\n"
    "    - Hyper Beam\n"
    "\n"
    "  Description:\n"
    "  Sleep Powder lets Venusaur incapacitate foes.\n"
    "\n"
    + _DASH_SEP + "\n"
    " Format: gen1pu\n"
    + _DASH_SEP + "\n"
    "\n"
    "[ Set: Sleeper ]\n"
    "  Moves:\n"
    "    - Sleep Powder\n"
    "    - Razor Leaf\n"
    "    - Body Slam\n"
    "    - Hyper Beam\n"
    "\n"
    + _DASH_SEP + "\n"
    " Format: gen9uu\n"
    + _DASH_SEP + "\n"
    "\n"
    "[ Set: Sun Sweeper ]\n"
    "  Tera Type: Fire\n"
    "  Item:      Life Orb\n"
    "  Ability:   Chlorophyll\n"
    "  Nature:    Timid\n"
    "  EVs:       252 SpA / 4 SpD / 252 Spe\n"
    "  IVs:       0 Atk\n"
    "  Moves:\n"
    "    - Growth\n"
    "    - Weather Ball\n"
    "    - Giga Drain\n"
    "    - Sludge Bomb\n"
    "\n"
    "  Description:\n"
    "  Venusaur uses Chlorophyll to sweep in sun.\n"
    "\n"
    + _EQ_SEP + "\n"
    "CHARIZARD  (Fire/Flying)  —  introduced Gen 1\n"
    "Smogon form: Charizard\n"
    + _EQ_SEP + "\n"
    "\n"
    + _DASH_SEP + "\n"
    " Format: gen1ou\n"
    + _DASH_SEP + "\n"
    "\n"
    "[ Overview ]\n"
    "Charizard is unviable in RBY OU.\n"
)


@pytest.mark.unit
class TestChunkSmogonDataFile:
    """Tests for the smogon_data.txt multi-block structured parser."""

    def _chunks(self) -> list:
        from src.retrieval.chunker import chunk_smogon_data_file

        return chunk_smogon_data_file(_SMOGON_DATA_SAMPLE)

    def test_returns_correct_chunk_count(self) -> None:
        # Overview(gen1ou) + Set:SwordsD(gen1ou) + Set:Sleeper(gen1pu) +
        # Set:SunSweeper(gen9uu) + Overview:Charizard(gen1ou) = 5
        chunks = self._chunks()
        assert len(chunks) == 5

    def test_source_is_smogon(self) -> None:
        chunks = self._chunks()
        assert all(c.source == "smogon" for c in chunks)

    def test_entity_type_is_pokemon(self) -> None:
        chunks = self._chunks()
        assert all(c.entity_type == "pokemon" for c in chunks)

    def test_score_is_zero(self) -> None:
        chunks = self._chunks()
        assert all(c.score == 0.0 for c in chunks)

    def test_venusaur_entity_name_on_all_venusaur_chunks(self) -> None:
        chunks = self._chunks()
        venusaur = [c for c in chunks if c.entity_name == "Venusaur"]
        assert len(venusaur) == 4

    def test_charizard_entity_name_on_charizard_chunk(self) -> None:
        chunks = self._chunks()
        charizard = [c for c in chunks if c.entity_name == "Charizard"]
        assert len(charizard) == 1

    def test_overview_chunk_text_format(self) -> None:
        chunks = self._chunks()
        overview = next(
            c for c in chunks if c.entity_name == "Venusaur" and "Overview" in c.text
        )
        assert overview.text.startswith("Venusaur in gen1ou — Overview")
        assert "strong sleeper" in overview.text

    def test_set_chunk_with_description_text_format(self) -> None:
        chunks = self._chunks()
        swords_dance = next(
            c for c in chunks if c.entity_name == "Venusaur" and "Swords Dance" in c.text
        )
        assert "Venusaur in gen1ou — Set: Swords Dance" in swords_dance.text
        assert "Sleep Powder" in swords_dance.text
        assert "incapacitate" in swords_dance.text

    def test_set_chunk_without_description_is_produced(self) -> None:
        chunks = self._chunks()
        sleeper = next(
            c for c in chunks if c.entity_name == "Venusaur" and "Sleeper" in c.text
        )
        assert "Venusaur in gen1pu — Set: Sleeper" in sleeper.text
        assert "Sleep Powder" in sleeper.text

    def test_set_chunk_with_optional_attributes(self) -> None:
        chunks = self._chunks()
        sun = next(c for c in chunks if "Sun Sweeper" in c.text)
        # At least one optional attribute must appear
        assert any(kw in sun.text for kw in ("Tera Type", "Life Orb", "Chlorophyll"))
        assert "Growth" in sun.text

    def test_moves_included_in_set_chunk(self) -> None:
        chunks = self._chunks()
        swords_dance = next(
            c for c in chunks if c.entity_name == "Venusaur" and "Swords Dance" in c.text
        )
        assert "Razor Leaf" in swords_dance.text
        assert "Hyper Beam" in swords_dance.text

    def test_charizard_overview_text(self) -> None:
        chunks = self._chunks()
        char_overview = next(c for c in chunks if c.entity_name == "Charizard")
        assert "Charizard in gen1ou — Overview" in char_overview.text
        assert "unviable" in char_overview.text

    def test_doc_id_contains_pokemon_slug(self) -> None:
        chunks = self._chunks()
        venusaur_chunks = [c for c in chunks if c.entity_name == "Venusaur"]
        assert all("venusaur" in c.original_doc_id for c in venusaur_chunks)

    def test_doc_id_contains_format_name(self) -> None:
        chunks = self._chunks()
        gen9uu = [c for c in chunks if "gen9uu" in (c.original_doc_id or "")]
        assert len(gen9uu) >= 1

    def test_chunk_index_zero_for_short_chunks(self) -> None:
        chunks = self._chunks()
        assert all(c.chunk_index == 0 for c in chunks)

    def test_chunk_file_dispatches_smogon_data_stem(self, tmp_path) -> None:
        p = tmp_path / "smogon_data.txt"
        p.write_text(_SMOGON_DATA_SAMPLE, encoding="utf-8")
        chunks = chunk_file(p, source="smogon")
        assert len(chunks) == 5
        entity_names = {c.entity_name for c in chunks}
        assert "Venusaur" in entity_names
        assert "Charizard" in entity_names

    def test_chunk_file_old_smogon_format_unaffected(self, tmp_path) -> None:
        p = tmp_path / "formats.txt"
        p.write_text(
            "Ubers (tier): The highest tier.\nOU (tier): Standard play tier.\n",
            encoding="utf-8",
        )
        chunks = chunk_file(p, source="smogon")
        assert len(chunks) >= 2
        assert all(c.source == "smogon" for c in chunks)
