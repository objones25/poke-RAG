import pytest

from src.generation.prompt_builder import build_prompt
from tests.conftest import make_chunk as _chunk


@pytest.mark.unit
class TestBuildPromptValidation:
    def test_raises_on_empty_chunks(self) -> None:
        with pytest.raises(ValueError, match="chunks"):
            build_prompt("What are Charizard's stats?", ())

    def test_raises_on_empty_query(self) -> None:
        chunk = _chunk("some text", score=0.9)
        with pytest.raises(ValueError, match="query"):
            build_prompt("", (chunk,))

    def test_raises_on_whitespace_only_query(self) -> None:
        chunk = _chunk("some text", score=0.9)
        with pytest.raises(ValueError, match="query"):
            build_prompt("   ", (chunk,))


@pytest.mark.unit
class TestBuildPromptContent:
    def test_contains_query(self) -> None:
        chunk = _chunk("Charizard is a Fire/Flying-type Pokémon.", score=0.9)
        prompt = build_prompt("What type is Charizard?", (chunk,))
        assert "What type is Charizard?" in prompt

    def test_single_chunk_includes_text(self) -> None:
        chunk = _chunk("Charizard is a Fire/Flying-type Pokémon.", score=0.9)
        prompt = build_prompt("What type is Charizard?", (chunk,))
        assert "Charizard is a Fire/Flying-type Pokémon." in prompt

    def test_single_chunk_includes_source(self) -> None:
        chunk = _chunk("Some text.", score=0.9, source="bulbapedia")
        prompt = build_prompt("What type is Charizard?", (chunk,))
        assert "bulbapedia" in prompt.lower()

    def test_includes_pokemon_name_when_present(self) -> None:
        chunk = _chunk("Charizard has 78 HP.", score=0.9, entity_name="Charizard")
        prompt = build_prompt("What is Charizard's HP?", (chunk,))
        assert "Charizard" in prompt

    def test_handles_none_pokemon_name(self) -> None:
        chunk = _chunk("Some generic fact.", score=0.9, entity_name=None)
        prompt = build_prompt("Tell me a fact.", (chunk,))
        assert "Some generic fact." in prompt

    def test_multiple_sources_listed_at_end(self) -> None:
        chunks = (
            _chunk("Bulbapedia text.", score=0.9, source="bulbapedia"),
            _chunk("Smogon text.", score=0.7, source="smogon"),
        )
        prompt = build_prompt("Tell me about Charizard.", chunks)
        assert "bulbapedia" in prompt.lower()
        assert "smogon" in prompt.lower()

    def test_no_standalone_sources_summary_line(self) -> None:
        # build_prompt no longer injects a "Sources: x, y" summary line;
        # source attribution lives in the per-chunk [Source: ...] headers.
        chunks = (
            _chunk("Text A.", score=0.9, source="bulbapedia", chunk_index=0),
            _chunk("Text B.", score=0.7, source="bulbapedia", chunk_index=1),
        )
        prompt = build_prompt("Question?", chunks)
        dynamic_section = prompt.split("EXAMPLES:")[1] if "EXAMPLES:" in prompt else prompt
        summary_lines = [
            line for line in dynamic_section.splitlines() if line.startswith("Sources:")
        ]
        assert summary_lines == []

    def test_multiple_chunks_highest_score_appears_first(self) -> None:
        chunks = (
            _chunk("Low score text.", score=0.3, source="pokeapi", chunk_index=0),
            _chunk("High score text.", score=0.95, source="bulbapedia", chunk_index=1),
        )
        prompt = build_prompt("Question?", chunks)
        assert prompt.index("High score text.") < prompt.index("Low score text.")

    def test_chunks_already_sorted_remain_in_order(self) -> None:
        chunks = (
            _chunk("First.", score=0.9, chunk_index=0),
            _chunk("Second.", score=0.8, chunk_index=1),
            _chunk("Third.", score=0.5, chunk_index=2),
        )
        prompt = build_prompt("Question?", chunks)
        assert prompt.index("First.") < prompt.index("Second.") < prompt.index("Third.")


@pytest.mark.unit
class TestBuildPromptSanitization:
    def test_control_chars_in_chunk_text_are_removed(self) -> None:
        chunk = _chunk("Pikachu\x00is\x01Electric.", score=0.9)
        prompt = build_prompt("What type?", (chunk,))
        assert "\x00" not in prompt
        assert "\x01" not in prompt
        assert "Pikachu" in prompt
        assert "Electric" in prompt

    def test_nfkc_normalization_applied_to_chunk_text(self) -> None:
        chunk = _chunk("The ﬁre type.", score=0.9)
        prompt = build_prompt("What type?", (chunk,))
        assert "fi" in prompt

    def test_invalid_source_in_chunk_becomes_unknown(self) -> None:
        from src.types import RetrievedChunk

        chunk = RetrievedChunk(  # type: ignore[arg-type]
            text="Some text.",
            score=0.9,
            source="wikipedia",
            entity_name=None,
            entity_type=None,
            chunk_index=0,
            original_doc_id="doc_0",
        )
        prompt = build_prompt("Question?", (chunk,))
        assert "unknown" in prompt.lower()
        assert "wikipedia" not in prompt

    def test_control_chars_in_entity_name_are_removed(self) -> None:
        chunk = _chunk("Pikachu text.", score=0.9, entity_name="Pika​chu")
        prompt = build_prompt("Question?", (chunk,))
        assert "​" not in prompt
        assert "Pikachu" in prompt
