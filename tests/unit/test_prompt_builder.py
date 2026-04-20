import pytest

from src.generation.prompt_builder import build_prompt
from src.types import RetrievedChunk


def _chunk(
    text: str,
    score: float,
    source: str = "bulbapedia",
    entity_name: str | None = "Charizard",
    entity_type: str | None = "pokemon",
    chunk_index: int = 0,
    original_doc_id: str = "doc1",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        source=source,  # type: ignore[arg-type]
        entity_name=entity_name,
        entity_type=entity_type,  # type: ignore[arg-type]
        chunk_index=chunk_index,
        original_doc_id=original_doc_id,
    )


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

    def test_sources_are_deduplicated(self) -> None:
        chunks = (
            _chunk("Text A.", score=0.9, source="bulbapedia", chunk_index=0),
            _chunk("Text B.", score=0.7, source="bulbapedia", chunk_index=1),
        )
        prompt = build_prompt("Question?", chunks)
        assert prompt.lower().count("bulbapedia") >= 1

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
