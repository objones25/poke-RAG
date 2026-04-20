"""Unit tests for src/retrieval/context_assembler.py — RED until implemented."""
from __future__ import annotations

import pytest

from src.retrieval.context_assembler import ContextAssembler
from src.types import RetrievedChunk


def _make_chunk(
    text: str = "sample text",
    score: float = 0.9,
    source: str = "pokeapi",
    entity_name: str | None = "Bulbasaur",
    entity_type: str | None = "pokemon",
    chunk_index: int = 0,
    doc_id: str = "doc_0",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        source=source,  # type: ignore[arg-type]
        entity_name=entity_name,
        entity_type=entity_type,  # type: ignore[arg-type]
        chunk_index=chunk_index,
        original_doc_id=doc_id,
    )


@pytest.mark.unit
class TestContextAssemblerAssemble:
    def test_empty_chunks_returns_empty_string(self) -> None:
        assembler = ContextAssembler()
        assert assembler.assemble([]) == ""

    def test_single_chunk_text_in_output(self) -> None:
        assembler = ContextAssembler()
        chunk = _make_chunk(text="Bulbasaur is a Grass/Poison type.")
        result = assembler.assemble([chunk])
        assert "Bulbasaur is a Grass/Poison type." in result

    def test_multiple_chunks_all_text_included(self) -> None:
        assembler = ContextAssembler()
        chunks = [
            _make_chunk(text="text one", doc_id="doc_0"),
            _make_chunk(text="text two", doc_id="doc_1", chunk_index=1),
        ]
        result = assembler.assemble(chunks)
        assert "text one" in result
        assert "text two" in result

    def test_source_attribution_in_output(self) -> None:
        assembler = ContextAssembler()
        chunk = _make_chunk(text="some fact", source="smogon")
        result = assembler.assemble([chunk])
        assert "smogon" in result

    def test_pokemon_name_in_output_when_present(self) -> None:
        assembler = ContextAssembler()
        chunk = _make_chunk(text="Pikachu moves", entity_name="Pikachu")
        result = assembler.assemble([chunk])
        assert "Pikachu" in result

    def test_chunks_separated_in_output(self) -> None:
        assembler = ContextAssembler()
        chunks = [
            _make_chunk(text="first chunk", doc_id="doc_0"),
            _make_chunk(text="second chunk", doc_id="doc_1", chunk_index=1),
        ]
        result = assembler.assemble(chunks)
        first_pos = result.index("first chunk")
        second_pos = result.index("second chunk")
        assert first_pos < second_pos

    def test_deduplicates_identical_text(self) -> None:
        assembler = ContextAssembler()
        chunks = [
            _make_chunk(text="duplicate text", score=0.9, doc_id="doc_0"),
            _make_chunk(text="duplicate text", score=0.5, doc_id="doc_1", chunk_index=1),
        ]
        result = assembler.assemble(chunks)
        assert result.count("duplicate text") == 1

    def test_deduplication_keeps_highest_score(self) -> None:
        assembler = ContextAssembler()
        chunks = [
            _make_chunk(text="dup", score=0.3, source="pokeapi", doc_id="doc_0"),
            _make_chunk(text="dup", score=0.9, source="smogon", doc_id="doc_1", chunk_index=1),
        ]
        result = assembler.assemble(chunks)
        assert "smogon" in result
        assert "pokeapi" not in result

    def test_deduplication_emits_winner_once(self) -> None:
        assembler = ContextAssembler()
        chunks = [
            _make_chunk(text="shared", score=0.5, doc_id="doc_0"),
            _make_chunk(text="shared", score=0.8, doc_id="doc_1", chunk_index=1),
            _make_chunk(text="shared", score=0.2, doc_id="doc_2", chunk_index=2),
        ]
        result = assembler.assemble(chunks)
        assert result.count("shared") == 1

    def test_token_budget_truncates_last_chunk_not_omits(self) -> None:
        assembler = ContextAssembler(max_tokens=10)
        short = _make_chunk(text="short", doc_id="doc_0")
        long_text = "word " * 100
        long_chunk = _make_chunk(text=long_text, doc_id="doc_1", chunk_index=1)
        result = assembler.assemble([short, long_chunk])
        assert "short" in result
        assert len(result.split()) < 30

    def test_returns_string(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble([_make_chunk()])
        assert isinstance(result, str)

    def test_max_tokens_limits_output_length(self) -> None:
        assembler = ContextAssembler(max_tokens=20)
        long_text = "word " * 200
        chunk = _make_chunk(text=long_text)
        result = assembler.assemble([chunk])
        approx_tokens = len(result.split()) / 0.75
        assert approx_tokens <= 20 * 1.2  # 20% tolerance for header overhead

    def test_chunk_order_preserved(self) -> None:
        assembler = ContextAssembler()
        chunks = [
            _make_chunk(text="alpha", doc_id="doc_0", chunk_index=0),
            _make_chunk(text="beta", doc_id="doc_1", chunk_index=1),
            _make_chunk(text="gamma", doc_id="doc_2", chunk_index=2),
        ]
        result = assembler.assemble(chunks)
        assert result.index("alpha") < result.index("beta") < result.index("gamma")

    def test_none_pokemon_name_not_included_as_none_string(self) -> None:
        assembler = ContextAssembler()
        chunk = _make_chunk(text="generic fact", entity_name=None)
        result = assembler.assemble([chunk])
        assert "None" not in result

    def test_separator_customisable(self) -> None:
        assembler = ContextAssembler(separator="===")
        chunks = [
            _make_chunk(text="chunk a", doc_id="doc_0"),
            _make_chunk(text="chunk b", doc_id="doc_1", chunk_index=1),
        ]
        result = assembler.assemble(chunks)
        assert "===" in result


@pytest.mark.unit
class TestContextAssemblerDefaults:
    def test_default_max_tokens_is_large(self) -> None:
        assembler = ContextAssembler()
        long_chunks = [
            _make_chunk(text="word " * 100, doc_id=f"doc_{i}", chunk_index=i) for i in range(5)
        ]
        result = assembler.assemble(long_chunks)
        assert len(result) > 0

    def test_custom_separator_used(self) -> None:
        sep = "\n~~\n"
        assembler = ContextAssembler(separator=sep)
        chunks = [
            _make_chunk(text="a", doc_id="doc_0"),
            _make_chunk(text="b", doc_id="doc_1", chunk_index=1),
        ]
        result = assembler.assemble(chunks)
        assert sep in result
