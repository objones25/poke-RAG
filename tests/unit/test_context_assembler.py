"""Unit tests for src/retrieval/context_assembler.py — RED until implemented."""

from __future__ import annotations

import pytest

from src.retrieval.context_assembler import ContextAssembler
from tests.conftest import make_chunk as _make_chunk


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
            _make_chunk(text="text one", original_doc_id="doc_0"),
            _make_chunk(text="text two", original_doc_id="doc_1", chunk_index=1),
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
            _make_chunk(text="first chunk", original_doc_id="doc_0"),
            _make_chunk(text="second chunk", original_doc_id="doc_1", chunk_index=1),
        ]
        result = assembler.assemble(chunks)
        first_pos = result.index("first chunk")
        second_pos = result.index("second chunk")
        assert first_pos < second_pos

    def test_deduplicates_identical_chunk_identity(self) -> None:
        assembler = ContextAssembler()
        chunks = [
            _make_chunk(text="duplicate text", score=0.9, original_doc_id="doc_0", chunk_index=0),
            _make_chunk(text="duplicate text", score=0.5, original_doc_id="doc_0", chunk_index=0),
        ]
        result = assembler.assemble(chunks)
        assert result.count("duplicate text") == 1

    def test_deduplication_keeps_highest_score(self) -> None:
        assembler = ContextAssembler()
        chunks = [
            _make_chunk(
                text="dup", score=0.3, source="pokeapi", original_doc_id="doc_0", chunk_index=0
            ),
            _make_chunk(
                text="dup", score=0.9, source="smogon", original_doc_id="doc_0", chunk_index=0
            ),
        ]
        result = assembler.assemble(chunks)
        assert "smogon" in result
        assert "pokeapi" not in result

    def test_deduplication_emits_winner_once(self) -> None:
        assembler = ContextAssembler()
        chunks = [
            _make_chunk(text="shared", score=0.5, original_doc_id="doc_0", chunk_index=0),
            _make_chunk(text="shared", score=0.8, original_doc_id="doc_0", chunk_index=0),
            _make_chunk(text="shared", score=0.2, original_doc_id="doc_0", chunk_index=0),
        ]
        result = assembler.assemble(chunks)
        assert result.count("shared") == 1

    def test_token_budget_truncates_last_chunk_not_omits(self) -> None:
        assembler = ContextAssembler(max_tokens=10)
        short = _make_chunk(text="short", original_doc_id="doc_0")
        long_text = "word " * 100
        long_chunk = _make_chunk(text=long_text, original_doc_id="doc_1", chunk_index=1)
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
        assert approx_tokens <= 20

    def test_chunk_order_preserved(self) -> None:
        assembler = ContextAssembler()
        chunks = [
            _make_chunk(text="alpha", original_doc_id="doc_0", chunk_index=0),
            _make_chunk(text="beta", original_doc_id="doc_1", chunk_index=1),
            _make_chunk(text="gamma", original_doc_id="doc_2", chunk_index=2),
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
            _make_chunk(text="chunk a", original_doc_id="doc_0"),
            _make_chunk(text="chunk b", original_doc_id="doc_1", chunk_index=1),
        ]
        result = assembler.assemble(chunks)
        assert "===" in result


@pytest.mark.unit
class TestContextAssemblerDefaults:
    def test_default_max_tokens_is_large(self) -> None:
        assembler = ContextAssembler()
        long_chunks = [
            _make_chunk(text="word " * 100, original_doc_id=f"doc_{i}", chunk_index=i)
            for i in range(5)
        ]
        result = assembler.assemble(long_chunks)
        assert len(result) > 0

    def test_custom_separator_used(self) -> None:
        sep = "\n~~\n"
        assembler = ContextAssembler(separator=sep)
        chunks = [
            _make_chunk(text="a", original_doc_id="doc_0"),
            _make_chunk(text="b", original_doc_id="doc_1", chunk_index=1),
        ]
        result = assembler.assemble(chunks)
        assert sep in result


@pytest.mark.unit
class TestContextAssemblerTokenBudget:
    """Test that context assembler respects token budget strictly."""

    def test_assembled_output_stays_within_token_budget(self) -> None:
        """Verify that assembled output stays within the max_tokens budget."""
        max_tokens = 50
        assembler = ContextAssembler(max_tokens=max_tokens)

        # Create a chunk with text longer than the budget
        long_text = "word " * 200
        chunk = _make_chunk(text=long_text)

        result = assembler.assemble([chunk])

        # Count approximate tokens: words / 0.75
        word_count = len(result.split())
        approx_tokens = word_count / 0.75

        # With 5% safety margin, should be strictly under budget
        assert approx_tokens <= max_tokens, (
            f"Result has ~{approx_tokens} tokens but budget is {max_tokens}"
        )

    def test_assembled_chunks_respect_budget_with_multiple_chunks(self) -> None:
        """Multiple chunks should respect the token budget when assembled."""
        max_tokens = 100
        assembler = ContextAssembler(max_tokens=max_tokens)

        chunks = [
            _make_chunk(text="word " * 100, original_doc_id="doc_0"),
            _make_chunk(text="word " * 100, original_doc_id="doc_1", chunk_index=1),
            _make_chunk(text="word " * 100, original_doc_id="doc_2", chunk_index=2),
        ]

        result = assembler.assemble(chunks)

        # Approximate token count
        word_count = len(result.split())
        approx_tokens = word_count / 0.75

        assert approx_tokens <= max_tokens, (
            f"Result has ~{approx_tokens} tokens but budget is {max_tokens}"
        )
