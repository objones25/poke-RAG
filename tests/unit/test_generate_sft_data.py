from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.training.generate_sft_data import run


def _make_processed_dir(tmp_path: Path) -> Path:
    """Create a minimal processed data directory for testing."""
    processed = tmp_path / "processed"
    for source in ("bulbapedia", "pokeapi", "smogon"):
        d = processed / source
        d.mkdir(parents=True)
        (d / "pokemon.txt").write_text(
            "\n".join(
                (
                    f"Entity{i} is a {source} Pokemon"
                    f" with base stats: HP 45, Atk 49, Def 49, SpA 65, SpD 65, Spe 45."
                )
                for i in range(10)
            )
        )
    return processed


@pytest.mark.unit
class TestRunGeneration:
    """Test error handling in generate_sft_data.run()."""

    def test_runtime_error_from_api_is_fatal(self, tmp_path: Path) -> None:
        """RED: RuntimeError (exhausted retries) should propagate, not be caught as warning."""
        processed = _make_processed_dir(tmp_path)
        output = tmp_path / "out.jsonl"

        def failing_qa(chunk: str, source: str):
            # Simulate the fatal error: all retries exhausted
            raise RuntimeError("Failed to generate Q&A pair after 3 attempts")

        with patch("scripts.training.generate_sft_data.GeminiClient") as MockClient:
            MockClient.return_value.generate_qa_pair = failing_qa
            # The fatal error should propagate, not be silently caught
            with pytest.raises(RuntimeError, match="Failed to generate"):
                run(
                    goal=5,
                    output=output,
                    processed_dir=processed,
                    api_key="fake",
                    model="gemini-3.1-flash-lite-preview",
                    seed=42,
                    delay=0.0,
                    max_per_entity=5,
                    source_weights={"bulbapedia": 1.0},
                    include_aug=False,
                )

    def test_non_fatal_exception_is_skipped(self, tmp_path: Path) -> None:
        """GREEN: Non-RuntimeError exceptions should be caught as warnings, not fatal."""
        processed = _make_processed_dir(tmp_path)
        output = tmp_path / "out.jsonl"
        call_count = [0]

        def sometimes_failing_qa(chunk: str, source: str):
            call_count[0] += 1
            if call_count[0] <= 2:
                # Non-fatal error (not RuntimeError)
                raise ValueError("Some transient API issue")
            # Success on 3rd try
            from scripts.training.schemas import GeminiQAPair

            return GeminiQAPair(
                question="What is this?",
                answer="This is a detailed answer with enough content to pass quality gates here.",
            )

        with patch("scripts.training.generate_sft_data.GeminiClient") as MockClient:
            MockClient.return_value.generate_qa_pair = sometimes_failing_qa
            # Should succeed by skipping the transient errors
            run(
                goal=1,
                output=output,
                processed_dir=processed,
                api_key="fake",
                model="gemini-3.1-flash-lite-preview",
                seed=42,
                delay=0.0,
                max_per_entity=5,
                source_weights={"bulbapedia": 1.0},
                include_aug=False,
            )
        # Should have written 1 line, and called qa multiple times due to retries
        lines = [line for line in output.read_text().splitlines() if line.strip()]
        assert len(lines) == 1


@pytest.mark.unit
class TestIsUsefulChunk:
    """Test _is_useful_chunk for filtering low-quality chunks."""

    def test_normal_chunk_is_useful(self) -> None:
        from scripts.training.generate_sft_data import _is_useful_chunk

        chunk = (
            "Pikachu is an Electric-type Pokémon known for its powerful "
            "Thunderbolt attack and iconic status as the mascot of the series."
        )
        assert _is_useful_chunk(chunk) is True

    def test_chunk_too_short_is_not_useful(self) -> None:
        from scripts.training.generate_sft_data import _is_useful_chunk

        chunk = "Pikachu"
        assert _is_useful_chunk(chunk) is False

    def test_chunk_below_min_length_threshold_is_not_useful(self) -> None:
        from scripts.training.generate_sft_data import _is_useful_chunk

        chunk = "X" * 79  # One below _MIN_CHUNK_LEN (80)
        assert _is_useful_chunk(chunk) is False

    def test_chunk_at_min_length_threshold_is_useful(self) -> None:
        from scripts.training.generate_sft_data import _is_useful_chunk

        chunk = "X" * 80  # Exactly at _MIN_CHUNK_LEN
        assert _is_useful_chunk(chunk) is True

    def test_chunk_with_var_placeholder_is_not_useful(self) -> None:
        from scripts.training.generate_sft_data import _is_useful_chunk

        chunk = (
            "This is a long chunk about Pokémon types and [VAR(name)] is important for the system."
            * 2
        )
        assert _is_useful_chunk(chunk) is False

    def test_chunk_with_db_id_is_not_useful(self) -> None:
        from scripts.training.generate_sft_data import _is_useful_chunk

        chunk = (
            "The Pokémon has Item0042 which references database record Dra6688 and is important."
            * 2
        )
        assert _is_useful_chunk(chunk) is False

    def test_chunk_with_short_number_suffix_is_useful(self) -> None:
        from scripts.training.generate_sft_data import _is_useful_chunk

        chunk = (
            "This is a long chunk about Pokemon with reference HP45 which is a normal stat "
            "abbreviation not a database ID." * 2
        )
        assert _is_useful_chunk(chunk) is True


@pytest.mark.unit
class TestCountLines:
    """Test _count_lines helper function."""

    def test_count_lines_nonexistent_file_returns_zero(self, tmp_path: Path) -> None:
        from scripts.training.generate_sft_data import _count_lines

        nonexistent = tmp_path / "nonexistent.txt"
        assert _count_lines(nonexistent) == 0

    def test_count_lines_empty_file_returns_zero(self, tmp_path: Path) -> None:
        from scripts.training.generate_sft_data import _count_lines

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        assert _count_lines(empty_file) == 0

    def test_count_lines_file_with_content(self, tmp_path: Path) -> None:
        from scripts.training.generate_sft_data import _count_lines

        content_file = tmp_path / "content.txt"
        content_file.write_text("line1\nline2\nline3\n")
        assert _count_lines(content_file) == 3

    def test_count_lines_skips_empty_lines(self, tmp_path: Path) -> None:
        from scripts.training.generate_sft_data import _count_lines

        content_file = tmp_path / "content.txt"
        content_file.write_text("line1\n\n\nline2\n  \nline3\n")
        assert _count_lines(content_file) == 3

    def test_count_lines_single_line_no_newline(self, tmp_path: Path) -> None:
        from scripts.training.generate_sft_data import _count_lines

        content_file = tmp_path / "single.txt"
        content_file.write_text("single_line")
        assert _count_lines(content_file) == 1
