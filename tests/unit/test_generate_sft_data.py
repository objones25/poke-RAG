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
