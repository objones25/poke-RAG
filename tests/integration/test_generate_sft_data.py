from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.training.generate_sft_data import run
from scripts.training.schemas import GeminiQAPair


def _fake_qa(chunk: str, source: str) -> GeminiQAPair:
    return GeminiQAPair(
        question=f"What does {source} say?",
        answer=f"According to {source}: {chunk[:40]}.",
    )


def _make_processed(tmp_path: Path) -> Path:
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
                for i in range(100)
            )
        )
    return processed


@pytest.mark.integration
class TestRunGeneration:
    def test_generates_goal_number_of_pairs(self, tmp_path: Path) -> None:
        processed = _make_processed(tmp_path)
        output = tmp_path / "out.jsonl"
        with patch("scripts.training.generate_sft_data.GeminiClient") as MockClient:
            MockClient.return_value.generate_qa_pair.side_effect = _fake_qa
            run(
                goal=10,
                output=output,
                processed_dir=processed,
                api_key="fake",
                model="gemini-3.1-flash-lite-preview",
                seed=42,
                delay=0.0,
                max_per_entity=5,
                source_weights={"bulbapedia": 0.4, "pokeapi": 0.4, "smogon": 0.2},
                include_aug=False,
            )
        lines = [line for line in output.read_text().splitlines() if line.strip()]
        assert len(lines) == 10

    def test_output_jsonl_has_correct_schema(self, tmp_path: Path) -> None:
        processed = _make_processed(tmp_path)
        output = tmp_path / "out.jsonl"
        with patch("scripts.training.generate_sft_data.GeminiClient") as MockClient:
            MockClient.return_value.generate_qa_pair.side_effect = _fake_qa
            run(
                goal=3,
                output=output,
                processed_dir=processed,
                api_key="fake",
                model="gemini-3.1-flash-lite-preview",
                seed=42,
                delay=0.0,
                max_per_entity=5,
                source_weights={"bulbapedia": 0.4, "pokeapi": 0.4, "smogon": 0.2},
                include_aug=False,
            )
        for raw_line in output.read_text().splitlines():
            data = json.loads(raw_line)
            msgs = data["messages"]
            assert len(msgs) == 2
            assert [m["role"] for m in msgs] == ["user", "assistant"]

    def test_resume_from_partial_output(self, tmp_path: Path) -> None:
        processed = _make_processed(tmp_path)
        output = tmp_path / "out.jsonl"
        call_count: dict[str, int] = {"n": 0}

        def counting_qa(chunk: str, source: str) -> GeminiQAPair:
            call_count["n"] += 1
            return _fake_qa(chunk, source)

        with patch("scripts.training.generate_sft_data.GeminiClient") as MockClient:
            MockClient.return_value.generate_qa_pair.side_effect = counting_qa
            run(
                goal=5,
                output=output,
                processed_dir=processed,
                api_key="fake",
                model="gemini-3.1-flash-lite-preview",
                seed=42,
                delay=0.0,
                max_per_entity=5,
                source_weights={"bulbapedia": 0.4, "pokeapi": 0.4, "smogon": 0.2},
                include_aug=False,
            )
        assert call_count["n"] == 5

        call_count["n"] = 0
        with patch("scripts.training.generate_sft_data.GeminiClient") as MockClient:
            MockClient.return_value.generate_qa_pair.side_effect = counting_qa
            run(
                goal=10,
                output=output,
                processed_dir=processed,
                api_key="fake",
                model="gemini-3.1-flash-lite-preview",
                seed=42,
                delay=0.0,
                max_per_entity=5,
                source_weights={"bulbapedia": 0.4, "pokeapi": 0.4, "smogon": 0.2},
                include_aug=False,
            )
        assert call_count["n"] == 5
        assert len([line for line in output.read_text().splitlines() if line.strip()]) == 10

    def test_deduplication_limits_per_entity(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        d = processed / "pokeapi"
        d.mkdir(parents=True)
        (d / "pokemon.txt").write_text(
            "\n".join(
                (
                    f"Pikachu is a Electric-type Pokemon"
                    f" with base Speed 90 and base Attack 55, variant {i}."
                )
                for i in range(30)
            )
        )
        output = tmp_path / "out.jsonl"
        with patch("scripts.training.generate_sft_data.GeminiClient") as MockClient:
            MockClient.return_value.generate_qa_pair.side_effect = _fake_qa
            run(
                goal=20,
                output=output,
                processed_dir=processed,
                api_key="fake",
                model="gemini-3.1-flash-lite-preview",
                seed=42,
                delay=0.0,
                max_per_entity=3,
                source_weights={"pokeapi": 1.0},
                include_aug=False,
            )
        lines = [line for line in output.read_text().splitlines() if line.strip()]
        assert len(lines) == 3

    def test_none_return_from_quality_gate_is_skipped(self, tmp_path: Path) -> None:
        processed = _make_processed(tmp_path)
        output = tmp_path / "out.jsonl"
        call_count: dict[str, int] = {"n": 0}

        def alternating_qa(chunk: str, source: str) -> GeminiQAPair | None:
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                return None
            return _fake_qa(chunk, source)

        with patch("scripts.training.generate_sft_data.GeminiClient") as MockClient:
            MockClient.return_value.generate_qa_pair.side_effect = alternating_qa
            run(
                goal=5,
                output=output,
                processed_dir=processed,
                api_key="fake",
                model="gemini-3.1-flash-lite-preview",
                seed=42,
                delay=0.0,
                max_per_entity=5,
                source_weights={"bulbapedia": 0.4, "pokeapi": 0.4, "smogon": 0.2},
                include_aug=False,
            )
        lines = [line for line in output.read_text().splitlines() if line.strip()]
        assert len(lines) == 5
        assert call_count["n"] > 5
