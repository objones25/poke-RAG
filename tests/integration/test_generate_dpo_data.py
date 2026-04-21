from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from scripts.training.generate_dpo_data import run
from scripts.training.schemas import JudgeScore
from src.types import RetrievalResult, RetrievedChunk


def _make_chunk(text: str = "Pikachu is Electric-type.") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=0.9,
        source="pokeapi",  # type: ignore[arg-type]
        entity_name="Pikachu",
        entity_type="pokemon",
        chunk_index=0,
        original_doc_id="doc_0",
    )


def _make_retrieval_result(texts: list[str]) -> RetrievalResult:
    return RetrievalResult(
        documents=tuple(_make_chunk(t) for t in texts),
        query="test",
    )


def _make_processor(responses: list[str] | None = None) -> MagicMock:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "<tmpl>prompt</tmpl>"
    encoded = MagicMock()
    encoded.__contains__ = MagicMock(return_value=False)
    input_ids = torch.zeros((1, 10), dtype=torch.long)
    encoded.__getitem__ = MagicMock(return_value=input_ids)
    processor.return_value = encoded

    if responses is not None:
        _idx = [0]

        def _decode(*args: object, **kwargs: object) -> str:
            resp = responses[_idx[0] % len(responses)]
            _idx[0] += 1
            return resp

        processor.decode.side_effect = _decode
    else:
        _n = [0]

        def _unique_decode(*args: object, **kwargs: object) -> str:
            _n[0] += 1
            return f"Response {_n[0]}"

        processor.decode.side_effect = _unique_decode

    return processor


def _make_model() -> MagicMock:
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.zeros((1, 15), dtype=torch.long)
    return model


def _score(total: int) -> JudgeScore:
    per = total // 3
    rem = total % 3
    return JudgeScore(
        accuracy=per + (1 if rem > 0 else 0),
        groundedness=per + (1 if rem > 1 else 0),
        domain_correctness=per,
    )


@pytest.mark.integration
class TestRunDPOGeneration:
    def test_generates_goal_number_of_pairs(self, tmp_path: Path) -> None:
        output = tmp_path / "dpo.jsonl"
        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(["chunk"])
        judge = MagicMock()
        judge.score_candidate.side_effect = [
            _score(270),
            _score(150),
            _score(200),
            _score(100),
        ]

        run(
            goal=2,
            output=output,
            questions=["What type is Pikachu?", "What type is Bulbasaur?"],
            retriever=retriever,
            model=_make_model(),
            processor=_make_processor(),
            judge=judge,
            k=2,
            delay=0.0,
        )

        lines = [line for line in output.read_text().splitlines() if line.strip()]
        assert len(lines) == 2

    def test_chosen_has_higher_judge_score(self, tmp_path: Path) -> None:
        output = tmp_path / "dpo.jsonl"
        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(["chunk"])
        judge = MagicMock()
        judge.score_candidate.side_effect = [_score(150), _score(270)]

        run(
            goal=1,
            output=output,
            questions=["What type is Pikachu?"],
            retriever=retriever,
            model=_make_model(),
            processor=_make_processor(
                responses=["Low quality answer.", "High quality detailed answer."]
            ),
            judge=judge,
            k=2,
            delay=0.0,
        )

        lines = [line for line in output.read_text().splitlines() if line.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["chosen"] == "High quality detailed answer."
        assert data["rejected"] == "Low quality answer."

    def test_skips_question_when_insufficient_scored_candidates(self, tmp_path: Path) -> None:
        output = tmp_path / "dpo.jsonl"
        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(["chunk"])
        judge = MagicMock()
        # Q1: both None → skipped; Q2: both scored → written
        judge.score_candidate.side_effect = [None, None, _score(200), _score(100)]

        run(
            goal=1,
            output=output,
            questions=["Unanswerable question?", "What type is Pikachu?"],
            retriever=retriever,
            model=_make_model(),
            processor=_make_processor(),
            judge=judge,
            k=2,
            delay=0.0,
        )

        lines = [line for line in output.read_text().splitlines() if line.strip()]
        assert len(lines) == 1

    def test_resume_from_partial_output(self, tmp_path: Path) -> None:
        output = tmp_path / "dpo.jsonl"
        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(["chunk"])
        questions = [f"Question {i}?" for i in range(10)]

        judge1 = MagicMock()
        judge1.score_candidate.side_effect = [_score(270), _score(150)] * 2
        run(
            goal=2,
            output=output,
            questions=questions,
            retriever=retriever,
            model=_make_model(),
            processor=_make_processor(),
            judge=judge1,
            k=2,
            delay=0.0,
        )
        assert len([ln for ln in output.read_text().splitlines() if ln.strip()]) == 2

        judge2 = MagicMock()
        judge2.score_candidate.side_effect = [_score(270), _score(150)] * 10
        run(
            goal=4,
            output=output,
            questions=questions,
            retriever=retriever,
            model=_make_model(),
            processor=_make_processor(),
            judge=judge2,
            k=2,
            delay=0.0,
        )

        assert len([ln for ln in output.read_text().splitlines() if ln.strip()]) == 4
        assert judge2.score_candidate.call_count == 4
