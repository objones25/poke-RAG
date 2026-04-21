from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.training.judge_protocol import GeminiJudge, JudgeProtocol
from scripts.training.schemas import JudgeScore


@pytest.mark.unit
class TestGeminiJudge:
    def test_satisfies_judge_protocol(self) -> None:
        with patch("scripts.training.judge_protocol.genai"):
            judge = GeminiJudge(api_key="fake")
        assert isinstance(judge, JudgeProtocol)

    def test_score_candidate_returns_judge_score(self) -> None:
        mock_response = MagicMock()
        mock_response.text = '{"accuracy": 85, "groundedness": 90, "domain_correctness": 80}'

        with patch("scripts.training.judge_protocol.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response

            judge = GeminiJudge(api_key="fake")
            score = judge.score_candidate(
                question="What is Pikachu's type?",
                response="Pikachu is Electric-type.",
                retrieved_chunks=["Pikachu is an Electric-type Pokémon."],
                reference_chunks=["Pikachu is an Electric-type Pokémon species."],
            )

        assert isinstance(score, JudgeScore)
        assert score.accuracy == 85
        assert score.groundedness == 90
        assert score.domain_correctness == 80
        assert score.total == 255

    def test_score_candidate_returns_none_on_persistent_error(self) -> None:
        with patch("scripts.training.judge_protocol.genai") as mock_genai:
            with patch("scripts.training.judge_protocol.time"):
                mock_client = MagicMock()
                mock_genai.Client.return_value = mock_client
                mock_client.models.generate_content.side_effect = RuntimeError("API failed")

                judge = GeminiJudge(api_key="fake", max_retries=2)
                score = judge.score_candidate(
                    question="q",
                    response="r",
                    retrieved_chunks=["c"],
                    reference_chunks=["ref"],
                )

        assert score is None

    def test_score_candidate_retries_on_rate_limit(self) -> None:
        mock_response = MagicMock()
        mock_response.text = '{"accuracy": 70, "groundedness": 75, "domain_correctness": 80}'
        call_count: dict[str, int] = {"n": 0}

        def side_effect(*_args: object, **_kwargs: object) -> MagicMock:
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise Exception("429 RESOURCE_EXHAUSTED")
            return mock_response

        with patch("scripts.training.judge_protocol.genai") as mock_genai:
            with patch("scripts.training.judge_protocol.time"):
                mock_client = MagicMock()
                mock_genai.Client.return_value = mock_client
                mock_client.models.generate_content.side_effect = side_effect

                judge = GeminiJudge(api_key="fake", max_retries=5)
                score = judge.score_candidate(
                    question="q",
                    response="r",
                    retrieved_chunks=["c"],
                    reference_chunks=["ref"],
                )

        assert score is not None
        assert call_count["n"] == 3

    def test_prompt_contains_question_response_and_context(self) -> None:
        mock_response = MagicMock()
        mock_response.text = '{"accuracy": 80, "groundedness": 80, "domain_correctness": 80}'

        with patch("scripts.training.judge_protocol.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response

            judge = GeminiJudge(api_key="fake")
            judge.score_candidate(
                question="What are Gardevoir's stats?",
                response="Gardevoir has SpA 125.",
                retrieved_chunks=["Gardevoir SpA: 125"],
                reference_chunks=["Gardevoir is Psychic/Fairy with SpA 125."],
            )

        contents = mock_client.models.generate_content.call_args.kwargs["contents"]
        assert "What are Gardevoir's stats?" in contents
        assert "Gardevoir has SpA 125." in contents
        assert "Gardevoir SpA: 125" in contents
        assert "Gardevoir is Psychic/Fairy with SpA 125." in contents
