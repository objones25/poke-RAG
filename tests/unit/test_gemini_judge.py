from __future__ import annotations

import pytest

import scripts.eval.gemini_judge as _judge_mod
from scripts.eval.gemini_judge import JudgeResult, check_claim, check_hallucination, judge_answer


@pytest.mark.unit
class TestCheckClaim:
    def test_returns_true_when_model_says_yes(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        mock_client.models.generate_content.return_value = mocker.MagicMock(text="yes")
        assert check_claim("Pikachu's base Speed is 90.", "Speed: 90") is True

    def test_returns_false_when_model_says_no(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        mock_client.models.generate_content.return_value = mocker.MagicMock(text="no")
        assert check_claim("Pikachu's base Speed is 90.", "Attack: 55") is False

    def test_yes_prefix_match_is_case_insensitive(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        mock_client.models.generate_content.return_value = mocker.MagicMock(text="Yes, it does.")
        assert check_claim("anything", "claim") is True

    def test_non_yes_response_treated_as_false(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        mock_client.models.generate_content.return_value = mocker.MagicMock(text="maybe")
        assert check_claim("anything", "claim") is False

    def test_returns_false_when_text_is_none(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        mock_client.models.generate_content.return_value = mocker.MagicMock(text=None)
        assert check_claim("answer", "claim") is False


@pytest.mark.unit
class TestCheckHallucination:
    def test_returns_empty_list_when_supported(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        mock_client.models.generate_content.return_value = mocker.MagicMock(text="SUPPORTED")
        assert check_hallucination("answer", "context") == []

    def test_returns_empty_list_when_supported_lowercase(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        mock_client.models.generate_content.return_value = mocker.MagicMock(text="supported")
        assert check_hallucination("answer", "context") == []

    def test_returns_list_of_unsupported_claims(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        mock_client.models.generate_content.return_value = mocker.MagicMock(
            text="Pikachu has 100 HP.\nPikachu can learn Surf by default."
        )
        result = check_hallucination("answer", "context")
        assert result == ["Pikachu has 100 HP.", "Pikachu can learn Surf by default."]

    def test_blank_lines_are_stripped(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        mock_client.models.generate_content.return_value = mocker.MagicMock(
            text="Claim one.\n\nClaim two.\n"
        )
        result = check_hallucination("answer", "context")
        assert result == ["Claim one.", "Claim two."]

    def test_returns_empty_list_when_text_is_none(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        mock_client.models.generate_content.return_value = mocker.MagicMock(text=None)
        assert check_hallucination("answer", "context") == []


@pytest.mark.unit
class TestJudgeAnswer:
    def test_aggregates_claim_hits_and_hallucinations(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        responses = [
            mocker.MagicMock(text="yes"),
            mocker.MagicMock(text="no"),
            mocker.MagicMock(text="Fake claim."),
        ]
        mock_client.models.generate_content.side_effect = responses
        result = judge_answer("answer", ["claim A", "claim B"], "context")
        assert isinstance(result, JudgeResult)
        assert result.claim_hits == [True, False]
        assert result.unsupported_claims == ["Fake claim."]

    def test_returns_judge_result_with_all_hits_and_no_hallucinations(self, mocker) -> None:
        mock_client = mocker.MagicMock()
        mocker.patch.object(_judge_mod, "_client", return_value=mock_client)
        responses = [
            mocker.MagicMock(text="yes"),
            mocker.MagicMock(text="SUPPORTED"),
        ]
        mock_client.models.generate_content.side_effect = responses
        result = judge_answer("answer", ["only claim"], "context")
        assert result.claim_hits == [True]
        assert result.unsupported_claims == []
