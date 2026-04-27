from __future__ import annotations

import pytest

import scripts.eval.run_gen_eval as _eval_mod
from scripts.eval.run_gen_eval import (
    EvalQuestion,
    QuestionResult,
    compute_metrics,
)


def _make_question(**kwargs) -> EvalQuestion:
    defaults = dict(
        id="el001",
        query="What is Pikachu's Speed?",
        category="easy_lexical",
        source_trust="pokeapi",
        reference_answer="Pikachu has 90 Speed.",
        required_claims=["Speed: 90"],
        adversarial=False,
    )
    defaults.update(kwargs)
    return EvalQuestion(**defaults)


def _make_result(
    question: EvalQuestion,
    *,
    claim_hits: list[bool],
    unsupported_claims: list[str],
) -> QuestionResult:
    return QuestionResult(
        question=question,
        answer="some answer",
        claim_hits=claim_hits,
        unsupported_claims=unsupported_claims,
    )


@pytest.mark.unit
class TestComputeMetrics:
    def test_perfect_result(self) -> None:
        q = _make_question(required_claims=["Speed: 90"])
        r = _make_result(q, claim_hits=[True], unsupported_claims=[])
        metrics = compute_metrics([r])
        assert metrics.claim_recall == 1.0
        assert metrics.hallucination_rate == 0.0
        assert metrics.pass_at_1 == 1.0

    def test_missed_claim_lowers_recall(self) -> None:
        q = _make_question(required_claims=["Speed: 90", "Attack: 55"])
        r = _make_result(q, claim_hits=[True, False], unsupported_claims=[])
        metrics = compute_metrics([r])
        assert metrics.claim_recall == pytest.approx(0.5)
        assert metrics.pass_at_1 == 0.0

    def test_hallucination_raises_rate(self) -> None:
        q = _make_question(required_claims=["Speed: 90"])
        r = _make_result(q, claim_hits=[True], unsupported_claims=["Fake claim."])
        metrics = compute_metrics([r])
        assert metrics.hallucination_rate == pytest.approx(1.0)
        assert metrics.pass_at_1 == 0.0

    def test_refusal_rate_counts_refusal_phrases(self) -> None:
        q = _make_question(required_claims=["context does not cover"])
        r = QuestionResult(
            question=q,
            answer="The context does not cover this topic.",
            claim_hits=[True],
            unsupported_claims=[],
        )
        metrics = compute_metrics([r])
        assert metrics.refusal_rate == pytest.approx(1.0)

    def test_per_category_recall(self) -> None:
        q1 = _make_question(
            id="el001", category="easy_lexical", required_claims=["Speed: 90"]
        )
        q2 = _make_question(
            id="ag001",
            category="aggregation",
            required_claims=["HP: 91", "Attack: 134"],
        )
        r1 = _make_result(q1, claim_hits=[True], unsupported_claims=[])
        r2 = _make_result(q2, claim_hits=[True, False], unsupported_claims=[])
        metrics = compute_metrics([r1, r2])
        assert metrics.per_category_recall["easy_lexical"] == pytest.approx(1.0)
        assert metrics.per_category_recall["aggregation"] == pytest.approx(0.5)

    def test_empty_results(self) -> None:
        metrics = compute_metrics([])
        assert metrics.claim_recall == 0.0
        assert metrics.hallucination_rate == 0.0
        assert metrics.refusal_rate == 0.0
        assert metrics.pass_at_1 == 0.0


@pytest.mark.unit
class TestLoadQuestions:
    def test_loads_yaml_and_returns_eval_questions(self, tmp_path) -> None:
        yaml_content = (
            "- id: el001\n"
            "  query: test query\n"
            "  category: easy_lexical\n"
            "  source_trust: pokeapi\n"
            "  reference_answer: test answer\n"
            "  required_claims:\n"
            "    - claim one\n"
            "  adversarial: false\n"
        )
        p = tmp_path / "questions.yaml"
        p.write_text(yaml_content)
        questions = _eval_mod.load_questions(str(p))
        assert len(questions) == 1
        assert questions[0].id == "el001"
        assert questions[0].required_claims == ["claim one"]
        assert questions[0].adversarial is False
