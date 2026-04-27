"""Generation eval runner.

Usage:
    uv run python scripts/eval/run_gen_eval.py
    uv run python scripts/eval/run_gen_eval.py --save-baseline
    uv run python scripts/eval/run_gen_eval.py --baseline 2026-04-27
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import requests
import yaml

from scripts.eval.gemini_judge import JudgeResult, judge_answer

_QUESTIONS_PATH = Path(__file__).parent / "gen_questions.yaml"
_BASELINES_DIR = Path(__file__).parent / "baselines"
_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
_REFUSAL_PHRASES = ("context does not cover", "not in the context", "cannot answer")


@dataclass
class EvalQuestion:
    id: str
    query: str
    category: str
    source_trust: str
    reference_answer: str
    required_claims: list[str]
    adversarial: bool


@dataclass
class QuestionResult:
    question: EvalQuestion
    answer: str
    claim_hits: list[bool]
    unsupported_claims: list[str]

    @property
    def pass_at_1(self) -> bool:
        return all(self.claim_hits) and not self.unsupported_claims

    @property
    def is_refusal(self) -> bool:
        lower = self.answer.lower()
        return any(phrase in lower for phrase in _REFUSAL_PHRASES)


@dataclass
class EvalMetrics:
    claim_recall: float
    hallucination_rate: float
    refusal_rate: float
    pass_at_1: float
    per_category_recall: dict[str, float] = field(default_factory=dict)
    n_questions: int = 0


def load_questions(path: str = str(_QUESTIONS_PATH)) -> list[EvalQuestion]:
    data = yaml.safe_load(Path(path).read_text())
    return [EvalQuestion(**q) for q in data]


def query_api(query: str) -> tuple[str, str]:
    resp = requests.post(f"{_API_URL}/query", json={"query": query}, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    return body["answer"], body.get("context") or ""


def compute_metrics(results: list[QuestionResult]) -> EvalMetrics:
    if not results:
        return EvalMetrics(
            claim_recall=0.0,
            hallucination_rate=0.0,
            refusal_rate=0.0,
            pass_at_1=0.0,
            n_questions=0,
        )

    total_claims = sum(len(r.claim_hits) for r in results)
    total_hits = sum(sum(h for h in r.claim_hits) for r in results)
    claim_recall = total_hits / total_claims if total_claims else 0.0

    hallucination_rate = sum(1 for r in results if r.unsupported_claims) / len(results)
    refusal_rate = sum(1 for r in results if r.is_refusal) / len(results)
    pass_at_1 = sum(1 for r in results if r.pass_at_1) / len(results)

    by_category: dict[str, list[QuestionResult]] = {}
    for r in results:
        by_category.setdefault(r.question.category, []).append(r)

    per_category_recall: dict[str, float] = {}
    for cat, cat_results in by_category.items():
        cat_total = sum(len(r.claim_hits) for r in cat_results)
        cat_hits = sum(sum(h for h in r.claim_hits) for r in cat_results)
        per_category_recall[cat] = cat_hits / cat_total if cat_total else 0.0

    return EvalMetrics(
        claim_recall=claim_recall,
        hallucination_rate=hallucination_rate,
        refusal_rate=refusal_rate,
        pass_at_1=pass_at_1,
        per_category_recall=per_category_recall,
        n_questions=len(results),
    )


def print_metrics(metrics: EvalMetrics, baseline: EvalMetrics | None = None) -> None:
    def delta(new: float, old: float | None) -> str:
        if old is None:
            return ""
        diff = new - old
        sign = "+" if diff >= 0 else ""
        return f"  ({sign}{diff:.3f})"

    b = baseline
    print(f"\n=== Generation Eval Results ({metrics.n_questions} questions) ===")
    print(
        f"  claim_recall:       {metrics.claim_recall:.3f}"
        f"{delta(metrics.claim_recall, b.claim_recall if b else None)}"
    )
    print(
        f"  hallucination_rate: {metrics.hallucination_rate:.3f}"
        f"{delta(metrics.hallucination_rate, b.hallucination_rate if b else None)}"
    )
    print(
        f"  refusal_rate:       {metrics.refusal_rate:.3f}"
        f"{delta(metrics.refusal_rate, b.refusal_rate if b else None)}"
    )
    print(
        f"  pass@1:             {metrics.pass_at_1:.3f}"
        f"{delta(metrics.pass_at_1, b.pass_at_1 if b else None)}"
    )
    print("\n  Per-category claim recall:")
    for cat, recall in sorted(metrics.per_category_recall.items()):
        b_cat = b.per_category_recall.get(cat) if b else None
        print(f"    {cat:<20} {recall:.3f}{delta(recall, b_cat)}")


def save_baseline(metrics: EvalMetrics, tag: str | None = None) -> Path:
    _BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    tag = tag or date.today().isoformat()
    path = _BASELINES_DIR / f"{tag}.json"
    payload = {
        "claim_recall": metrics.claim_recall,
        "hallucination_rate": metrics.hallucination_rate,
        "refusal_rate": metrics.refusal_rate,
        "pass_at_1": metrics.pass_at_1,
        "per_category_recall": metrics.per_category_recall,
        "n_questions": metrics.n_questions,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_baseline(tag: str) -> EvalMetrics:
    path = _BASELINES_DIR / f"{tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"No baseline found at {path}")
    data = json.loads(path.read_text())
    return EvalMetrics(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation eval against live API")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--baseline", metavar="YYYY-MM-DD", default=None)
    args = parser.parse_args()

    questions = load_questions()
    results: list[QuestionResult] = []

    for q in questions:
        print(f"  {q.id}: querying...", file=sys.stderr, end=" ")
        try:
            answer, context = query_api(q.query)
        except Exception as exc:
            print(f"ERROR ({exc})", file=sys.stderr)
            continue
        print("judging...", file=sys.stderr, end=" ")
        judge: JudgeResult = judge_answer(answer, q.required_claims, context)
        results.append(
            QuestionResult(
                question=q,
                answer=answer,
                claim_hits=judge.claim_hits,
                unsupported_claims=judge.unsupported_claims,
            )
        )
        status = "PASS" if results[-1].pass_at_1 else "FAIL"
        print(status, file=sys.stderr)

    metrics = compute_metrics(results)
    baseline: EvalMetrics | None = None
    if args.baseline:
        baseline = load_baseline(args.baseline)

    print_metrics(metrics, baseline)

    if args.save_baseline:
        path = save_baseline(metrics)
        print(f"\nBaseline saved to {path}")


if __name__ == "__main__":
    main()
