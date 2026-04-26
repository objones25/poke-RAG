"""Retrieval evaluation harness.

Runs a YAML question set against the live retriever and reports:
  Hit@5, Hit@10, Hit@20, MRR@10

Usage::

    uv run python scripts/eval/run_eval.py
    uv run python scripts/eval/run_eval.py --routing --no-hyde --top-k 20
    uv run python scripts/eval/run_eval.py --verbose

Requires a live Qdrant connection (QDRANT_URL / QDRANT_API_KEY env vars).
Does NOT load the generator — no GPU required.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml  # type: ignore[import-untyped]

_QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"


# ---------------------------------------------------------------------------
# Pure metric functions — importable by unit tests
# ---------------------------------------------------------------------------


class _HasText(Protocol):
    text: str


def is_relevant(text: str, keywords: list[str]) -> bool:
    """True if any keyword appears in text (case-insensitive substring)."""
    if not text or not keywords:
        return False
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def hit_at_k(chunks: list[_HasText], *, keywords: list[str], k: int) -> bool:
    """True if any of the top-k chunks is relevant."""
    return any(is_relevant(c.text, keywords) for c in chunks[:k])


def mrr_at_k(chunks: list[_HasText], *, keywords: list[str], k: int) -> float:
    """Reciprocal rank of the first relevant chunk within top-k (0 if none)."""
    for rank, chunk in enumerate(chunks[:k], start=1):
        if is_relevant(chunk.text, keywords):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class _Question:
    query: str
    expected_keywords: list[str]
    source_hint: str | None = None


@dataclass
class _Result:
    query: str
    hit5: bool
    hit10: bool
    hit20: bool
    mrr10: float
    top_sources: list[str]
    source_hint: str | None = None


def _load_questions(path: Path) -> list[_Question]:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return [
        _Question(
            query=item["query"],
            expected_keywords=item["expected_keywords"],
            source_hint=item.get("source_hint"),
        )
        for item in raw
    ]


def _build_retriever(*, routing: bool, hyde: bool) -> tuple[Any, Any]:
    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("ROUTING_ENABLED", "true" if routing else "false")
    os.environ.setdefault("HYDE_ENABLED", "true" if hyde else "false")

    from qdrant_client import QdrantClient

    from src.config import Settings
    from src.retrieval.embedder import BGEEmbedder
    from src.retrieval.query_router import QueryRouter
    from src.retrieval.query_transformer import PassthroughTransformer
    from src.retrieval.reranker import BGEReranker
    from src.retrieval.retriever import Retriever
    from src.retrieval.vector_store import QdrantVectorStore

    settings = Settings.from_env()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    client = QdrantClient(url=str(settings.qdrant_url), api_key=api_key)
    embedder = BGEEmbedder.from_pretrained(model_name=settings.embed_model, device=settings.device)
    store = QdrantVectorStore(client=client)
    reranker = BGEReranker.from_pretrained(model_name=settings.rerank_model, device=settings.device)
    router = QueryRouter() if routing else None
    transformer = PassthroughTransformer()

    retriever = Retriever(
        vector_store=store,
        embedder=embedder,
        reranker=reranker,
        query_transformer=transformer,
    )
    return retriever, router


def _run(
    questions: list[_Question], retriever: Any, router: Any, top_k: int, verbose: bool
) -> list[_Result]:

    results: list[_Result] = []
    for q in questions:
        try:
            sources = router.route(q.query) if router is not None else None
            retrieval = retriever.retrieve(q.query, top_k=top_k, sources=sources)
            chunks = list(retrieval.documents)
        except Exception as exc:
            print(f"  ERROR retrieving '{q.query[:50]}': {exc}", file=sys.stderr)
            results.append(
                _Result(
                    query=q.query,
                    hit5=False,
                    hit10=False,
                    hit20=False,
                    mrr10=0.0,
                    top_sources=[],
                )
            )
            continue

        kw = q.expected_keywords
        result = _Result(
            query=q.query,
            hit5=hit_at_k(chunks, keywords=kw, k=5),
            hit10=hit_at_k(chunks, keywords=kw, k=10),
            hit20=hit_at_k(chunks, keywords=kw, k=20),
            mrr10=mrr_at_k(chunks, keywords=kw, k=10),
            top_sources=[c.source for c in chunks[:5]],
            source_hint=q.source_hint,
        )
        results.append(result)

        if verbose:
            status = "HIT5" if result.hit5 else ("HIT10" if result.hit10 else "MISS")
            print(f"  [{status}] MRR={result.mrr10:.2f}  {q.query[:60]}")

    return results


def _print_table(results: list[_Result]) -> None:
    n = len(results)
    if n == 0:
        print("No results.")
        return

    hit5 = sum(r.hit5 for r in results) / n
    hit10 = sum(r.hit10 for r in results) / n
    hit20 = sum(r.hit20 for r in results) / n
    mrr10 = sum(r.mrr10 for r in results) / n

    print("\n| Metric    | Score  |")
    print("|-----------|--------|")
    print(f"| Hit@5     | {hit5:.3f}  |")
    print(f"| Hit@10    | {hit10:.3f}  |")
    print(f"| Hit@20    | {hit20:.3f}  |")
    print(f"| MRR@10    | {mrr10:.3f}  |")
    print(f"| Questions | {n}      |")

    # Per-source breakdown
    by_source: dict[str, list[_Result]] = {}
    for r in results:
        key = r.source_hint or "unknown"
        by_source.setdefault(key, []).append(r)

    if len(by_source) > 1:
        print("\nPer-source Hit@5:")
        for src in sorted(by_source):
            grp = by_source[src]
            h5 = sum(r.hit5 for r in grp) / len(grp)
            print(f"  {src:<12} {h5:.3f}  ({len(grp)} questions)")

    misses = [r for r in results if not r.hit20]
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for r in misses:
            print(f"  - {r.query[:80]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval eval harness")
    parser.add_argument("--routing", action="store_true", default=False)
    parser.add_argument("--no-routing", dest="routing", action="store_false")
    parser.add_argument("--hyde", action="store_true", default=False)
    parser.add_argument("--no-hyde", dest="hyde", action="store_false")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--questions",
        type=Path,
        default=_QUESTIONS_PATH,
        help="Path to questions YAML",
    )
    args = parser.parse_args()

    print(f"Loading questions from {args.questions}")
    questions = _load_questions(args.questions)
    print(f"Loaded {len(questions)} questions")
    print(f"Config: routing={args.routing} hyde={args.hyde} top_k={args.top_k}\n")

    retriever, router = _build_retriever(routing=args.routing, hyde=args.hyde)

    results = _run(questions, retriever, router, top_k=args.top_k, verbose=args.verbose)
    _print_table(results)


if __name__ == "__main__":
    main()
