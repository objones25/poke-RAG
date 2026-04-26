"""Retrieval evaluation harness — v3.

What changed since v2 (and why):

1. **Resolver uses chunk metadata, not text regex.**
   Chunks expose ``entity_name``, ``entity_type``, ``original_doc_id``, and
   ``source`` fields (visible at the Qdrant payload level — see the chunk
   payload schema). The previous regex-on-text approach silently failed for
   any chunk that wasn't the leading line of an entity (e.g. mid-paragraph
   Smogon fragments), which made multi-hop and aggregation results
   measurement artifacts rather than real signal.

2. **nDCG dedup by canonical ID.**
   When a chunker splits one entity's text into N chunks, all N resolve to
   the same ``<file>:<entity>`` id. v2 summed DCG across each occurrence
   while normalizing by ``min(k, |gold|)``, which let nDCG go > 1.0
   (observed: vr002 = 1.88, lt005 = 2.45). v3 dedups: each gold entity is
   credited at most once, at its earliest rank in the top-k.

3. **Multi-hop is now a tagged capability, not a category.**
   The literature on multi-hop QA (HotpotQA, MuSiQue, 2WikiMultiHopQA) is
   clear: questions whose answer requires composing facts across passages
   cannot be solved by single-pass dense retrieval. Recent agentic-RAG /
   Self-RAG work treats query decomposition as a separate stage. v3 adds
   a ``requires_decomposition`` flag on questions; by default the harness
   reports a separate "decomposition-required" bucket so vanilla retrieval
   numbers aren't dragged down by questions designed to fail.

4. **Context Precision / Context Recall (RAGAS).**
   The current standard for retrieval-only evaluation in RAG (Es et al.,
   2023). Context precision = fraction of top-k that's relevant.
   Context recall = fraction of relevant ground-truth chunks retrieved.
   For our gold-ID-based setup these are precision@k and recall@k with
   relevance defined as gold-ID match — equivalent to RAGAS when the
   judge has full information.

5. **Audit mode.**
   ``--audit`` prints every retrieved chunk's resolved id + first 80 chars
   for each question. Lets you sanity-check whether ``?`` chunks are real
   misses or resolver gaps before you trust the headline numbers.

References:
    - Es et al. RAGAS (2023): https://arxiv.org/abs/2309.15217
    - Thakur et al. BEIR (2021): nDCG@10 standard for retrieval
    - Ho et al. 2WikiMultiHopQA (2020): multi-hop requires composition
    - Asai et al. Self-RAG (2023): retrieval+critique loop for hard queries
    - Gao et al. HyDE (2023): hypothetical-doc embeddings for hard queries

Usage::

    uv run python scripts/eval/run_eval.py
    uv run python scripts/eval/run_eval.py --top-k 30 --verbose
    uv run python scripts/eval/run_eval.py --audit                # dump chunks
    uv run python scripts/eval/run_eval.py --include-decomposition  # don't bucket out multi-hop
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

_QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"


# ---------------------------------------------------------------------------
# Canonical-ID resolution from chunk metadata
# ---------------------------------------------------------------------------
#
# Chunks in this codebase expose at least the following payload fields
# (verified against a production Qdrant chunk):
#
#   payload.text              the chunk text
#   payload.source            "pokeapi" | "smogon" | "bulbapedia"   (high-level)
#   payload.entity_name       "armaldo" | "earthquake" | ...        (lowercased)
#   payload.entity_type       "pokemon" | "move" | "ability" | "item"
#   payload.original_doc_id   "pokemon_moves_1252"  (<file>_<n>)
#   payload.chunk_index       int (sub-chunk position within the doc)
#
# The canonical id we use for gold matching is:
#   <file_prefix>:<entity_name_lower>
# where file_prefix is original_doc_id with the trailing `_<n>` stripped.

_DOC_ID_RX = re.compile(r"^(.+?)_\d+$")


def _read_field(chunk: Any, name: str) -> Any:
    """Try several access patterns; chunks land here as either bare objects
    with attributes, or wrappers with a ``payload`` dict/object."""
    v = getattr(chunk, name, None)
    if v is not None:
        return v
    payload = getattr(chunk, "payload", None)
    if isinstance(payload, dict):
        if name in payload:
            return payload[name]
    elif payload is not None:
        v = getattr(payload, name, None)
        if v is not None:
            return v
    metadata = getattr(chunk, "metadata", None)
    if isinstance(metadata, dict) and name in metadata:
        return metadata[name]
    return None


def resolve_chunk_id(chunk: Any) -> str | None:
    """Return canonical ``<file>:<entity>`` id for a chunk, or None.

    Preferred path is ``original_doc_id`` + ``entity_name`` from metadata.
    If those aren't present, falls back to a text-prefix regex (handles
    chunks with no metadata at all).
    """
    entity = _read_field(chunk, "entity_name")
    doc_id = _read_field(chunk, "original_doc_id")
    if entity and doc_id:
        m = _DOC_ID_RX.match(str(doc_id))
        if m:
            return f"{m.group(1)}:{str(entity).lower()}"

    # Fallback path: regex on the leading text.
    text = _read_field(chunk, "text") or ""
    if not text:
        return None
    head = text.lstrip()[:300]
    for src, pat in _TEXT_FALLBACK_RESOLVERS:
        m = pat.match(head)
        if m:
            return f"{src}:{m.group(1).strip().lower()}"
    return None


_TEXT_FALLBACK_RESOLVERS: list[tuple[str, re.Pattern[str]]] = [
    ("pokemon_species",    re.compile(r"^([A-Z][^.\n]+?) is a (?:[A-Za-z][\w\-]* )*Pokémon\.")),
    ("pokemon_encounters", re.compile(r"^([A-Z][^.\n]+?) is a wild Pokémon found in:")),
    ("move",               re.compile(r"^([A-Z][^.\n]+?) is a Pokémon move\.")),
    ("ability",            re.compile(r"^([A-Z][^.\n]+?) is a Pokémon ability\.")),
    ("item",               re.compile(r"^([A-Z][^.\n]+?) is a Pokémon item\.")),
    ("pokemon",            re.compile(r"^([A-Z][^()]+?) \([A-Z]+ competitive strategy\):")),
    ("pokemon_moves",      re.compile(r"^([A-Z][^.\n]*?) (?:learns|can learn|can hatch)\b")),
]


# ---------------------------------------------------------------------------
# Pure metric functions (importable, unit-testable)
# ---------------------------------------------------------------------------


def _first_seen(ids: Iterable[str | None], gold: set[str]) -> dict[str, int]:
    """Map each gold id to its earliest 1-indexed rank in `ids`."""
    seen: dict[str, int] = {}
    for rank, cid in enumerate(ids, start=1):
        if cid is None:
            continue
        cid_l = cid.lower()
        if cid_l in gold and cid_l not in seen:
            seen[cid_l] = rank
    return seen


def hit_at_k(chunk_ids: list[str | None], *, gold: set[str], k: int) -> bool:
    """Any gold id in the top-k."""
    gold_l = {g.lower() for g in gold}
    return any(cid and cid.lower() in gold_l for cid in chunk_ids[:k])


def recall_at_k(chunk_ids: list[str | None], *, gold: set[str], k: int) -> float:
    """Fraction of gold ids found in the top-k (deduplicated)."""
    if not gold:
        return 0.0
    gold_l = {g.lower() for g in gold}
    found = _first_seen(chunk_ids[:k], gold_l)
    return len(found) / len(gold_l)


def precision_at_k(chunk_ids: list[str | None], *, gold: set[str], k: int) -> float:
    """Fraction of UNIQUE chunks-by-canonical-id in top-k that are in gold.

    Deduplicating by id avoids over-counting when one entity gets multiple
    chunks; that gives a fairer signal of 'how diverse and relevant is the
    top-k as a whole'."""
    if k <= 0:
        return 0.0
    gold_l = {g.lower() for g in gold}
    seen: set[str] = set()
    relevant = 0
    total = 0
    for cid in chunk_ids[:k]:
        key = (cid or "").lower() or f"_anon_{total}"
        if key in seen:
            continue
        seen.add(key)
        total += 1
        if cid and cid.lower() in gold_l:
            relevant += 1
    return relevant / total if total > 0 else 0.0


def mrr_at_k(chunk_ids: list[str | None], *, gold: set[str], k: int) -> float:
    """Reciprocal rank of the first gold hit in top-k (0 if none)."""
    gold_l = {g.lower() for g in gold}
    for rank, cid in enumerate(chunk_ids[:k], start=1):
        if cid and cid.lower() in gold_l:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(chunk_ids: list[str | None], *, gold: set[str], k: int) -> float:
    """Binary-relevance nDCG@k with dedup.

    Each gold entity is credited at most once, at its earliest rank.
    Bounded in [0, 1] (asserted in tests).
    """
    if not gold:
        return 0.0
    gold_l = {g.lower() for g in gold}
    seen = _first_seen(chunk_ids[:k], gold_l)
    dcg = sum(1.0 / math.log2(rank + 1) for rank in seen.values())
    ideal_n = min(k, len(gold_l))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hard_negative_leak_at_k(
    chunk_ids: list[str | None], *, hard_negs: set[str], k: int
) -> float:
    """Fraction of top-k that are flagged hard negatives (with dedup)."""
    if not hard_negs or k <= 0:
        return 0.0
    hn_l = {h.lower() for h in hard_negs}
    seen: set[str] = set()
    leaked = 0
    total = 0
    for cid in chunk_ids[:k]:
        key = (cid or "").lower() or f"_anon_{total}"
        if key in seen:
            continue
        seen.add(key)
        total += 1
        if cid and cid.lower() in hn_l:
            leaked += 1
    return leaked / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _Question:
    id: str
    query: str
    category: str
    difficulty: str
    source_hint: str | None
    gold: set[str]
    hard_negatives: set[str] = field(default_factory=set)
    match: str = "any"
    min_recall: float = 1.0
    requires_decomposition: bool = False
    notes: str | None = None


@dataclass
class _Result:
    qid: str
    category: str
    difficulty: str
    source_hint: str | None
    requires_decomposition: bool
    # core metrics
    hit5: bool
    hit10: bool
    hit20: bool
    recall5: float
    recall10: float
    recall20: float
    precision10: float
    mrr10: float
    ndcg10: float
    passed: bool
    hard_neg_leak10: float
    top_ids: list[str | None]


# ---------------------------------------------------------------------------
# I/O + retriever wiring
# ---------------------------------------------------------------------------


def _load_questions(path: Path) -> list[_Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: list[_Question] = []
    for item in raw:
        out.append(_Question(
            id=item["id"],
            query=item["query"],
            category=item["category"],
            difficulty=item["difficulty"],
            source_hint=item.get("source_hint"),
            gold=set(item.get("gold") or []),
            hard_negatives=set(item.get("hard_negatives") or []),
            match=item.get("match", "any"),
            min_recall=float(item.get("min_recall", 1.0)),
            requires_decomposition=bool(item.get("requires_decomposition", False)),
            notes=item.get("notes"),
        ))
    return out


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
        vector_store=store, embedder=embedder, reranker=reranker,
        query_transformer=transformer,
    )
    return retriever, router


# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------


def _passed(q: _Question, ids: list[str | None], k: int) -> bool:
    if q.match == "any":
        return hit_at_k(ids, gold=q.gold, k=k)
    if q.match == "all":
        return recall_at_k(ids, gold=q.gold, k=k) >= q.min_recall
    raise ValueError(f"unknown match mode: {q.match!r}")


def _score(q: _Question, chunks: list[Any]) -> _Result:
    ids = [resolve_chunk_id(c) for c in chunks]
    return _Result(
        qid=q.id,
        category=q.category,
        difficulty=q.difficulty,
        source_hint=q.source_hint,
        requires_decomposition=q.requires_decomposition,
        hit5=hit_at_k(ids, gold=q.gold, k=5),
        hit10=hit_at_k(ids, gold=q.gold, k=10),
        hit20=hit_at_k(ids, gold=q.gold, k=20),
        recall5=recall_at_k(ids, gold=q.gold, k=5),
        recall10=recall_at_k(ids, gold=q.gold, k=10),
        recall20=recall_at_k(ids, gold=q.gold, k=20),
        precision10=precision_at_k(ids, gold=q.gold, k=10),
        mrr10=mrr_at_k(ids, gold=q.gold, k=10),
        ndcg10=ndcg_at_k(ids, gold=q.gold, k=10),
        passed=_passed(q, ids, k=20),
        hard_neg_leak10=hard_negative_leak_at_k(ids, hard_negs=q.hard_negatives, k=10),
        top_ids=ids[:10],
    )


def _audit_print(q: _Question, chunks: list[Any]) -> None:
    print(f"\n--- AUDIT  {q.id}  [{q.category}/{q.difficulty}]")
    print(f"    query: {q.query}")
    print(f"    gold:  {q.gold}")
    if q.hard_negatives:
        print(f"    hard:  {q.hard_negatives}")
    for i, c in enumerate(chunks[:20], 1):
        cid = resolve_chunk_id(c)
        text = (_read_field(c, "text") or "")[:80].replace("\n", " ")
        marker = ""
        if cid and cid.lower() in {g.lower() for g in q.gold}:
            marker = "  [GOLD]"
        elif cid and cid.lower() in {g.lower() for g in q.hard_negatives}:
            marker = "  [HARD-NEG]"
        print(f"      {i:2d}. {str(cid):40s} | {text}{marker}")


def _run(
    questions: list[_Question], retriever: Any, router: Any,
    *, top_k: int, verbose: bool, audit: bool,
) -> list[_Result]:
    out: list[_Result] = []
    for q in questions:
        try:
            sources = router.route(q.query) if router is not None else None
            retrieval = retriever.retrieve(q.query, top_k=top_k, sources=sources)
            chunks = list(retrieval.documents)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {q.id}: {exc}", file=sys.stderr)
            out.append(_Result(
                qid=q.id, category=q.category, difficulty=q.difficulty,
                source_hint=q.source_hint, requires_decomposition=q.requires_decomposition,
                hit5=False, hit10=False, hit20=False,
                recall5=0.0, recall10=0.0, recall20=0.0,
                precision10=0.0, mrr10=0.0, ndcg10=0.0,
                passed=False, hard_neg_leak10=0.0, top_ids=[],
            ))
            continue

        if audit:
            _audit_print(q, chunks)

        r = _score(q, chunks)
        # Internal sanity assertions (catch v2-style bugs early).
        assert 0.0 <= r.ndcg10 <= 1.0, f"nDCG out of bounds: {r.ndcg10}"
        assert 0.0 <= r.recall10 <= 1.0, f"recall out of bounds: {r.recall10}"
        assert 0.0 <= r.precision10 <= 1.0, f"precision out of bounds: {r.precision10}"
        out.append(r)

        if verbose and not audit:
            tag = "PASS" if r.passed else "FAIL"
            decomp = " (decomp)" if q.requires_decomposition else ""
            print(
                f"  [{tag}] {r.qid} {q.category:12s} {q.difficulty:6s} "
                f"R@10={r.recall10:.2f} nDCG={r.ndcg10:.2f} "
                f"hard={r.hard_neg_leak10:.2f}{decomp}  {q.query[:50]}"
            )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _agg(rs: Iterable[_Result]) -> dict[str, float]:
    rs = list(rs)
    if not rs:
        return {}
    n = len(rs)
    return {
        "n":           n,
        "pass_rate":   sum(r.passed for r in rs) / n,
        "hit@5":       sum(r.hit5 for r in rs) / n,
        "hit@10":      sum(r.hit10 for r in rs) / n,
        "hit@20":      sum(r.hit20 for r in rs) / n,
        "ctx_recall@10":    sum(r.recall10 for r in rs) / n,    # = recall@10
        "ctx_recall@20":    sum(r.recall20 for r in rs) / n,
        "ctx_precision@10": sum(r.precision10 for r in rs) / n,
        "mrr@10":      sum(r.mrr10 for r in rs) / n,
        "ndcg@10":     sum(r.ndcg10 for r in rs) / n,
        "hard_neg@10": sum(r.hard_neg_leak10 for r in rs) / n,
    }


def _print_breakdown(label: str, groups: dict[str, list[_Result]]) -> None:
    print(f"\n{label}:")
    headers = ["bucket", "n", "pass", "hit@5", "hit@10",
               "ctx_R@10", "ctx_P@10", "nDCG@10", "hard@10"]
    print("  " + "  ".join(f"{h:>11}" for h in headers))
    print("  " + "  ".join("-" * 11 for _ in headers))
    for k in sorted(groups):
        a = _agg(groups[k])
        if not a:
            continue
        row = [
            k, f"{a['n']:.0f}",
            f"{a['pass_rate']:.3f}",
            f"{a['hit@5']:.3f}", f"{a['hit@10']:.3f}",
            f"{a['ctx_recall@10']:.3f}", f"{a['ctx_precision@10']:.3f}",
            f"{a['ndcg@10']:.3f}", f"{a['hard_neg@10']:.3f}",
        ]
        print("  " + "  ".join(f"{v:>11}" for v in row))


def _print_report(
    results: list[_Result],
    *,
    breakdowns: list[str],
    bucket_decomposition: bool,
) -> None:
    if not results:
        print("No results.")
        return

    if bucket_decomposition:
        decomp = [r for r in results if r.requires_decomposition]
        single = [r for r in results if not r.requires_decomposition]
        print("\n=== Single-pass retrieval (vanilla expectation) ===")
        a = _agg(single)
        for k in ["n", "pass_rate", "hit@5", "hit@10", "hit@20",
                  "ctx_recall@10", "ctx_precision@10",
                  "mrr@10", "ndcg@10", "hard_neg@10"]:
            v = a.get(k, 0.0)
            if k == "n":
                print(f"  {k:<22} {int(v)}")
            else:
                print(f"  {k:<22} {v:.3f}")
        if decomp:
            print(f"\n=== Decomposition-required ({len(decomp)} questions) ===")
            print("  These need query decomposition / HyDE / iterative retrieval.")
            print("  Vanilla single-pass is expected to fail; report tracked separately.")
            a = _agg(decomp)
            for k in ["pass_rate", "hit@10", "ctx_recall@10", "ndcg@10"]:
                print(f"  {k:<22} {a[k]:.3f}")
        # Use single-pass results for breakdowns by default.
        rs_for_breakdown = single
    else:
        print("\n=== Overall ===")
        a = _agg(results)
        for k in ["n", "pass_rate", "hit@5", "hit@10", "hit@20",
                  "ctx_recall@10", "ctx_precision@10",
                  "mrr@10", "ndcg@10", "hard_neg@10"]:
            v = a.get(k, 0.0)
            if k == "n":
                print(f"  {k:<22} {int(v)}")
            else:
                print(f"  {k:<22} {v:.3f}")
        rs_for_breakdown = results

    keys = {
        "category":   lambda r: r.category,
        "difficulty": lambda r: r.difficulty,
        "source":     lambda r: r.source_hint or "?",
    }
    for b in breakdowns:
        if b not in keys:
            continue
        groups: dict[str, list[_Result]] = defaultdict(list)
        for r in rs_for_breakdown:
            groups[keys[b](r)].append(r)
        _print_breakdown(f"By {b}", dict(groups))

    fails = [r for r in rs_for_breakdown if not r.passed]
    if fails:
        print(f"\nFailures ({len(fails)}):")
        for r in fails:
            top = ", ".join(str(x) for x in r.top_ids[:3])
            print(f"  - {r.qid} [{r.category}/{r.difficulty}] R@10={r.recall10:.2f}  top: {top}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Retrieval eval harness (v3)")
    p.add_argument("--routing", action="store_true", default=False)
    p.add_argument("--no-routing", dest="routing", action="store_false")
    p.add_argument("--hyde", action="store_true", default=False)
    p.add_argument("--no-hyde", dest="hyde", action="store_false")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--audit", action="store_true", default=False,
                   help="Print every retrieved chunk's id + text head per question.")
    p.add_argument("--questions", type=Path, default=_QUESTIONS_PATH)
    p.add_argument("--by", action="append", default=None,
                   choices=["category", "difficulty", "source"])
    p.add_argument(
        "--include-decomposition",
        action="store_true", default=False,
        help="Mix decomposition-required questions into the headline numbers "
             "instead of bucketing them out. Use this when you've enabled "
             "--hyde or have a query rewriter and want to see if those help.",
    )
    args = p.parse_args()

    breakdowns = args.by if args.by else ["category", "difficulty", "source"]

    print(f"Loading questions from {args.questions}")
    qs = _load_questions(args.questions)
    print(f"Loaded {len(qs)} questions  "
          f"({sum(q.requires_decomposition for q in qs)} require decomposition)")
    print(f"Config: routing={args.routing} hyde={args.hyde} top_k={args.top_k}\n")

    retriever, router = _build_retriever(routing=args.routing, hyde=args.hyde)
    results = _run(qs, retriever, router,
                   top_k=args.top_k, verbose=args.verbose, audit=args.audit)
    _print_report(
        results, breakdowns=breakdowns,
        bucket_decomposition=not args.include_decomposition,
    )


if __name__ == "__main__":
    main()