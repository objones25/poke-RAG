"""CRAG-style post-retrieval refinement: action triage, strip-level filtering, gap detection."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from src.retrieval.types import RefinementResult
from src.types import RetrievedChunk

if TYPE_CHECKING:
    from src.retrieval.protocols import RerankerProtocol

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_GEN_RE = re.compile(r"\bgen\s*([0-9]+)", re.IGNORECASE)
_TIER_RE = re.compile(r"\b(ou|uu|ru|nu|pu|ubers|lc|vgc|doubles)\b", re.IGNORECASE)

_MIN_SENTENCES_FOR_STRIP_FILTER = 2


class KnowledgeRefiner:
    """Post-retrieval refiner: triage by score, strip-filter, detect constraint gaps."""

    def __init__(
        self,
        reranker: RerankerProtocol,
        *,
        upper_threshold: float = 0.0,
        lower_threshold: float = -3.0,
        strip_threshold: float = -1.0,
    ) -> None:
        if lower_threshold >= upper_threshold:
            raise ValueError(
                f"lower_threshold ({lower_threshold}) must be less than "
                f"upper_threshold ({upper_threshold})"
            )
        self._reranker = reranker
        self._upper = upper_threshold
        self._lower = lower_threshold
        self._strip_threshold = strip_threshold

    # ------------------------------------------------------------------
    # Action triage
    # ------------------------------------------------------------------

    def _triage(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[RetrievedChunk]]:
        """Partition chunks into (accepted, uncertain, dropped) based on reranker score."""
        accepted: list[RetrievedChunk] = []
        uncertain: list[RetrievedChunk] = []
        dropped: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.score >= self._upper:
                accepted.append(chunk)
            elif chunk.score < self._lower:
                dropped.append(chunk)
            else:
                meta = dict(chunk.metadata or {})
                meta["uncertain"] = True
                uncertain.append(replace(chunk, metadata=meta))
        return accepted, uncertain, dropped

    # ------------------------------------------------------------------
    # Strip-level filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = [s.strip() for s in _SENTENCE_RE.split(text)]
        return [p for p in parts if p]

    def _filter_strips(self, query: str, chunk: RetrievedChunk) -> RetrievedChunk | None:
        """Split chunk into sentences, drop low-scoring ones, recompose in original order.

        Returns None if all strips are dropped. Returns chunk unchanged if too short to split.
        """
        sentences = self._split_sentences(chunk.text)
        if len(sentences) < _MIN_SENTENCES_FOR_STRIP_FILTER:
            return chunk

        # chunk_index is temporarily overwritten with sentence position (0, 1, 2…) so
        # we can restore original order after reranker sorts strips by score.  The
        # final replace() bases off the original chunk, which restores the real
        # chunk_index — these temporary strip chunks never escape this method.
        strip_chunks = [
            replace(chunk, text=sent, chunk_index=i) for i, sent in enumerate(sentences)
        ]
        scored = self._reranker.rerank(query, strip_chunks, top_k=len(strip_chunks))
        surviving = sorted(
            (s for s in scored if s.score >= self._strip_threshold),
            key=lambda c: c.chunk_index,
        )
        if not surviving:
            return None
        return replace(chunk, text=" ".join(s.text for s in surviving))

    # ------------------------------------------------------------------
    # Sufficiency check
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_constraint_keywords(query: str) -> list[str]:
        keywords: list[str] = []
        for m in _GEN_RE.finditer(query):
            keywords.append(f"gen{m.group(1)}")
        for m in _TIER_RE.finditer(query):
            keywords.append(m.group(1).lower())
        return keywords

    @staticmethod
    def _covers_keyword(chunk: RetrievedChunk, keyword: str) -> bool:
        """Return True if chunk satisfies the constraint keyword.

        Smogon chunks use metadata fields (generation/tier) when available to avoid
        false matches from cross-gen text references. All other sources use text search.
        """
        meta = chunk.metadata or {}
        if chunk.source == "smogon":
            if keyword.startswith("gen") and "generation" in meta:
                try:
                    return bool(meta["generation"] == int(keyword[3:]))
                except ValueError:
                    return False
            if not keyword.startswith("gen") and "tier" in meta:
                return bool(str(meta["tier"]).lower() == keyword)
        return keyword in chunk.text.lower()

    @staticmethod
    def _check_sufficiency(query: str, chunks: list[RetrievedChunk]) -> list[str]:
        """Return constraint keywords that appear in the query but not in any surviving chunk."""
        keywords = KnowledgeRefiner._extract_constraint_keywords(query)
        if not keywords:
            return []
        return [
            kw for kw in keywords
            if not any(KnowledgeRefiner._covers_keyword(c, kw) for c in chunks)
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def refine(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        constraints: list[str] | None = None,
    ) -> RefinementResult:
        if not chunks:
            return RefinementResult(chunks=(), gaps=())

        accepted, uncertain, _ = self._triage(chunks)

        refined: list[RetrievedChunk] = []
        for chunk in accepted:
            filtered = self._filter_strips(query, chunk)
            if filtered is not None:
                refined.append(filtered)

        refined.extend(uncertain)

        gaps = tuple(self._check_sufficiency(query, refined))
        return RefinementResult(chunks=tuple(refined), gaps=gaps)
