"""Formats retrieved chunks into a single context string for the LLM prompt."""

from __future__ import annotations

import math

from src.retrieval.constants import WORDS_PER_TOKEN as _WORDS_PER_TOKEN
from src.types import RetrievedChunk

_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_SEPARATOR = "\n\n---\n\n"
_SAFETY_MARGIN = 0.95


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) / _WORDS_PER_TOKEN)


def _format_chunk(chunk: RetrievedChunk) -> str:
    parts = [f"[Source: {chunk.source}"]
    if chunk.entity_name is not None:
        parts.append(f" | Entity: {chunk.entity_name}")
    parts.append("]\n")
    return "".join(parts) + chunk.text


class ContextAssembler:
    def __init__(
        self,
        *,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        separator: str = _DEFAULT_SEPARATOR,
    ) -> None:
        self._max_tokens = max_tokens
        self._separator = separator

    def assemble(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return ""

        seen: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            key = f"{chunk.original_doc_id}:{chunk.chunk_index}"
            existing = seen.get(key)
            if existing is None or chunk.score > existing.score:
                seen[key] = chunk

        emitted: set[str] = set()
        ordered: list[RetrievedChunk] = []
        for chunk in chunks:
            key = f"{chunk.original_doc_id}:{chunk.chunk_index}"
            winner = seen[key]
            if key not in emitted:
                emitted.add(key)
                ordered.append(winner)

        formatted: list[str] = []
        budget = self._max_tokens
        for chunk in ordered:
            block = _format_chunk(chunk)
            tokens = _approx_tokens(block)
            if tokens > budget:
                words_allowed = math.floor(budget * _WORDS_PER_TOKEN * _SAFETY_MARGIN)
                if words_allowed == 0:
                    break
                words = block.split()
                block = " ".join(words[:words_allowed])
                formatted.append(block)
                break
            formatted.append(block)
            budget -= tokens

        return self._separator.join(formatted)
