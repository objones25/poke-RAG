"""Formats retrieved chunks into a single context string for the LLM prompt."""

from __future__ import annotations

from src.types import RetrievedChunk

_WORDS_PER_TOKEN = 0.75
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_SEPARATOR = "\n\n---\n\n"


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
            existing = seen.get(chunk.text)
            if existing is None or chunk.score > existing.score:
                seen[chunk.text] = chunk

        emitted: set[str] = set()
        ordered: list[RetrievedChunk] = []
        for chunk in chunks:
            winner = seen[chunk.text]
            if winner.text not in emitted:
                emitted.add(winner.text)
                ordered.append(winner)

        formatted: list[str] = []
        budget = self._max_tokens
        for chunk in ordered:
            block = _format_chunk(chunk)
            tokens = _approx_tokens(block)
            if tokens > budget:
                words_allowed = int(budget * _WORDS_PER_TOKEN)
                words = block.split()
                block = " ".join(words[:words_allowed])
                formatted.append(block)
                break
            formatted.append(block)
            budget -= tokens

        return self._separator.join(formatted)
