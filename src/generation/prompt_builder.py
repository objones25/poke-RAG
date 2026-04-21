from __future__ import annotations

import unicodedata

from src.types import RetrievedChunk

_SYSTEM_PROMPT = (
    "You are a knowledgeable Pokémon expert. "
    "Answer the question using only the context provided below. "
    "Cite your sources at the end of your answer."
)

_ALLOWED_SOURCES = frozenset({"bulbapedia", "pokeapi", "smogon"})


def _sanitize_for_prompt(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(" " if unicodedata.category(ch)[0] == "C" else ch for ch in normalized).strip()


def build_prompt(query: str, chunks: tuple[RetrievedChunk, ...]) -> str:
    if not query.strip():
        raise ValueError("query must not be empty")
    if not chunks:
        raise ValueError("chunks must not be empty")

    sanitized_query = "".join(
        " " if unicodedata.category(ch)[0] == "C" else ch for ch in query
    ).strip()

    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

    context_parts: list[str] = []
    for chunk in sorted_chunks:
        safe_source = chunk.source if chunk.source in _ALLOWED_SOURCES else "unknown"
        safe_text = _sanitize_for_prompt(chunk.text)
        if chunk.entity_name:
            safe_entity = _sanitize_for_prompt(chunk.entity_name)
            header = f"[Source: {safe_source} | Entity: {safe_entity}]"
        else:
            header = f"[Source: {safe_source}]"
        context_parts.append(f"{header}\n{safe_text}")

    context_block = "\n\n".join(context_parts)
    unique_sources = sorted(
        {c.source if c.source in _ALLOWED_SOURCES else "unknown" for c in chunks}
    )
    sources_line = "Sources: " + ", ".join(unique_sources)

    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_block}\n\n"
        f"{sources_line}\n\n"
        f"Question: {sanitized_query}\n\n"
        f"Answer:"
    )
