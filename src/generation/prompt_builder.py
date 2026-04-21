from __future__ import annotations

from src.types import RetrievedChunk

_SYSTEM_PROMPT = (
    "You are a knowledgeable Pokémon expert. "
    "Answer the question using only the context provided below. "
    "Cite your sources at the end of your answer."
)


def build_prompt(query: str, chunks: tuple[RetrievedChunk, ...]) -> str:
    if not query.strip():
        raise ValueError("query must not be empty")
    if not chunks:
        raise ValueError("chunks must not be empty")

    sanitized_query = query.replace("\n", " ")

    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

    context_parts: list[str] = []
    for chunk in sorted_chunks:
        if chunk.entity_name:
            header = f"[Source: {chunk.source} | Entity: {chunk.entity_name}]"
        else:
            header = f"[Source: {chunk.source}]"
        context_parts.append(f"{header}\n{chunk.text}")

    context_block = "\n\n".join(context_parts)
    unique_sources = sorted({c.source for c in chunks})
    sources_line = "Sources: " + ", ".join(unique_sources)

    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_block}\n\n"
        f"{sources_line}\n\n"
        f"Question: {sanitized_query}\n\n"
        f"Answer:"
    )
