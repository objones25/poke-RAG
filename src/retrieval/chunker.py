"""Source-aware document chunker for pokeapi, smogon, and bulbapedia files."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from src.types import RetrievedChunk, Source

_LOG = logging.getLogger(__name__)

_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_RE_SMOGON_NAME = re.compile(r"^([\w\s\-'.]+?)\s*\(")
_RE_BULBA_NAME = re.compile(r"^(.+?)\s+\(Pokémon\)\s*$")
_RE_POKEAPI_NAME = re.compile(r"^(.*?)\s+is\s+(?:a|an|the)\s+")
_RE_BULBA_DOC_SPLIT = re.compile(r"\n(?=Title:)")

_SMOGON_TARGET_TOKENS = 400
_BULBA_TARGET_TOKENS = 512
_OVERLAP_RATIO = 0.1
_WORDS_PER_TOKEN = 0.75  # rough approximation: 1 token ≈ 0.75 words


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / _WORDS_PER_TOKEN))


def _split_sentences(text: str) -> list[str]:
    parts = _RE_SENTENCE_SPLIT.split(text.strip())
    return [p for p in parts if p.strip()]


def _merge_into_chunks(segments: list[str], target_tokens: int) -> list[str]:
    """Merge segments into token-bounded chunks with token-based overlap."""
    if not segments:
        return []

    overlap_budget = int(target_tokens * _OVERLAP_RATIO)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = _approx_tokens(seg)
        if current_tokens + seg_tokens > target_tokens and current:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            overlap_tokens = 0
            for s in reversed(current):
                s_tok = _approx_tokens(s)
                if overlap_tokens + s_tok > overlap_budget:
                    break
                overlap.insert(0, s)
                overlap_tokens += s_tok
            current = overlap
            current_tokens = overlap_tokens
        current.append(seg)
        current_tokens += seg_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _recursive_split(text: str, target_tokens: int) -> list[str]:
    """Split text into ≤target_token chunks, trying paragraphs then sentences."""
    stripped = text.strip()
    if not stripped:
        return []
    if _approx_tokens(stripped) <= target_tokens:
        return [stripped]

    paragraphs = [p.strip() for p in stripped.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        merged = _merge_into_chunks(paragraphs, target_tokens)
        if len(merged) > 1:
            return merged

    sentences = _split_sentences(stripped)
    if len(sentences) > 1:
        return _merge_into_chunks(sentences, target_tokens)

    return [stripped]


def _extract_smogon_name(line: str) -> str | None:
    """Extract Pokémon name from 'Name (tier): ...' format."""
    match = _RE_SMOGON_NAME.match(line)
    return match.group(1).strip() if match else None


def _extract_bulbapedia_name(title: str) -> str | None:
    """Extract name from 'Name (Pokémon)' title pattern."""
    match = _RE_BULBA_NAME.match(title)
    return match.group(1).strip() if match else None


def _extract_pokeapi_name(line: str) -> str | None:
    """Extract name from 'Name is a/an/the ...' pattern."""
    match = _RE_POKEAPI_NAME.match(line)
    return match.group(1).strip() if match else None


def chunk_pokeapi_line(line: str, *, doc_id: str) -> list[RetrievedChunk]:
    """Each pokeapi line is already an atomic fact chunk."""
    stripped = line.strip()
    if not stripped:
        return []
    return [
        RetrievedChunk(
            text=stripped,
            score=0.0,
            source="pokeapi",
            pokemon_name=_extract_pokeapi_name(stripped),
            chunk_index=0,
            original_doc_id=doc_id,
        )
    ]


def chunk_smogon_line(line: str, *, doc_id: str) -> list[RetrievedChunk]:
    """Split one smogon entry (possibly long) into token-bounded chunks."""
    stripped = line.strip()
    if not stripped:
        return []

    pokemon_name = _extract_smogon_name(stripped)

    colon_idx = stripped.find(":")
    body = stripped[colon_idx + 1:].strip() if colon_idx != -1 else stripped

    raw_chunks = _recursive_split(body, _SMOGON_TARGET_TOKENS)
    if not raw_chunks:
        return []

    return [
        RetrievedChunk(
            text=chunk_text,
            score=0.0,
            source="smogon",
            pokemon_name=pokemon_name,
            chunk_index=i,
            original_doc_id=doc_id,
        )
        for i, chunk_text in enumerate(raw_chunks)
    ]


def chunk_bulbapedia_doc(doc: str, *, doc_id: str) -> list[RetrievedChunk]:
    """Split one bulbapedia document (Title: header + body) into chunks."""
    stripped = doc.strip()
    if not stripped:
        return []

    lines = stripped.split("\n")
    pokemon_name: str | None = None

    if lines[0].startswith("Title:"):
        title = lines[0][len("Title:"):].strip()
        pokemon_name = _extract_bulbapedia_name(title)
        body = "\n".join(lines[1:]).strip()
    else:
        body = stripped

    if not body:
        return [
            RetrievedChunk(
                text=stripped,
                score=0.0,
                source="bulbapedia",
                pokemon_name=pokemon_name,
                chunk_index=0,
                original_doc_id=doc_id,
            )
        ]

    raw_chunks = _recursive_split(body, _BULBA_TARGET_TOKENS)
    if not raw_chunks:
        return []

    return [
        RetrievedChunk(
            text=chunk_text,
            score=0.0,
            source="bulbapedia",
            pokemon_name=pokemon_name,
            chunk_index=i,
            original_doc_id=doc_id,
        )
        for i, chunk_text in enumerate(raw_chunks)
    ]


def chunk_file(path: Path, *, source: Source) -> list[RetrievedChunk]:
    """Chunk an entire file according to its source format."""
    text = path.read_text(encoding="utf-8")
    chunks: list[RetrievedChunk] = []

    if source == "pokeapi":
        for i, line in enumerate(text.splitlines()):
            chunks.extend(chunk_pokeapi_line(line, doc_id=f"{path.stem}_{i}"))

    elif source == "smogon":
        for i, line in enumerate(text.splitlines()):
            chunks.extend(chunk_smogon_line(line, doc_id=f"{path.stem}_{i}"))

    elif source == "bulbapedia":
        docs = _RE_BULBA_DOC_SPLIT.split(text)
        for i, doc in enumerate(docs):
            doc = doc.strip()
            if doc:
                chunks.extend(chunk_bulbapedia_doc(doc, doc_id=f"{path.stem}_{i}"))

    _LOG.debug("Chunked '%s' (source=%s) → %d chunk(s)", path.name, source, len(chunks))
    return chunks
