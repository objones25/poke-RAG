"""Source-aware document chunker for pokeapi, smogon, and bulbapedia files."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.retrieval.constants import WORDS_PER_TOKEN as _WORDS_PER_TOKEN
from src.types import EntityType, RetrievedChunk, Source

_LOG = logging.getLogger(__name__)

_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_RE_SMOGON_NAME = re.compile(r"^([\w\s\-'.]+?)\s*\(")
_RE_POKEAPI_NAME = re.compile(
    r"^(.*?)\s+(?:is\s+(?:a|an|the)|(?:learns|can\s+learn|can\s+hatch)\s+)\s*"
)
_RE_BULBA_DOC_SPLIT = re.compile(r"\n(?=Title:)")

# smogon_data.txt structured-format patterns
_RE_SD_POKE_SEP = re.compile(r"^={80}$", re.MULTILINE)
_RE_SD_FMT_SEP = re.compile(r"^-{40}$", re.MULTILINE)
_RE_SD_SMOGON_FORM = re.compile(r"^Smogon form:\s*(.+)$", re.MULTILINE)
_RE_SD_FORMAT_NAME = re.compile(r"^\s*Format:\s*(\S+)\s*$", re.MULTILINE)
_RE_SD_SECTION = re.compile(r"^\[\s*(Overview|Set:\s*[^\]]+?)\s*\]\s*$", re.MULTILINE)
_RE_SD_MOVES_HEADER = re.compile(r"^\s*Moves:\s*$", re.MULTILINE)
_RE_SD_DESC_HEADER = re.compile(r"^\s*Description:\s*$", re.MULTILINE)
_RE_SD_ATTR = re.compile(r"^\s*(Tera Type|Item|Ability|Nature|EVs|IVs):\s*(.+)$", re.MULTILINE)
_RE_SD_MOVE_ITEM = re.compile(r"^\s*-\s*(.+)$", re.MULTILINE)
_RE_SD_GEN_TIER = re.compile(r"^gen(\d+)(.+)$")
# Strips trailing _\d+ (or _aug_\d+) from a doc_id to get the stem
_RE_POKEAPI_STEM = re.compile(r"(?:_aug)?_\d+$")

_SMOGON_TARGET_TOKENS = 400
_BULBA_TARGET_TOKENS = 400
_OVERLAP_RATIO = 0.1


_STEM_TO_ENTITY_TYPE: dict[str, EntityType] = {
    "ability": "ability",
    "item": "item",
    "move": "move",
    "pokemon": "pokemon",
    "pokemon_species": "pokemon",
    "pokemon_moves": "pokemon",
    "pokemon_encounters": "pokemon",
    "format": "format",
    "formats": "format",
    "smogon_data": "pokemon",
}


def _entity_type_from_stem(stem: str) -> EntityType | None:
    return _STEM_TO_ENTITY_TYPE.get(stem)


def _approx_tokens(text: str, *, tokenize_fn: Callable[[str], int] | None = None) -> int:
    if tokenize_fn is not None:
        return tokenize_fn(text)
    return max(1, int(len(text.split()) / _WORDS_PER_TOKEN))


def _split_sentences(text: str) -> list[str]:
    parts = _RE_SENTENCE_SPLIT.split(text.strip())
    return [p for p in parts if p.strip()]


def _merge_into_chunks(
    segments: list[str],
    target_tokens: int,
    *,
    tokenize_fn: Callable[[str], int] | None = None,
) -> list[str]:
    """Merge segments into token-bounded chunks with token-based overlap."""
    if not segments:
        return []

    overlap_budget = int(target_tokens * _OVERLAP_RATIO)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = _approx_tokens(seg, tokenize_fn=tokenize_fn)
        if current_tokens + seg_tokens > target_tokens and current:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            overlap_tokens = 0
            for s in reversed(current):
                s_tok = _approx_tokens(s, tokenize_fn=tokenize_fn)
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


def _recursive_split(
    text: str,
    target_tokens: int,
    *,
    tokenize_fn: Callable[[str], int] | None = None,
) -> list[str]:
    """Split text into ≤target_token chunks, trying paragraphs then sentences."""
    stripped = text.strip()
    if not stripped:
        return []
    if _approx_tokens(stripped, tokenize_fn=tokenize_fn) <= target_tokens:
        return [stripped]

    paragraphs = [p.strip() for p in stripped.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        merged = _merge_into_chunks(paragraphs, target_tokens, tokenize_fn=tokenize_fn)
        if len(merged) > 1:
            return merged

    sentences = _split_sentences(stripped)
    if len(sentences) > 1:
        return _merge_into_chunks(sentences, target_tokens, tokenize_fn=tokenize_fn)

    return [stripped]


def _extract_smogon_metadata(
    *,
    format_name: str | None,
    chunk_kind: str | None = None,
    set_name: str | None = None,
    attrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build metadata dict for a smogon_data chunk."""
    if format_name is None:
        return {}
    meta: dict[str, Any] = {"format_name": format_name}
    gen_match = _RE_SD_GEN_TIER.match(format_name)
    if gen_match:
        meta["generation"] = int(gen_match.group(1))
        meta["tier"] = gen_match.group(2)
    if chunk_kind is not None:
        meta["chunk_kind"] = chunk_kind
    if chunk_kind == "set" and set_name is not None:
        meta["set_name"] = set_name
    if attrs:
        _ATTR_KEY_MAP = {
            "Tera Type": "tera_type",
            "Item": "item",
            "Ability": "ability",
            "Nature": "nature",
        }
        for src_key, dst_key in _ATTR_KEY_MAP.items():
            if src_key in attrs:
                meta[dst_key] = attrs[src_key]
    return meta


def _extract_pokeapi_metadata(*, doc_id: str) -> dict[str, Any]:
    """Build metadata dict for a pokeapi chunk from its doc_id."""
    stem_match = _RE_POKEAPI_STEM.search(doc_id)
    if not stem_match:
        return {}
    stem = doc_id[: stem_match.start()]
    _STEM_TO_SUBTYPE: dict[str, str] = {
        "pokemon_species": "species",
        "pokemon_moves": "moves",
        "pokemon_encounters": "encounters",
        "ability": "ability",
        "item": "item",
        "move": "move",
    }
    subtype = _STEM_TO_SUBTYPE.get(stem)
    if subtype is None:
        return {}
    return {"entity_subtype": subtype}


def _extract_bulbapedia_metadata() -> dict[str, Any]:
    """Return an empty metadata stub; topics are populated offline by the Gemini utility."""
    return {}


def _extract_smogon_name(line: str) -> str | None:
    """Extract entity name from 'Name (tier): ...' format."""
    match = _RE_SMOGON_NAME.match(line)
    return match.group(1).strip() if match else None


def _extract_bulbapedia_name(title: str) -> str | None:
    """Extract entity name from bulbapedia title (everything before the first '(')."""
    if "(" in title:
        name = title[: title.index("(")].strip()
        return name or None
    return title.strip() or None


def _extract_pokeapi_name(line: str) -> str | None:
    """Extract name from 'Name is a/an/the ...' pattern."""
    match = _RE_POKEAPI_NAME.match(line)
    return match.group(1).strip() if match else None


def chunk_pokeapi_line(
    line: str,
    *,
    doc_id: str,
    entity_type: EntityType | None = None,
) -> list[RetrievedChunk]:
    """Each pokeapi line is already an atomic fact chunk."""
    stripped = line.strip()
    if not stripped:
        return []
    return [
        RetrievedChunk(
            text=stripped,
            score=0.0,
            source="pokeapi",
            entity_name=_extract_pokeapi_name(stripped),
            entity_type=entity_type,
            chunk_index=0,
            original_doc_id=doc_id,
            metadata=_extract_pokeapi_metadata(doc_id=doc_id),
        )
    ]


def chunk_smogon_line(
    line: str,
    *,
    doc_id: str,
    entity_type: EntityType | None = None,
    tokenize_fn: Callable[[str], int] | None = None,
) -> list[RetrievedChunk]:
    """Split one smogon entry (possibly long) into token-bounded chunks."""
    stripped = line.strip()
    if not stripped:
        return []

    entity_name = _extract_smogon_name(stripped)

    colon_idx = stripped.find(":")
    body = stripped[colon_idx + 1 :].strip() if colon_idx != -1 else stripped

    raw_chunks = _recursive_split(body, _SMOGON_TARGET_TOKENS, tokenize_fn=tokenize_fn)
    if not raw_chunks:
        return []

    prefix = f"{entity_name}: " if entity_name else ""
    smogon_meta = _extract_smogon_metadata(format_name=None)
    return [
        RetrievedChunk(
            text=f"{prefix}{chunk_text}",
            score=0.0,
            source="smogon",
            entity_name=entity_name,
            entity_type=entity_type,
            chunk_index=i,
            original_doc_id=doc_id,
            metadata=smogon_meta,
        )
        for i, chunk_text in enumerate(raw_chunks)
    ]


def chunk_bulbapedia_doc(
    doc: str,
    *,
    doc_id: str,
    entity_type: EntityType | None = None,
    tokenize_fn: Callable[[str], int] | None = None,
) -> list[RetrievedChunk]:
    """Split one bulbapedia document (Title: header + body) into chunks."""
    stripped = doc.strip()
    if not stripped:
        return []

    lines = stripped.split("\n")
    entity_name: str | None = None

    if lines[0].startswith("Title:"):
        title = lines[0][len("Title:") :].strip()
        entity_name = _extract_bulbapedia_name(title)
        body = "\n".join(lines[1:]).strip()
    else:
        body = stripped

    bulba_meta = _extract_bulbapedia_metadata()
    if not body:
        return [
            RetrievedChunk(
                text=stripped,
                score=0.0,
                source="bulbapedia",
                entity_name=entity_name,
                entity_type=entity_type,
                chunk_index=0,
                original_doc_id=doc_id,
                metadata=bulba_meta,
            )
        ]

    raw_chunks = _recursive_split(body, _BULBA_TARGET_TOKENS, tokenize_fn=tokenize_fn)
    if not raw_chunks:
        return []

    return [
        RetrievedChunk(
            text=chunk_text,
            score=0.0,
            source="bulbapedia",
            entity_name=entity_name,
            entity_type=entity_type,
            chunk_index=i,
            original_doc_id=doc_id,
            metadata=bulba_meta,
        )
        for i, chunk_text in enumerate(raw_chunks)
    ]


def _slugify(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name.lower()).strip("_")


def _parse_set_body(
    body: str,
) -> tuple[dict[str, str], list[str], str]:
    """Return (attributes, moves, description) parsed from a set section body."""
    attrs: dict[str, str] = {}
    moves: list[str] = []
    description = ""

    moves_match = _RE_SD_MOVES_HEADER.search(body)
    desc_match = _RE_SD_DESC_HEADER.search(body)

    attr_end = (
        moves_match.start() if moves_match else (desc_match.start() if desc_match else len(body))
    )
    for m in _RE_SD_ATTR.finditer(body[:attr_end]):
        attrs[m.group(1)] = m.group(2).strip()

    if moves_match:
        moves_end = desc_match.start() if desc_match else len(body)
        for m in _RE_SD_MOVE_ITEM.finditer(body[moves_match.end() : moves_end]):
            moves.append(m.group(1).strip())

    if desc_match:
        description = body[desc_match.end() :].strip()

    return attrs, moves, description


def _make_set_header(
    entity_name: str,
    format_name: str,
    set_name: str,
    attrs: dict[str, str],
    moves: list[str],
) -> str:
    parts = [f"{entity_name} in {format_name} — Set: {set_name}"]
    if attrs:
        parts.append(" | ".join(f"{k}: {v}" for k, v in attrs.items()))
    if moves:
        parts.append(f"Moves: {', '.join(moves)}")
    return "\n".join(parts)


def chunk_smogon_data_file(
    text: str,
    *,
    tokenize_fn: Callable[[str], int] | None = None,
) -> list[RetrievedChunk]:
    """Parse the multi-block smogon_data.txt format into RetrievedChunks.

    File structure:
      ={80} line  ← Pokémon block separator
      header (name + "Smogon form: <name>" line)
      ={80} line
      body: one or more format sections, each delimited by -{40} lines
        Format header line: " Format: <name>"
        [ Overview ]  and/or  [ Set: <name> ]  sections
    """
    chunks: list[RetrievedChunk] = []
    poke_parts = _RE_SD_POKE_SEP.split(text)
    # poke_parts layout: [pre, header, body, header, body, ...]
    i = 1
    while i + 1 < len(poke_parts):
        header_block = poke_parts[i]
        body_block = poke_parts[i + 1]
        i += 2

        form_match = _RE_SD_SMOGON_FORM.search(header_block)
        if not form_match:
            continue
        entity_name = form_match.group(1).strip()
        entity_slug = _slugify(entity_name)

        fmt_parts = _RE_SD_FMT_SEP.split(body_block)
        # fmt_parts layout: [pre, fmt_header, content, fmt_header, content, ...]
        j = 1
        while j + 1 < len(fmt_parts):
            fmt_header = fmt_parts[j]
            fmt_content = fmt_parts[j + 1]
            j += 2

            fmt_match = _RE_SD_FORMAT_NAME.search(fmt_header)
            if not fmt_match:
                continue
            format_name = fmt_match.group(1).strip()

            section_parts = _RE_SD_SECTION.split(fmt_content)
            # section_parts: [pre, section_label, body, section_label, body, ...]
            k = 1
            while k + 1 < len(section_parts):
                section_label = section_parts[k].strip()
                section_body = section_parts[k + 1]
                k += 2

                if section_label == "Overview":
                    overview_text = section_body.strip()
                    if not overview_text:
                        continue
                    header = f"{entity_name} in {format_name} — Overview"
                    doc_id = f"smogon_data_{entity_slug}_{format_name}_overview"
                    full_text = f"{header}\n{overview_text}"
                    overview_meta = _extract_smogon_metadata(
                        format_name=format_name, chunk_kind="overview"
                    )
                    if _approx_tokens(full_text, tokenize_fn=tokenize_fn) <= _SMOGON_TARGET_TOKENS:
                        chunks.append(
                            RetrievedChunk(
                                text=full_text,
                                score=0.0,
                                source="smogon",
                                entity_name=entity_name,
                                entity_type="pokemon",
                                chunk_index=0,
                                original_doc_id=doc_id,
                                metadata=overview_meta,
                            )
                        )
                    else:
                        header_tokens = _approx_tokens(header, tokenize_fn=tokenize_fn)
                        sub_texts = _recursive_split(
                            overview_text,
                            _SMOGON_TARGET_TOKENS - header_tokens,
                            tokenize_fn=tokenize_fn,
                        )
                        for ci, sub in enumerate(sub_texts):
                            chunks.append(
                                RetrievedChunk(
                                    text=f"{header}\n{sub}",
                                    score=0.0,
                                    source="smogon",
                                    entity_name=entity_name,
                                    entity_type="pokemon",
                                    chunk_index=ci,
                                    original_doc_id=doc_id,
                                    metadata=overview_meta,
                                )
                            )

                elif section_label.startswith("Set:"):
                    set_name = section_label[len("Set:") :].strip()
                    attrs, moves, description = _parse_set_body(section_body)
                    set_header = _make_set_header(entity_name, format_name, set_name, attrs, moves)
                    doc_id = f"smogon_data_{entity_slug}_{format_name}_set_{_slugify(set_name)}"
                    full_text = (
                        f"{set_header}\n{description}".strip() if description else set_header
                    )
                    set_meta = _extract_smogon_metadata(
                        format_name=format_name,
                        chunk_kind="set",
                        set_name=set_name,
                        attrs=attrs,
                    )
                    if _approx_tokens(full_text, tokenize_fn=tokenize_fn) <= _SMOGON_TARGET_TOKENS:
                        chunks.append(
                            RetrievedChunk(
                                text=full_text,
                                score=0.0,
                                source="smogon",
                                entity_name=entity_name,
                                entity_type="pokemon",
                                chunk_index=0,
                                original_doc_id=doc_id,
                                metadata=set_meta,
                            )
                        )
                    else:
                        header_tokens = _approx_tokens(set_header, tokenize_fn=tokenize_fn)
                        desc_subs = _recursive_split(
                            description,
                            _SMOGON_TARGET_TOKENS - header_tokens,
                            tokenize_fn=tokenize_fn,
                        )
                        for ci, sub in enumerate(desc_subs):
                            chunks.append(
                                RetrievedChunk(
                                    text=f"{set_header}\n{sub}",
                                    score=0.0,
                                    source="smogon",
                                    entity_name=entity_name,
                                    entity_type="pokemon",
                                    chunk_index=ci,
                                    original_doc_id=doc_id,
                                    metadata=set_meta,
                                )
                            )

    return chunks


def chunk_file(
    path: Path,
    *,
    source: Source,
    tokenize_fn: Callable[[str], int] | None = None,
) -> list[RetrievedChunk]:
    """Chunk an entire file according to its source format."""
    text = path.read_text(encoding="utf-8")
    entity_type = _entity_type_from_stem(path.stem)
    chunks: list[RetrievedChunk] = []

    if source == "pokeapi":
        for i, line in enumerate(text.splitlines()):
            chunks.extend(
                chunk_pokeapi_line(line, doc_id=f"{path.stem}_{i}", entity_type=entity_type)
            )

    elif source == "smogon":
        if path.stem == "smogon_data":
            chunks.extend(chunk_smogon_data_file(text, tokenize_fn=tokenize_fn))
        else:
            for i, line in enumerate(text.splitlines()):
                chunks.extend(
                    chunk_smogon_line(
                        line,
                        doc_id=f"{path.stem}_{i}",
                        entity_type=entity_type,
                        tokenize_fn=tokenize_fn,
                    )
                )

    elif source == "bulbapedia":
        docs = _RE_BULBA_DOC_SPLIT.split(text)
        for i, doc in enumerate(docs):
            doc = doc.strip()
            if doc:
                chunks.extend(
                    chunk_bulbapedia_doc(
                        doc,
                        doc_id=f"{path.stem}_{i}",
                        entity_type=entity_type,
                        tokenize_fn=tokenize_fn,
                    )
                )

    _LOG.debug("Chunked '%s' (source=%s) → %d chunk(s)", path.name, source, len(chunks))
    return chunks
