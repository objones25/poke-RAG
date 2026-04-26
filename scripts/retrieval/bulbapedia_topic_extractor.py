"""Offline utility: extract topic tags for Bulbapedia chunks using Gemini.

Reads the Bulbapedia processed files, calls gemini-3.1-flash-lite-preview to
classify each chunk's topic, and writes a JSON cache mapping
original_doc_id → {"topics": [...], "entity_type_hint": "..."}.

Usage:
    uv run python scripts/retrieval/bulbapedia_topic_extractor.py \\
        --processed-dir processed/bulbapedia \\
        --output cache/bulbapedia_topics.json \\
        [--batch-size 50] [--dry-run]

The JSON cache can be passed to build_index.py via --topic-cache to enrich
Bulbapedia chunk payloads before upsert.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

load_dotenv()

_MODEL = "gemini-3.1-flash-lite-preview"
_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0
_LOG = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a metadata tagger for a Pokémon knowledge base.
Given a short chunk of text from Bulbapedia, return a JSON object with:
- "topics": list of 1–3 lowercase tags from this fixed set:
    ["move_mechanics", "ability_mechanics", "item_effect", "lore", "evolution",
     "competitive", "type_chart", "game_mechanics", "species_info", "location"]
- "entity_type_hint": one of "pokemon", "move", "ability", "item", "format", or null

Return ONLY valid JSON — no markdown, no explanation."""

_RE_BULBA_TITLE = re.compile(r"^Title:\s*(.+)$", re.MULTILINE)
_RE_BULBA_DOC_SPLIT = re.compile(r"\n(?=Title:)")


def _build_client() -> genai.Client:
    api_key = os.environ["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key, http_options=HttpOptions(timeout=60_000))


def _call_gemini(client: genai.Client, chunk_text: str) -> dict:
    prompt = f"{_SYSTEM_PROMPT}\n\nCHUNK:\n{chunk_text[:1200]}"
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=_MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            parsed = json.loads(raw)
            topics = parsed.get("topics", [])
            entity_type_hint = parsed.get("entity_type_hint")
            if not isinstance(topics, list):
                topics = []
            return {"topics": topics, "entity_type_hint": entity_type_hint}
        except json.JSONDecodeError as exc:
            _LOG.warning("JSON parse error on attempt %d: %s", attempt + 1, exc)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status == 429 or attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                _LOG.warning(
                    "Gemini error (attempt %d): %s — retrying in %.0fs", attempt + 1, exc, delay
                )
                time.sleep(delay)
            else:
                _LOG.error("Gemini call failed after %d attempts: %s", _MAX_RETRIES, exc)
                break
    return {"topics": [], "entity_type_hint": None}


def _iter_bulbapedia_docs(processed_dir: Path) -> list[tuple[str, str]]:
    """Yield (doc_id, doc_text) pairs from all bulbapedia .txt files."""
    docs: list[tuple[str, str]] = []
    for txt_path in sorted(processed_dir.glob("*.txt")):
        if txt_path.stem.endswith("_aug"):
            continue
        text = txt_path.read_text(encoding="utf-8")
        parts = _RE_BULBA_DOC_SPLIT.split(text)
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                doc_id = f"{txt_path.stem}_{i}"
                docs.append((doc_id, part))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Bulbapedia chunk topics via Gemini.")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed/bulbapedia"))
    parser.add_argument("--output", type=Path, default=Path("cache/bulbapedia_topics.json"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Skip Gemini calls; use empty tags")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, dict] = {}
    if args.output.exists():
        cache = json.loads(args.output.read_text())
        _LOG.info("Loaded %d cached entries from %s", len(cache), args.output)

    docs = _iter_bulbapedia_docs(args.processed_dir)
    _LOG.info("Found %d Bulbapedia docs in %s", len(docs), args.processed_dir)

    client = None if args.dry_run else _build_client()

    new_count = 0
    for i, (doc_id, doc_text) in enumerate(docs):
        if doc_id in cache:
            continue
        if args.dry_run:
            cache[doc_id] = {"topics": [], "entity_type_hint": None}
        else:
            cache[doc_id] = _call_gemini(client, doc_text)  # type: ignore[arg-type]
        new_count += 1

        if new_count % args.batch_size == 0:
            args.output.write_text(json.dumps(cache, indent=2))
            _LOG.info(
                "Saved %d entries (%d new) after doc %d/%d",
                len(cache),
                new_count,
                i + 1,
                len(docs),
            )

    args.output.write_text(json.dumps(cache, indent=2))
    _LOG.info("Done. %d new entries, %d total → %s", new_count, len(cache), args.output)


if __name__ == "__main__":
    main()
