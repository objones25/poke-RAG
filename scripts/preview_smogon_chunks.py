"""Preview chunks produced from processed/smogon/smogon_data.txt.

Usage:
    uv run python scripts/preview_smogon_chunks.py                # first 20 chunks
    uv run python scripts/preview_smogon_chunks.py --n 50         # first 50 chunks
    uv run python scripts/preview_smogon_chunks.py --pokemon Venusaur
    uv run python scripts/preview_smogon_chunks.py --pokemon Venusaur --format gen9uu
    uv run python scripts/preview_smogon_chunks.py --stats         # summary only
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "processed" / "smogon" / "smogon_data.txt"

sys.path.insert(0, str(ROOT))

from src.retrieval.chunker import chunk_smogon_data_file  # noqa: E402


def _chunk_kind(doc_id: str) -> str:
    if doc_id.endswith("_overview"):
        return "overview"
    if "_set_" in doc_id:
        return "set"
    return "other"


def _print_chunk(chunk, index: int) -> None:
    kind = _chunk_kind(chunk.original_doc_id)
    print(f"{'─' * 72}")
    print(f"  [{index}]  entity={chunk.entity_name}  kind={kind}  chunk_index={chunk.chunk_index}")
    print(f"       doc_id={chunk.original_doc_id}")
    print()
    for line in chunk.text.splitlines():
        print(f"  {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview smogon_data chunks")
    parser.add_argument("--n", type=int, default=20, help="max chunks to display (default 20)")
    parser.add_argument("--pokemon", type=str, default=None, help="filter by entity_name (case-insensitive)")
    parser.add_argument("--format", type=str, default=None, help="filter by format name in doc_id")
    parser.add_argument("--stats", action="store_true", help="print summary stats only")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found", file=sys.stderr)
        sys.exit(1)

    text = DATA_FILE.read_text(encoding="utf-8")
    chunks = chunk_smogon_data_file(text)

    kind_counts = Counter(_chunk_kind(c.original_doc_id) for c in chunks)
    entities = {c.entity_name for c in chunks}
    print(f"Total chunks : {len(chunks)}")
    print(f"Unique Pokémon: {len(entities)}")
    print(f"  overview chunks : {kind_counts['overview']}")
    print(f"  set chunks      : {kind_counts['set']}")

    if args.stats:
        return

    filtered = chunks
    if args.pokemon:
        needle = args.pokemon.lower()
        filtered = [c for c in filtered if (c.entity_name or "").lower() == needle]
    if args.format:
        filtered = [c for c in filtered if args.format in (c.original_doc_id or "")]

    if not filtered:
        print("\nNo chunks matched the filter.")
        return

    print(f"\nShowing {min(args.n, len(filtered))} of {len(filtered)} matching chunks:\n")
    for i, chunk in enumerate(filtered[: args.n]):
        _print_chunk(chunk, i)


if __name__ == "__main__":
    main()
