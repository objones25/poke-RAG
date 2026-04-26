"""Embed processed Pokémon data and upsert into Qdrant, file by file."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.retrieval.chunker import chunk_file
from src.retrieval.types import EmbeddingOutput
from src.types import RetrievedChunk, Source

if TYPE_CHECKING:
    from src.retrieval.embedder import BGEEmbedder
    from src.retrieval.vector_store import QdrantVectorStore

_LOG = logging.getLogger(__name__)

_PROCESSED_DIR = Path(__file__).parent.parent / "processed"
_ALL_SOURCES: tuple[Source, ...] = ("bulbapedia", "pokeapi", "smogon")
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_CHECKPOINT = Path(__file__).parent.parent / ".build_index_checkpoint.json"


def discover_files(
    processed_dir: Path,
    sources: tuple[Source, ...],
) -> list[tuple[Source, Path]]:
    results: list[tuple[Source, Path]] = []
    for source in sources:
        source_dir = processed_dir / source
        if not source_dir.exists():
            _LOG.warning("Source directory not found, skipping: %s", source_dir)
            continue
        files = sorted(p for p in source_dir.glob("*.txt") if not p.stem.endswith("_aug"))
        results.extend((source, f) for f in files)
    return results


def chunk_all_files(
    files: list[tuple[Source, Path]],
) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for source, path in files:
        chunks.extend(chunk_file(path, source=source))
    return chunks


def group_by_source(
    chunks: list[RetrievedChunk],
    embeddings: EmbeddingOutput,
) -> dict[Source, tuple[list[RetrievedChunk], EmbeddingOutput]]:
    if not chunks:
        return {}

    grouped: dict[
        Source,
        tuple[
            list[RetrievedChunk],
            list[list[float]],
            list[dict[int, float]],
            list[list[list[float]]] | None,
        ],
    ]
    grouped = {}

    for idx, chunk in enumerate(chunks):
        source = chunk.source
        if source not in grouped:
            colbert_acc: list[list[list[float]]] | None = (
                [] if embeddings.colbert is not None else None
            )
            grouped[source] = ([], [], [], colbert_acc)
        grouped[source][0].append(chunk)
        grouped[source][1].append(embeddings.dense[idx])
        grouped[source][2].append(embeddings.sparse[idx])
        if embeddings.colbert is not None and grouped[source][3] is not None:
            grouped[source][3].append(embeddings.colbert[idx])

    return {
        src: (
            clist,
            EmbeddingOutput(dense=dlist, sparse=slist, colbert=cblist),
        )
        for src, (clist, dlist, slist, cblist) in grouped.items()
    }


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError):
        _LOG.warning("Could not read checkpoint file %s — starting fresh", path)
        return set()


def _save_checkpoint(path: Path, completed: set[str]) -> None:
    path.write_text(json.dumps(sorted(completed)))


def run(
    *,
    embedder: BGEEmbedder,
    vector_store: QdrantVectorStore,
    sources: tuple[Source, ...],
    processed_dir: Path,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
    checkpoint_path: Path | None = None,
    drop_collections: bool = False,
    colbert_enabled: bool = False,
    topic_lookup: dict[str, dict] | None = None,
) -> None:
    files = discover_files(processed_dir, sources)
    if not files:
        _LOG.warning("No files found in %s for sources %s", processed_dir, sources)
        return

    completed = _load_checkpoint(checkpoint_path) if checkpoint_path else set()
    remaining = [(src, p) for src, p in files if f"{src}/{p.name}" not in completed]

    if not remaining:
        _LOG.info("All %d file(s) already indexed — nothing to do.", len(files))
        return

    _LOG.info(
        "Discovered %d file(s); %d already indexed, %d to process.",
        len(files),
        len(files) - len(remaining),
        len(remaining),
    )

    if not dry_run:
        if drop_collections:
            _LOG.info("Dropping existing collections before rebuild")
            vector_store.drop_collections()
        vector_store.ensure_collections()

    for source, path in remaining:
        file_key = f"{source}/{path.name}"
        try:
            chunks = chunk_file(path, source=source, topic_lookup=topic_lookup)
            _LOG.info("Processing '%s' → %d chunk(s)", file_key, len(chunks))

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                texts = [c.text for c in batch]
                result = embedder.encode(texts)
                if len(result.dense) != len(batch) or len(result.sparse) != len(batch):
                    raise RuntimeError(
                        f"Embedder returned {len(result.dense)} dense and "
                        f"{len(result.sparse)} sparse vectors for batch of {len(batch)}"
                    )
                if colbert_enabled and (
                    result.colbert is None or len(result.colbert) != len(batch)
                ):
                    raise RuntimeError(
                        f"ColBERT enabled but embedder returned "
                        f"{len(result.colbert) if result.colbert else None} ColBERT vectors "
                        f"for batch of {len(batch)}"
                    )
                if dry_run:
                    _LOG.info("[dry-run] would upsert %d chunk(s) into '%s'", len(batch), source)
                else:
                    vector_store.upsert(source, batch, result)
                    _LOG.debug("Upserted batch %d–%d for '%s'", i, i + len(batch), file_key)

            completed.add(file_key)
            if checkpoint_path:
                _save_checkpoint(checkpoint_path, completed)
            _LOG.info("Done: %s", file_key)
        except Exception as exc:
            _LOG.error("Failed to process '%s': %s", file_key, exc)
            raise

    _LOG.info("Indexing complete.")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build Qdrant index from processed Pokémon data.")
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        choices=list(_ALL_SOURCES),
        help="Source(s) to index (default: all). May be repeated.",
    )
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--dry-run", action="store_true", help="Log what would be indexed without writing."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help="Path to checkpoint file (default: .build_index_checkpoint.json).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing (re-index everything).",
    )
    parser.add_argument(
        "--colbert",
        action="store_true",
        help="Enable ColBERT multi-vector embeddings (requires --drop-collections on first run).",
    )
    parser.add_argument(
        "--drop-collections",
        action="store_true",
        help="Drop existing Qdrant collections before indexing (required when changing schema).",
    )
    parser.add_argument(
        "--topic-cache",
        type=Path,
        default=None,
        help="Path to JSON topic cache from scripts/retrieval/bulbapedia_topic_extractor.py.",
    )
    args = parser.parse_args()

    sources: tuple[Source, ...] = tuple(args.sources) if args.sources else _ALL_SOURCES
    checkpoint_path: Path | None = None if args.no_checkpoint else args.checkpoint
    colbert_enabled: bool = args.colbert

    topic_lookup: dict[str, dict[str, Any]] | None = None
    if args.topic_cache is not None:
        if not args.topic_cache.exists():
            _LOG.error("--topic-cache file not found: %s", args.topic_cache)
            sys.exit(1)
        topic_lookup = json.loads(args.topic_cache.read_text())
        _LOG.info("Loaded %d topic cache entries from %s", len(topic_lookup), args.topic_cache)

    try:
        from src.config import Settings

        settings = Settings.from_env()
    except KeyError as exc:
        _LOG.error("Missing required environment variable: %s. Set QDRANT_URL and retry.", exc)
        sys.exit(1)

    from qdrant_client import QdrantClient

    from src.retrieval.embedder import BGEEmbedder
    from src.retrieval.vector_store import QdrantVectorStore

    embedder = BGEEmbedder.from_pretrained(
        model_name=settings.embed_model,
        device=settings.device,
        colbert_enabled=colbert_enabled,
    )
    api_key_str = (
        None if settings.qdrant_api_key is None else settings.qdrant_api_key.get_secret_value()
    )
    timeout = 120 if colbert_enabled else 30
    client = QdrantClient(url=settings.qdrant_url, api_key=api_key_str, timeout=timeout)
    vector_store = QdrantVectorStore(client, colbert_enabled=colbert_enabled)

    run(
        embedder=embedder,
        vector_store=vector_store,
        sources=sources,
        processed_dir=_PROCESSED_DIR,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        checkpoint_path=checkpoint_path,
        drop_collections=args.drop_collections,
        colbert_enabled=colbert_enabled,
        topic_lookup=topic_lookup,
    )


if __name__ == "__main__":
    main()
