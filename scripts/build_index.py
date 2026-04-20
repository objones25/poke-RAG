"""Embed processed Pokémon data and upsert into Qdrant."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
        files = sorted(
            p for p in source_dir.glob("*.txt") if not p.stem.endswith("_aug")
        )
        results.extend((source, f) for f in files)
    return results


def chunk_all_files(
    files: list[tuple[Source, Path]],
) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for source, path in files:
        chunks.extend(chunk_file(path, source=source))
    return chunks


def embed_in_batches(
    embedder: BGEEmbedder,
    chunks: list[RetrievedChunk],
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> EmbeddingOutput:
    if not chunks:
        return EmbeddingOutput(dense=[], sparse=[])

    all_dense: list[list[float]] = []
    all_sparse: list[dict[int, float]] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        result = embedder.encode(texts)
        if len(result.dense) != len(batch) or len(result.sparse) != len(batch):
            raise RuntimeError(
                f"Embedder returned {len(result.dense)} dense and "
                f"{len(result.sparse)} sparse vectors for batch of {len(batch)}"
            )
        all_dense.extend(result.dense)
        all_sparse.extend(result.sparse)

    return EmbeddingOutput(dense=all_dense, sparse=all_sparse)


def group_by_source(
    chunks: list[RetrievedChunk],
    embeddings: EmbeddingOutput,
) -> dict[Source, tuple[list[RetrievedChunk], EmbeddingOutput]]:
    if not chunks:
        return {}

    grouped: dict[Source, tuple[list[RetrievedChunk], list[list[float]], list[dict[int, float]]]]
    grouped = {}

    for idx, chunk in enumerate(chunks):
        source = chunk.source
        if source not in grouped:
            grouped[source] = ([], [], [])
        grouped[source][0].append(chunk)
        grouped[source][1].append(embeddings.dense[idx])
        grouped[source][2].append(embeddings.sparse[idx])

    return {
        src: (clist, EmbeddingOutput(dense=dlist, sparse=slist))
        for src, (clist, dlist, slist) in grouped.items()
    }


def run(
    *,
    embedder: BGEEmbedder,
    vector_store: QdrantVectorStore,
    sources: tuple[Source, ...],
    processed_dir: Path,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> None:
    files = discover_files(processed_dir, sources)
    if not files:
        _LOG.warning("No files found in %s for sources %s", processed_dir, sources)
        return

    _LOG.info("Discovered %d files across %d sources", len(files), len(sources))
    chunks = chunk_all_files(files)
    _LOG.info("Produced %d chunks", len(chunks))

    embeddings = embed_in_batches(embedder, chunks, batch_size=batch_size)
    _LOG.info("Embedded %d chunks", len(embeddings.dense))

    grouped = group_by_source(chunks, embeddings)

    if dry_run:
        for source, (src_chunks, _) in grouped.items():
            _LOG.info("[dry-run] would upsert %d chunks into '%s'", len(src_chunks), source)
        return

    vector_store.ensure_collections()
    for source, (src_chunks, src_embeddings) in grouped.items():
        _LOG.info("Upserting %d chunks into '%s'", len(src_chunks), source)
        vector_store.upsert(source, src_chunks, src_embeddings)
    _LOG.info("Done.")


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
    args = parser.parse_args()

    sources: tuple[Source, ...] = tuple(args.sources) if args.sources else _ALL_SOURCES

    try:
        from src.config import Settings
        settings = Settings.from_env()
    except KeyError as exc:
        _LOG.error("Missing required environment variable: %s. Set QDRANT_URL and retry.", exc)
        sys.exit(1)

    import torch
    from qdrant_client import QdrantClient

    from src.retrieval.embedder import BGEEmbedder
    from src.retrieval.vector_store import QdrantVectorStore

    use_fp16 = torch.cuda.is_available()
    embedder = BGEEmbedder.from_pretrained(model_name=settings.embed_model, use_fp16=use_fp16)
    client = QdrantClient(url=settings.qdrant_url)
    vector_store = QdrantVectorStore(client)

    run(
        embedder=embedder,
        vector_store=vector_store,
        sources=sources,
        processed_dir=_PROCESSED_DIR,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
