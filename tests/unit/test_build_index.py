"""Unit tests for scripts/build_index.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.build_index import group_by_source, run
from src.retrieval.types import EmbeddingOutput
from src.types import RetrievedChunk
from tests.conftest import make_chunk as _make_chunk


def _make_embeddings(n: int = 1, with_colbert: bool = False) -> EmbeddingOutput:
    colbert = [[[0.1] * 1024 for _ in range(4)] for _ in range(n)] if with_colbert else None
    return EmbeddingOutput(
        dense=[[0.1] * 1024 for _ in range(n)],
        sparse=[{i: 0.5 for i in range(3)} for _ in range(n)],
        colbert=colbert,
    )


def _make_chunks(n: int = 1, source: str = "pokeapi") -> list[RetrievedChunk]:
    return [_make_chunk(source=source, chunk_index=i, original_doc_id=f"doc_{i}") for i in range(n)]  # type: ignore[arg-type]


@pytest.mark.unit
class TestGroupBySource:
    def test_groups_by_source(self) -> None:
        chunks = [
            _make_chunk(source="pokeapi", chunk_index=0, original_doc_id="d0"),
            _make_chunk(source="smogon", chunk_index=0, original_doc_id="d1"),
        ]
        embeddings = _make_embeddings(n=2)
        result = group_by_source(chunks, embeddings)
        assert "pokeapi" in result
        assert "smogon" in result

    def test_empty_chunks_returns_empty_dict(self) -> None:
        result = group_by_source([], _make_embeddings(n=0))
        assert result == {}

    def test_colbert_accumulated_when_present(self) -> None:
        chunks = _make_chunks(n=2, source="pokeapi")
        embeddings = _make_embeddings(n=2, with_colbert=True)
        result = group_by_source(chunks, embeddings)
        grouped_emb = result["pokeapi"][1]
        assert grouped_emb.colbert is not None
        assert len(grouped_emb.colbert) == 2

    def test_colbert_none_when_not_present(self) -> None:
        chunks = _make_chunks(n=1, source="pokeapi")
        embeddings = _make_embeddings(n=1, with_colbert=False)
        result = group_by_source(chunks, embeddings)
        grouped_emb = result["pokeapi"][1]
        assert grouped_emb.colbert is None

    def test_correct_chunk_count_per_source(self) -> None:
        chunks = [
            _make_chunk(source="pokeapi", chunk_index=0, original_doc_id="d0"),
            _make_chunk(source="pokeapi", chunk_index=1, original_doc_id="d1"),
            _make_chunk(source="smogon", chunk_index=0, original_doc_id="d2"),
        ]
        embeddings = _make_embeddings(n=3)
        result = group_by_source(chunks, embeddings)
        assert len(result["pokeapi"][0]) == 2
        assert len(result["smogon"][0]) == 1


@pytest.mark.unit
class TestRunDropCollections:
    def test_drop_collections_called_when_flag_set(self, tmp_path: Path) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = _make_embeddings(n=1)
        vector_store = MagicMock()

        processed_dir = tmp_path / "processed"
        (processed_dir / "pokeapi").mkdir(parents=True)
        (processed_dir / "pokeapi" / "bulbasaur.txt").write_text("Name: Bulbasaur\nGrass type.")

        chunks = _make_chunks(n=1, source="pokeapi")
        with patch("scripts.build_index.chunk_file", return_value=chunks):
            run(
                embedder=embedder,
                vector_store=vector_store,
                sources=("pokeapi",),
                processed_dir=processed_dir,
                drop_collections=True,
            )

        vector_store.drop_collections.assert_called_once()

    def test_drop_collections_not_called_when_flag_false(self, tmp_path: Path) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = _make_embeddings(n=1)
        vector_store = MagicMock()

        processed_dir = tmp_path / "processed"
        (processed_dir / "pokeapi").mkdir(parents=True)
        (processed_dir / "pokeapi" / "bulbasaur.txt").write_text("Name: Bulbasaur\nGrass type.")

        chunks = _make_chunks(n=1, source="pokeapi")
        with patch("scripts.build_index.chunk_file", return_value=chunks):
            run(
                embedder=embedder,
                vector_store=vector_store,
                sources=("pokeapi",),
                processed_dir=processed_dir,
                drop_collections=False,
            )

        vector_store.drop_collections.assert_not_called()

    def test_drop_collections_not_called_in_dry_run(self, tmp_path: Path) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = _make_embeddings(n=1)
        vector_store = MagicMock()

        processed_dir = tmp_path / "processed"
        (processed_dir / "pokeapi").mkdir(parents=True)
        (processed_dir / "pokeapi" / "bulbasaur.txt").write_text("Name: Bulbasaur\nGrass type.")

        chunks = _make_chunks(n=1, source="pokeapi")
        with patch("scripts.build_index.chunk_file", return_value=chunks):
            run(
                embedder=embedder,
                vector_store=vector_store,
                sources=("pokeapi",),
                processed_dir=processed_dir,
                drop_collections=True,
                dry_run=True,
            )

        vector_store.drop_collections.assert_not_called()


@pytest.mark.unit
class TestRunColBERTValidation:
    def test_raises_when_colbert_enabled_but_missing_from_embedder(self, tmp_path: Path) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = _make_embeddings(n=1, with_colbert=False)
        vector_store = MagicMock()

        processed_dir = tmp_path / "processed"
        (processed_dir / "pokeapi").mkdir(parents=True)
        (processed_dir / "pokeapi" / "bulbasaur.txt").write_text("Name: Bulbasaur\nGrass type.")

        chunks = _make_chunks(n=1, source="pokeapi")
        with (
            patch("scripts.build_index.chunk_file", return_value=chunks),
            pytest.raises(RuntimeError, match="ColBERT enabled"),
        ):
            run(
                embedder=embedder,
                vector_store=vector_store,
                sources=("pokeapi",),
                processed_dir=processed_dir,
                colbert_enabled=True,
            )

    def test_no_colbert_validation_when_disabled(self, tmp_path: Path) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = _make_embeddings(n=1, with_colbert=False)
        vector_store = MagicMock()

        processed_dir = tmp_path / "processed"
        (processed_dir / "pokeapi").mkdir(parents=True)
        (processed_dir / "pokeapi" / "bulbasaur.txt").write_text("Name: Bulbasaur\nGrass type.")

        chunks = _make_chunks(n=1, source="pokeapi")
        with patch("scripts.build_index.chunk_file", return_value=chunks):
            run(
                embedder=embedder,
                vector_store=vector_store,
                sources=("pokeapi",),
                processed_dir=processed_dir,
                colbert_enabled=False,
            )

        vector_store.upsert.assert_called_once()

    def test_succeeds_when_colbert_enabled_and_present(self, tmp_path: Path) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = _make_embeddings(n=1, with_colbert=True)
        vector_store = MagicMock()

        processed_dir = tmp_path / "processed"
        (processed_dir / "pokeapi").mkdir(parents=True)
        (processed_dir / "pokeapi" / "bulbasaur.txt").write_text("Name: Bulbasaur\nGrass type.")

        chunks = _make_chunks(n=1, source="pokeapi")
        with patch("scripts.build_index.chunk_file", return_value=chunks):
            run(
                embedder=embedder,
                vector_store=vector_store,
                sources=("pokeapi",),
                processed_dir=processed_dir,
                colbert_enabled=True,
            )

        vector_store.upsert.assert_called_once()


@pytest.mark.unit
class TestRunNoFiles:
    def test_run_logs_warning_when_no_files(self, tmp_path: Path) -> None:
        embedder = MagicMock()
        vector_store = MagicMock()
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        run(
            embedder=embedder,
            vector_store=vector_store,
            sources=("pokeapi",),
            processed_dir=processed_dir,
        )

        vector_store.upsert.assert_not_called()
        vector_store.ensure_collections.assert_not_called()
