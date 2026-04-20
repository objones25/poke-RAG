"""Integration tests for scripts/build_index.py — RED until implementation exists."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.types import EmbeddingOutput
from src.types import RetrievedChunk, Source
from tests.conftest import make_chunk as _make_chunk


def _make_embedder(dense_dim: int = 3, num_texts: int = 1) -> MagicMock:
    mock = MagicMock()
    mock.encode.return_value = EmbeddingOutput(
        dense=[[0.1] * dense_dim for _ in range(num_texts)],
        sparse=[{1: 0.5} for _ in range(num_texts)],
    )
    return mock


# ---------------------------------------------------------------------------
# discover_files
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDiscoverFiles:
    def test_returns_non_aug_txt_files(self, tmp_path: Path) -> None:
        from scripts.build_index import discover_files

        source_dir = tmp_path / "pokeapi"
        source_dir.mkdir()
        (source_dir / "ability.txt").write_text("data")
        (source_dir / "ability_aug.txt").write_text("aug data")

        results = discover_files(tmp_path, ("pokeapi",))
        assert len(results) == 1
        assert results[0] == ("pokeapi", source_dir / "ability.txt")

    def test_excludes_aug_files(self, tmp_path: Path) -> None:
        from scripts.build_index import discover_files

        source_dir = tmp_path / "smogon"
        source_dir.mkdir()
        (source_dir / "pokemon.txt").write_text("data")
        (source_dir / "pokemon_aug.txt").write_text("aug")
        (source_dir / "formats_aug.txt").write_text("aug")

        results = discover_files(tmp_path, ("smogon",))
        names = [p.name for _, p in results]
        assert "pokemon_aug.txt" not in names
        assert "formats_aug.txt" not in names
        assert "pokemon.txt" in names

    def test_respects_source_filter(self, tmp_path: Path) -> None:
        from scripts.build_index import discover_files

        for source in ("bulbapedia", "pokeapi", "smogon"):
            d = tmp_path / source
            d.mkdir()
            (d / "data.txt").write_text("data")

        results = discover_files(tmp_path, ("pokeapi",))
        assert all(src == "pokeapi" for src, _ in results)
        assert len(results) == 1

    def test_returns_all_sources_when_no_filter(self, tmp_path: Path) -> None:
        from scripts.build_index import discover_files

        for source in ("bulbapedia", "pokeapi", "smogon"):
            d = tmp_path / source
            d.mkdir()
            (d / "data.txt").write_text("data")

        results = discover_files(tmp_path, ("bulbapedia", "pokeapi", "smogon"))
        sources_found = {src for src, _ in results}
        assert sources_found == {"bulbapedia", "pokeapi", "smogon"}

    def test_warns_and_skips_missing_source_dir(self, tmp_path: Path) -> None:
        from scripts.build_index import discover_files

        results = discover_files(tmp_path, ("pokeapi",))
        assert results == []

    def test_results_are_sorted_by_filename(self, tmp_path: Path) -> None:
        from scripts.build_index import discover_files

        source_dir = tmp_path / "pokeapi"
        source_dir.mkdir()
        (source_dir / "z_data.txt").write_text("z")
        (source_dir / "a_data.txt").write_text("a")

        results = discover_files(tmp_path, ("pokeapi",))
        names = [p.name for _, p in results]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# chunk_all_files
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestChunkAllFiles:
    def test_calls_chunk_file_for_each_file(self, tmp_path: Path) -> None:
        from scripts.build_index import chunk_all_files

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("x")
        f2.write_text("y")

        with patch("scripts.build_index.chunk_file") as mock_chunk:
            mock_chunk.return_value = [_make_chunk()]
            chunk_all_files([("pokeapi", f1), ("smogon", f2)])

        assert mock_chunk.call_count == 2
        mock_chunk.assert_any_call(f1, source="pokeapi")
        mock_chunk.assert_any_call(f2, source="smogon")

    def test_accumulates_chunks_from_all_files(self, tmp_path: Path) -> None:
        from scripts.build_index import chunk_all_files

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("x")
        f2.write_text("y")

        with patch("scripts.build_index.chunk_file") as mock_chunk:
            mock_chunk.side_effect = [
                [_make_chunk(text="a1"), _make_chunk(text="a2")],
                [_make_chunk(text="b1")],
            ]
            result = chunk_all_files([("pokeapi", f1), ("pokeapi", f2)])

        assert len(result) == 3
        assert [c.text for c in result] == ["a1", "a2", "b1"]

    def test_empty_files_list_returns_empty(self) -> None:
        from scripts.build_index import chunk_all_files

        result = chunk_all_files([])
        assert result == []


# ---------------------------------------------------------------------------
# embed_in_batches
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEmbedInBatches:
    def test_empty_chunks_returns_empty_output(self) -> None:
        from scripts.build_index import embed_in_batches

        embedder = MagicMock()
        result = embed_in_batches(embedder, [], batch_size=32)
        assert result == EmbeddingOutput(dense=[], sparse=[])
        embedder.encode.assert_not_called()

    def test_single_batch_calls_encode_once(self) -> None:
        from scripts.build_index import embed_in_batches

        chunks = [_make_chunk(text=f"text {i}") for i in range(3)]
        embedder = _make_embedder(num_texts=3)

        embed_in_batches(embedder, chunks, batch_size=32)
        embedder.encode.assert_called_once_with(["text 0", "text 1", "text 2"])

    def test_multiple_batches_called_with_correct_texts(self) -> None:
        from scripts.build_index import embed_in_batches

        chunks = [_make_chunk(text=f"t{i}") for i in range(5)]

        def encode_side_effect(texts: list[str]) -> EmbeddingOutput:
            return EmbeddingOutput(
                dense=[[float(i)] for i in range(len(texts))],
                sparse=[{0: 1.0} for _ in texts],
            )

        embedder = MagicMock()
        embedder.encode.side_effect = encode_side_effect

        embed_in_batches(embedder, chunks, batch_size=2)
        assert embedder.encode.call_count == 3
        embedder.encode.assert_any_call(["t0", "t1"])
        embedder.encode.assert_any_call(["t2", "t3"])
        embedder.encode.assert_any_call(["t4"])

    def test_output_length_matches_input(self) -> None:
        from scripts.build_index import embed_in_batches

        chunks = [_make_chunk(text=f"t{i}") for i in range(5)]
        embedder = MagicMock()
        embedder.encode.side_effect = lambda texts: EmbeddingOutput(
            dense=[[0.1] for _ in texts], sparse=[{0: 1.0} for _ in texts]
        )

        result = embed_in_batches(embedder, chunks, batch_size=2)
        assert len(result.dense) == 5
        assert len(result.sparse) == 5

    def test_raises_if_embedder_returns_wrong_count(self) -> None:
        from scripts.build_index import embed_in_batches

        chunks = [_make_chunk(text=f"t{i}") for i in range(3)]
        embedder = MagicMock()
        embedder.encode.return_value = EmbeddingOutput(dense=[[0.1]], sparse=[{0: 1.0}])

        with pytest.raises(RuntimeError, match="batch"):
            embed_in_batches(embedder, chunks, batch_size=32)


# ---------------------------------------------------------------------------
# group_by_source
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGroupBySource:
    def test_chunks_grouped_by_source(self) -> None:
        from scripts.build_index import group_by_source

        chunks = [
            _make_chunk(text="a", source="bulbapedia", chunk_index=0),
            _make_chunk(text="b", source="pokeapi", chunk_index=0),
            _make_chunk(text="c", source="bulbapedia", chunk_index=1),
        ]
        embeddings = EmbeddingOutput(
            dense=[[1.0], [2.0], [3.0]],
            sparse=[{0: 1.0}, {1: 1.0}, {2: 1.0}],
        )

        result = group_by_source(chunks, embeddings)
        assert set(result.keys()) == {"bulbapedia", "pokeapi"}
        assert len(result["bulbapedia"][0]) == 2
        assert len(result["pokeapi"][0]) == 1

    def test_embeddings_follow_chunk_order(self) -> None:
        from scripts.build_index import group_by_source

        chunks = [
            _make_chunk(text="a", source="smogon", chunk_index=0),
            _make_chunk(text="b", source="smogon", chunk_index=1),
        ]
        embeddings = EmbeddingOutput(
            dense=[[1.0, 0.0], [0.0, 1.0]],
            sparse=[{0: 0.9}, {1: 0.8}],
        )

        result = group_by_source(chunks, embeddings)
        smogon_chunks, smogon_emb = result["smogon"]
        assert smogon_emb.dense[0] == [1.0, 0.0]
        assert smogon_emb.dense[1] == [0.0, 1.0]
        assert smogon_emb.sparse[0] == {0: 0.9}

    def test_empty_input_returns_empty_dict(self) -> None:
        from scripts.build_index import group_by_source

        result = group_by_source([], EmbeddingOutput(dense=[], sparse=[]))
        assert result == {}


# ---------------------------------------------------------------------------
# run() — orchestration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRun:
    def _make_deps(self) -> tuple[MagicMock, MagicMock]:
        embedder = MagicMock()
        embedder.encode.side_effect = lambda texts: EmbeddingOutput(
            dense=[[0.1] for _ in texts], sparse=[{0: 1.0} for _ in texts]
        )
        vector_store = MagicMock()
        return embedder, vector_store

    def test_dry_run_skips_upsert(self, tmp_path: Path) -> None:
        from scripts.build_index import run

        source_dir = tmp_path / "pokeapi"
        source_dir.mkdir()
        (source_dir / "ability.txt").write_text("Pikachu is an electric pokemon.")

        embedder, vector_store = self._make_deps()

        with patch("scripts.build_index.chunk_file") as mock_chunk:
            mock_chunk.return_value = [_make_chunk()]
            run(
                embedder=embedder,
                vector_store=vector_store,
                sources=("pokeapi",),
                processed_dir=tmp_path,
                dry_run=True,
            )

        vector_store.upsert.assert_not_called()
        vector_store.ensure_collections.assert_not_called()

    def test_normal_run_calls_ensure_collections(self, tmp_path: Path) -> None:
        from scripts.build_index import run

        source_dir = tmp_path / "pokeapi"
        source_dir.mkdir()
        (source_dir / "ability.txt").write_text("x")

        embedder, vector_store = self._make_deps()

        with patch("scripts.build_index.chunk_file") as mock_chunk:
            mock_chunk.return_value = [_make_chunk()]
            run(
                embedder=embedder,
                vector_store=vector_store,
                sources=("pokeapi",),
                processed_dir=tmp_path,
                dry_run=False,
            )

        vector_store.ensure_collections.assert_called_once()

    def test_normal_run_calls_upsert_per_source(self, tmp_path: Path) -> None:
        from scripts.build_index import run

        for src in ("bulbapedia", "pokeapi"):
            d = tmp_path / src
            d.mkdir()
            (d / "data.txt").write_text("x")

        embedder, vector_store = self._make_deps()

        def chunk_side_effect(path: Path, *, source: Source) -> list[RetrievedChunk]:
            return [_make_chunk(source=source)]

        with patch("scripts.build_index.chunk_file", side_effect=chunk_side_effect):
            run(
                embedder=embedder,
                vector_store=vector_store,
                sources=("bulbapedia", "pokeapi"),
                processed_dir=tmp_path,
                dry_run=False,
            )

        assert vector_store.upsert.call_count == 2
        upserted_sources = {c.args[0] for c in vector_store.upsert.call_args_list}
        assert upserted_sources == {"bulbapedia", "pokeapi"}

    def test_no_files_found_returns_early(self, tmp_path: Path) -> None:
        from scripts.build_index import run

        embedder, vector_store = self._make_deps()

        run(
            embedder=embedder,
            vector_store=vector_store,
            sources=("pokeapi",),
            processed_dir=tmp_path,
            dry_run=False,
        )

        vector_store.ensure_collections.assert_not_called()
        vector_store.upsert.assert_not_called()
