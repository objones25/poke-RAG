"""Unit tests for src/retrieval/vector_store.py — RED until vector_store.py is implemented."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client.models import Fusion, FusionQuery, SparseVector

from src.retrieval.types import EmbeddingOutput
from src.retrieval.vector_store import QdrantVectorStore
from src.types import RetrievedChunk, VectorIndexError
from tests.conftest import make_chunk as _make_chunk


def _make_client() -> MagicMock:
    return MagicMock()


def _make_embeddings(n: int = 1, dense_dim: int = 1024) -> EmbeddingOutput:
    return EmbeddingOutput(
        dense=[[0.1] * dense_dim for _ in range(n)],
        sparse=[{i: 0.5 for i in range(3)} for _ in range(n)],
    )


@pytest.mark.unit
class TestEnsureCollections:
    def test_creates_three_collections(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        store.ensure_collections()
        assert client.create_collection.call_count == 3

    def test_collection_names_are_sources(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        store.ensure_collections()
        names = {c[1]["collection_name"] for c in client.create_collection.call_args_list}
        assert names == {"bulbapedia", "pokeapi", "smogon"}

    def test_skips_existing_collections(self) -> None:
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = _make_client()
        client.create_collection.side_effect = UnexpectedResponse(
            status_code=400, reason_phrase="already exists", content=b"", headers={}  # type: ignore[arg-type]
        )
        store = QdrantVectorStore(client)
        store.ensure_collections()  # must not raise


@pytest.mark.unit
class TestUpsert:
    def test_calls_upsert_on_client(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        chunks = [_make_chunk()]
        embeddings = _make_embeddings(n=1)
        store.upsert("pokeapi", chunks, embeddings)
        client.upsert.assert_called_once()

    def test_upserts_into_correct_collection(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        store.upsert("smogon", [_make_chunk(source="smogon")], _make_embeddings(n=1))
        call_kwargs = client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "smogon"

    def test_upserts_all_chunks(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        chunks = [_make_chunk(original_doc_id=f"doc_{i}") for i in range(5)]
        store.upsert("pokeapi", chunks, _make_embeddings(n=5))
        points = client.upsert.call_args[1]["points"]
        assert len(points) == 5

    def test_payload_contains_text(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        chunk = _make_chunk(text="Bulbasaur is grass type")
        store.upsert("pokeapi", [chunk], _make_embeddings(n=1))
        points = client.upsert.call_args[1]["points"]
        assert points[0].payload["text"] == "Bulbasaur is grass type"

    def test_payload_contains_metadata(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        chunk = _make_chunk(
            source="pokeapi", entity_name="Ivysaur", chunk_index=2, original_doc_id="d_1"
        )
        store.upsert("pokeapi", [chunk], _make_embeddings(n=1))
        payload = client.upsert.call_args[1]["points"][0].payload
        assert payload["source"] == "pokeapi"
        assert payload["entity_name"] == "ivysaur"
        assert payload["chunk_index"] == 2
        assert payload["original_doc_id"] == "d_1"

    def test_point_vector_includes_dense(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        embeddings = _make_embeddings(n=1, dense_dim=1024)
        store.upsert("pokeapi", [_make_chunk()], embeddings)
        point = client.upsert.call_args[1]["points"][0]
        assert "dense" in point.vector

    def test_point_vector_includes_sparse(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        store.upsert("pokeapi", [_make_chunk()], _make_embeddings(n=1))
        point = client.upsert.call_args[1]["points"][0]
        vector = point.vector
        assert "sparse" in vector

    def test_upsert_normalizes_entity_name_to_lowercase(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        chunk = _make_chunk(entity_name="Pikachu")
        store.upsert("pokeapi", [chunk], _make_embeddings(n=1))
        payload = client.upsert.call_args[1]["points"][0].payload
        assert payload["entity_name"] == "pikachu"

    def test_upsert_preserves_none_entity_name(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        chunk = _make_chunk(entity_name=None)
        store.upsert("pokeapi", [chunk], _make_embeddings(n=1))
        payload = client.upsert.call_args[1]["points"][0].payload
        assert payload["entity_name"] is None

    def test_upsert_preserves_empty_string_entity_name(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        chunk = _make_chunk(entity_name="")
        store.upsert("pokeapi", [chunk], _make_embeddings(n=1))
        payload = client.upsert.call_args[1]["points"][0].payload
        assert payload["entity_name"] == ""

    def test_raises_on_embedding_length_mismatch(self) -> None:
        client = _make_client()
        store = QdrantVectorStore(client)
        chunks = [_make_chunk(original_doc_id=f"doc_{i}") for i in range(3)]
        embeddings = _make_embeddings(n=2)  # only 2 embeddings for 3 chunks
        with pytest.raises(ValueError, match="length mismatch"):
            store.upsert("pokeapi", chunks, embeddings)


@pytest.mark.unit
class TestSearch:
    def _make_scored_point(self, text: str, score: float, entity: str | None = None) -> MagicMock:
        p = MagicMock()
        p.score = score
        p.payload = {
            "text": text,
            "source": "pokeapi",
            "entity_name": entity,
            "entity_type": None,
            "chunk_index": 0,
            "original_doc_id": "doc_0",
        }
        return p

    def test_returns_retrieved_chunks(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = [
            self._make_scored_point("Bulbasaur is grass", 0.9)
        ]
        store = QdrantVectorStore(client)
        results = store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5)
        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)

    def test_score_assigned_from_qdrant(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = [self._make_scored_point("text", 0.87)]
        store = QdrantVectorStore(client)
        results = store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5)
        assert results[0].score == pytest.approx(0.87)

    def test_queries_correct_collection(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = []
        store = QdrantVectorStore(client)
        store.search("bulbapedia", [0.1] * 1024, {1: 0.5}, top_k=3)
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs["collection_name"] == "bulbapedia"

    def test_entity_name_filter_applied_when_given(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = [
            self._make_scored_point("Pikachu text", 0.9, "Pikachu")
        ]
        store = QdrantVectorStore(client)
        store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5, entity_name="Pikachu")
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs.get("query_filter") is not None

    def test_entity_filter_retries_without_filter_on_zero_results(self) -> None:
        client = _make_client()
        empty_response = MagicMock()
        empty_response.points = []
        result_response = MagicMock()
        result_response.points = [self._make_scored_point("Pikachu is electric", 0.9, "Pikachu")]
        client.query_points.side_effect = [empty_response, result_response]
        store = QdrantVectorStore(client)
        results = store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5, entity_name="Pikachu")
        assert client.query_points.call_count == 2
        second_call_kwargs = client.query_points.call_args_list[1][1]
        assert second_call_kwargs.get("query_filter") is None
        assert len(results) == 1

    def test_search_normalizes_entity_name_to_lowercase_in_filter(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = [
            self._make_scored_point("Pikachu text", 0.9, "pikachu")
        ]
        store = QdrantVectorStore(client)
        store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5, entity_name="Pikachu")
        first_call_kwargs = client.query_points.call_args_list[0][1]
        query_filter = first_call_kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must[0].match.value == "pikachu"

    def test_no_filter_when_entity_name_is_none(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = []
        store = QdrantVectorStore(client)
        store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5, entity_name=None)
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs.get("query_filter") is None

    def test_search_with_empty_string_entity_name_creates_filter(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = [
            self._make_scored_point("Some text", 0.9, "")
        ]
        store = QdrantVectorStore(client)
        store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5, entity_name="")
        first_call_kwargs = client.query_points.call_args_list[0][1]
        query_filter = first_call_kwargs.get("query_filter")
        # Empty string after strip is still empty, so filter is created with empty string value
        assert query_filter is not None
        assert query_filter.must[0].match.value == ""

    def test_top_k_forwarded(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = []
        store = QdrantVectorStore(client)
        store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=7)
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs.get("limit") == 7

    def test_empty_results_returns_empty_list(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = []
        store = QdrantVectorStore(client)
        results = store.search("pokeapi", [0.1] * 1024, {}, top_k=5)
        assert results == []

    def test_chunk_fields_match_payload(self) -> None:
        client = _make_client()
        p = MagicMock()
        p.score = 0.75
        p.payload = {
            "text": "Pikachu is electric",
            "source": "pokeapi",
            "entity_name": "Pikachu",
            "entity_type": "pokemon",
            "chunk_index": 1,
            "original_doc_id": "poke_5",
        }
        client.query_points.return_value.points = [p]
        store = QdrantVectorStore(client)
        results = store.search("pokeapi", [0.1] * 1024, {}, top_k=1)
        chunk = results[0]
        assert chunk.text == "Pikachu is electric"
        assert chunk.source == "pokeapi"
        assert chunk.entity_name == "Pikachu"
        assert chunk.chunk_index == 1
        assert chunk.original_doc_id == "poke_5"

    def test_search_uses_fusion_rrf(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = []
        store = QdrantVectorStore(client)
        store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5)
        call_kwargs = client.query_points.call_args[1]
        query_arg = call_kwargs.get("query")
        assert isinstance(query_arg, FusionQuery)
        assert query_arg.fusion == Fusion.RRF

    def test_search_sends_two_prefetch_legs(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = []
        store = QdrantVectorStore(client)
        store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5)
        call_kwargs = client.query_points.call_args[1]
        prefetches = call_kwargs.get("prefetch", [])
        assert len(prefetches) == 2

    def test_search_sparse_prefetch_is_sparse_vector(self) -> None:
        client = _make_client()
        client.query_points.return_value.points = []
        store = QdrantVectorStore(client)
        store.search("pokeapi", [0.1] * 1024, {10: 0.7, 20: 0.3}, top_k=5)
        call_kwargs = client.query_points.call_args[1]
        prefetches = call_kwargs.get("prefetch", [])
        sparse_pf = next(p for p in prefetches if p.using == "sparse")
        assert isinstance(sparse_pf.query, SparseVector)
        assert set(sparse_pf.query.indices) == {10, 20}

    def test_malformed_payload_skipped_when_other_valid_results_exist(self) -> None:
        client = _make_client()
        bad_point = MagicMock()
        bad_point.score = 0.8
        bad_point.id = "bad_id"
        bad_point.payload = {"source": "pokeapi"}  # missing "text" and others
        good_point = self._make_scored_point("Valid text", 0.9)
        client.query_points.return_value.points = [bad_point, good_point]
        store = QdrantVectorStore(client)
        results = store.search("pokeapi", [0.1] * 1024, {}, top_k=1)
        assert len(results) == 1
        assert results[0].text == "Valid text"

    def test_malformed_payload_raises_when_all_invalid(self) -> None:
        client = _make_client()
        bad_point = MagicMock()
        bad_point.id = "bad_id"
        bad_point.score = 0.8
        bad_point.payload = {"source": "pokeapi"}  # missing "text" and others
        client.query_points.return_value.points = [bad_point]
        store = QdrantVectorStore(client)
        with pytest.raises(VectorIndexError):
            store.search("pokeapi", [0.1] * 1024, {}, top_k=1)


@pytest.mark.unit
class TestVectorStoreProtocolCompliance:
    def test_satisfies_vector_store_protocol(self) -> None:
        from src.retrieval.protocols import VectorStoreProtocol

        store = QdrantVectorStore(_make_client())
        assert isinstance(store, VectorStoreProtocol)
