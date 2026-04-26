"""Unit tests for src/retrieval/vector_store.py::AsyncQdrantVectorStore — TDD RED first."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import Fusion, FusionQuery, SparseVector

from src.retrieval.types import EmbeddingOutput
from src.retrieval.vector_store import AsyncQdrantVectorStore
from src.types import RetrievedChunk, VectorIndexError
from tests.conftest import make_chunk as _make_chunk


def _make_async_client() -> AsyncMock:
    return AsyncMock()


def _make_embeddings(n: int = 1, dense_dim: int = 1024) -> EmbeddingOutput:
    return EmbeddingOutput(
        dense=[[0.1] * dense_dim for _ in range(n)],
        sparse=[{i: 0.5 for i in range(3)} for _ in range(n)],
    )


@pytest.mark.unit
class TestAsyncEnsureCollections:
    @pytest.mark.anyio
    async def test_calls_collection_exists_for_each_source(self) -> None:
        client = _make_async_client()
        client.collection_exists.return_value = False
        store = AsyncQdrantVectorStore(client)
        await store.ensure_collections()
        assert client.collection_exists.call_count == 3

    @pytest.mark.anyio
    async def test_creates_three_collections(self) -> None:
        client = _make_async_client()
        client.collection_exists.return_value = False
        store = AsyncQdrantVectorStore(client)
        await store.ensure_collections()
        assert client.create_collection.call_count == 3

    @pytest.mark.anyio
    async def test_collection_names_are_sources(self) -> None:
        client = _make_async_client()
        client.collection_exists.return_value = False
        store = AsyncQdrantVectorStore(client)
        await store.ensure_collections()
        names = {c[1]["collection_name"] for c in client.create_collection.call_args_list}
        assert names == {"bulbapedia", "pokeapi", "smogon"}

    @pytest.mark.anyio
    async def test_skips_existing_collections(self) -> None:
        client = _make_async_client()
        client.collection_exists.return_value = True
        store = AsyncQdrantVectorStore(client)
        await store.ensure_collections()  # must not raise
        # Should not call create_collection for existing collections
        assert client.create_collection.call_count == 0

    @pytest.mark.anyio
    async def test_raises_on_http_error_from_create_collection(self) -> None:
        """HTTP errors during create_collection should propagate."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = _make_async_client()
        client.collection_exists.return_value = False
        client.create_collection.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="server error",
            content=b"",
            headers={},  # type: ignore[arg-type]
        )
        store = AsyncQdrantVectorStore(client)
        with pytest.raises(UnexpectedResponse):
            await store.ensure_collections()


@pytest.mark.unit
class TestAsyncUpsert:
    @pytest.mark.anyio
    async def test_calls_upsert_on_client(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunks = [_make_chunk()]
        embeddings = _make_embeddings(n=1)
        await store.upsert("pokeapi", chunks, embeddings)
        client.upsert.assert_called()

    @pytest.mark.anyio
    async def test_upserts_into_correct_collection(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        await store.upsert("smogon", [_make_chunk(source="smogon")], _make_embeddings(n=1))
        call_kwargs = client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "smogon"

    @pytest.mark.anyio
    async def test_upserts_all_chunks(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunks = [_make_chunk(original_doc_id=f"doc_{i}") for i in range(5)]
        await store.upsert("pokeapi", chunks, _make_embeddings(n=5))
        # Upsert may be called multiple times for batching
        total_points = sum(len(call[1]["points"]) for call in client.upsert.call_args_list)
        assert total_points == 5

    @pytest.mark.anyio
    async def test_payload_contains_text(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunk = _make_chunk(text="Bulbasaur is grass type")
        await store.upsert("pokeapi", [chunk], _make_embeddings(n=1))
        points = client.upsert.call_args[1]["points"]
        assert points[0].payload["text"] == "Bulbasaur is grass type"

    @pytest.mark.anyio
    async def test_payload_contains_metadata(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunk = _make_chunk(
            source="pokeapi", entity_name="Ivysaur", chunk_index=2, original_doc_id="d_1"
        )
        await store.upsert("pokeapi", [chunk], _make_embeddings(n=1))
        payload = client.upsert.call_args[1]["points"][0].payload
        assert payload["source"] == "pokeapi"
        assert payload["entity_name"] == "ivysaur"
        assert payload["chunk_index"] == 2
        assert payload["original_doc_id"] == "d_1"

    @pytest.mark.anyio
    async def test_point_vector_includes_dense(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        embeddings = _make_embeddings(n=1, dense_dim=1024)
        await store.upsert("pokeapi", [_make_chunk()], embeddings)
        point = client.upsert.call_args[1]["points"][0]
        assert "dense" in point.vector

    @pytest.mark.anyio
    async def test_point_vector_includes_sparse(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        await store.upsert("pokeapi", [_make_chunk()], _make_embeddings(n=1))
        point = client.upsert.call_args[1]["points"][0]
        vector = point.vector
        assert "sparse" in vector

    @pytest.mark.anyio
    async def test_upsert_normalizes_entity_name_to_lowercase(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunk = _make_chunk(entity_name="Pikachu")
        await store.upsert("pokeapi", [chunk], _make_embeddings(n=1))
        payload = client.upsert.call_args[1]["points"][0].payload
        assert payload["entity_name"] == "pikachu"

    @pytest.mark.anyio
    async def test_upsert_preserves_none_entity_name(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunk = _make_chunk(entity_name=None)
        await store.upsert("pokeapi", [chunk], _make_embeddings(n=1))
        payload = client.upsert.call_args[1]["points"][0].payload
        assert payload["entity_name"] is None

    @pytest.mark.anyio
    async def test_raises_on_embedding_length_mismatch(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunks = [_make_chunk(original_doc_id=f"doc_{i}") for i in range(3)]
        embeddings = _make_embeddings(n=2)  # only 2 embeddings for 3 chunks
        with pytest.raises(ValueError, match="length mismatch"):
            await store.upsert("pokeapi", chunks, embeddings)

    @pytest.mark.anyio
    async def test_wraps_qdrant_exception_in_vector_index_error(self) -> None:
        """When client.upsert raises, it should be wrapped as VectorIndexError."""
        client = _make_async_client()
        client.upsert.side_effect = Exception("connection refused")
        store = AsyncQdrantVectorStore(client)
        chunks = [_make_chunk()]
        embeddings = _make_embeddings(n=1)
        with pytest.raises(VectorIndexError):
            await store.upsert("pokeapi", chunks, embeddings)


@pytest.mark.unit
class TestAsyncSearch:
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

    @pytest.mark.anyio
    async def test_returns_retrieved_chunks(self) -> None:
        client = _make_async_client()
        client.query_points.return_value.points = [
            self._make_scored_point("Bulbasaur is grass", 0.9)
        ]
        store = AsyncQdrantVectorStore(client)
        results = await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5)
        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)

    @pytest.mark.anyio
    async def test_score_assigned_from_qdrant(self) -> None:
        client = _make_async_client()
        client.query_points.return_value.points = [self._make_scored_point("text", 0.87)]
        store = AsyncQdrantVectorStore(client)
        results = await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5)
        assert results[0].score == pytest.approx(0.87)

    @pytest.mark.anyio
    async def test_queries_correct_collection(self) -> None:
        client = _make_async_client()
        client.query_points.return_value.points = []
        store = AsyncQdrantVectorStore(client)
        await store.search("bulbapedia", [0.1] * 1024, {1: 0.5}, top_k=3)
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs["collection_name"] == "bulbapedia"

    @pytest.mark.anyio
    async def test_entity_name_filter_applied_when_given(self) -> None:
        client = _make_async_client()
        client.query_points.return_value.points = [
            self._make_scored_point("Pikachu text", 0.9, "Pikachu")
        ]
        store = AsyncQdrantVectorStore(client)
        await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5, entity_name="Pikachu")
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs.get("query_filter") is not None

    @pytest.mark.anyio
    async def test_entity_filter_retries_without_filter_on_zero_results(self) -> None:
        client = _make_async_client()
        empty_response = MagicMock()
        empty_response.points = []
        result_response = MagicMock()
        result_response.points = [self._make_scored_point("Pikachu is electric", 0.9, "Pikachu")]
        client.query_points.side_effect = [empty_response, result_response]
        store = AsyncQdrantVectorStore(client)
        results = await store.search(
            "pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5, entity_name="Pikachu"
        )
        assert client.query_points.call_count == 2
        second_call_kwargs = client.query_points.call_args_list[1][1]
        assert second_call_kwargs.get("query_filter") is None
        assert len(results) == 1

    @pytest.mark.anyio
    async def test_entity_filter_fallback_marks_chunks_in_metadata(self) -> None:
        """B5: async path — fallback chunks must carry entity_filter_fallback=True in metadata."""
        client = _make_async_client()
        empty_response = MagicMock()
        empty_response.points = []
        result_response = MagicMock()
        result_response.points = [self._make_scored_point("Pikachu is electric", 0.9, "Pikachu")]
        client.query_points.side_effect = [empty_response, result_response]
        store = AsyncQdrantVectorStore(client)
        results = await store.search(
            "pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5, entity_name="Pikachu"
        )
        assert len(results) == 1
        assert results[0].metadata is not None
        assert results[0].metadata.get("entity_filter_fallback") is True

    @pytest.mark.anyio
    async def test_search_normalizes_entity_name_to_lowercase_in_filter(self) -> None:
        client = _make_async_client()
        client.query_points.return_value.points = [
            self._make_scored_point("Pikachu text", 0.9, "pikachu")
        ]
        store = AsyncQdrantVectorStore(client)
        await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5, entity_name="Pikachu")
        first_call_kwargs = client.query_points.call_args_list[0][1]
        query_filter = first_call_kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must[0].match.value == "pikachu"

    @pytest.mark.anyio
    async def test_no_filter_when_entity_name_is_none(self) -> None:
        client = _make_async_client()
        client.query_points.return_value.points = []
        store = AsyncQdrantVectorStore(client)
        await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5, entity_name=None)
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs.get("query_filter") is None

    @pytest.mark.anyio
    async def test_top_k_forwarded(self) -> None:
        client = _make_async_client()
        client.query_points.return_value.points = []
        store = AsyncQdrantVectorStore(client)
        await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=7)
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs.get("limit") == 7

    @pytest.mark.anyio
    async def test_empty_results_returns_empty_list(self) -> None:
        client = _make_async_client()
        client.query_points.return_value.points = []
        store = AsyncQdrantVectorStore(client)
        results = await store.search("pokeapi", [0.1] * 1024, {}, top_k=5)
        assert results == []

    @pytest.mark.anyio
    async def test_chunk_fields_match_payload(self) -> None:
        client = _make_async_client()
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
        store = AsyncQdrantVectorStore(client)
        results = await store.search("pokeapi", [0.1] * 1024, {}, top_k=1)
        chunk = results[0]
        assert chunk.text == "Pikachu is electric"
        assert chunk.source == "pokeapi"
        assert chunk.entity_name == "Pikachu"
        assert chunk.chunk_index == 1
        assert chunk.original_doc_id == "poke_5"

    @pytest.mark.anyio
    async def test_search_uses_fusion_rrf(self) -> None:
        client = _make_async_client()
        client.query_points.return_value.points = []
        store = AsyncQdrantVectorStore(client)
        await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5)
        call_kwargs = client.query_points.call_args[1]
        query_arg = call_kwargs.get("query")
        assert isinstance(query_arg, FusionQuery)
        assert query_arg.fusion == Fusion.RRF

    @pytest.mark.anyio
    async def test_search_sends_two_prefetch_legs(self) -> None:
        client = _make_async_client()
        client.query_points.return_value.points = []
        store = AsyncQdrantVectorStore(client)
        await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5)
        call_kwargs = client.query_points.call_args[1]
        prefetches = call_kwargs.get("prefetch", [])
        assert len(prefetches) == 2

    @pytest.mark.anyio
    async def test_search_sparse_prefetch_is_sparse_vector(self) -> None:
        client = _make_async_client()
        client.query_points.return_value.points = []
        store = AsyncQdrantVectorStore(client)
        await store.search("pokeapi", [0.1] * 1024, {10: 0.7, 20: 0.3}, top_k=5)
        call_kwargs = client.query_points.call_args[1]
        prefetches = call_kwargs.get("prefetch", [])
        sparse_pf = next(p for p in prefetches if p.using == "sparse")
        assert isinstance(sparse_pf.query, SparseVector)
        assert set(sparse_pf.query.indices) == {10, 20}

    @pytest.mark.anyio
    async def test_malformed_payload_skipped_when_other_valid_results_exist(self) -> None:
        client = _make_async_client()
        bad_point = MagicMock()
        bad_point.score = 0.8
        bad_point.id = "bad_id"
        bad_point.payload = {"source": "pokeapi"}  # missing "text" and others
        good_point = self._make_scored_point("Valid text", 0.9)
        client.query_points.return_value.points = [bad_point, good_point]
        store = AsyncQdrantVectorStore(client)
        results = await store.search("pokeapi", [0.1] * 1024, {}, top_k=1)
        assert len(results) == 1
        assert results[0].text == "Valid text"

    @pytest.mark.anyio
    async def test_malformed_payload_raises_when_all_invalid(self) -> None:
        client = _make_async_client()
        bad_point = MagicMock()
        bad_point.id = "bad_id"
        bad_point.score = 0.8
        bad_point.payload = {"source": "pokeapi"}  # missing "text" and others
        client.query_points.return_value.points = [bad_point]
        store = AsyncQdrantVectorStore(client)
        with pytest.raises(VectorIndexError):
            await store.search("pokeapi", [0.1] * 1024, {}, top_k=1)

    @pytest.mark.anyio
    async def test_wraps_qdrant_exception_in_search(self) -> None:
        """When client.query_points raises, it should be wrapped as VectorIndexError."""
        client = _make_async_client()
        client.query_points.side_effect = OSError("connection reset")
        store = AsyncQdrantVectorStore(client)
        with pytest.raises(VectorIndexError):
            await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=5)


@pytest.mark.unit
class TestAsyncClose:
    @pytest.mark.anyio
    async def test_calls_close_on_client(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        await store.close()
        client.close.assert_called_once()


@pytest.mark.unit
class TestAsyncUpsertBatchBoundaries:
    @pytest.mark.anyio
    async def test_upsert_empty_documents(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        await store.upsert("pokeapi", [], _make_embeddings(n=0))
        client.upsert.assert_not_called()

    @pytest.mark.anyio
    async def test_upsert_99_documents_one_batch(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunks = [_make_chunk(original_doc_id=f"doc_{i}") for i in range(99)]
        await store.upsert("pokeapi", chunks, _make_embeddings(n=99))
        assert client.upsert.call_count == 1
        points = client.upsert.call_args[1]["points"]
        assert len(points) == 99

    @pytest.mark.anyio
    async def test_upsert_100_documents_exactly_one_batch(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunks = [_make_chunk(original_doc_id=f"doc_{i}") for i in range(100)]
        await store.upsert("pokeapi", chunks, _make_embeddings(n=100))
        assert client.upsert.call_count == 1
        points = client.upsert.call_args[1]["points"]
        assert len(points) == 100

    @pytest.mark.anyio
    async def test_upsert_101_documents_two_batches(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunks = [_make_chunk(original_doc_id=f"doc_{i}") for i in range(101)]
        await store.upsert("pokeapi", chunks, _make_embeddings(n=101))
        assert client.upsert.call_count == 2
        first_batch = client.upsert.call_args_list[0][1]["points"]
        second_batch = client.upsert.call_args_list[1][1]["points"]
        assert len(first_batch) == 100
        assert len(second_batch) == 1

    @pytest.mark.anyio
    async def test_upsert_200_documents_exactly_two_batches(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunks = [_make_chunk(original_doc_id=f"doc_{i}") for i in range(200)]
        await store.upsert("pokeapi", chunks, _make_embeddings(n=200))
        assert client.upsert.call_count == 2
        first_batch = client.upsert.call_args_list[0][1]["points"]
        second_batch = client.upsert.call_args_list[1][1]["points"]
        assert len(first_batch) == 100
        assert len(second_batch) == 100


@pytest.mark.unit
class TestAsyncEnsureCollectionsErrorHandling:
    @pytest.mark.anyio
    async def test_ensure_collections_propagates_collection_exists_error(self) -> None:
        client = _make_async_client()
        client.collection_exists.side_effect = Exception("connection error")
        store = AsyncQdrantVectorStore(client)
        with pytest.raises(Exception, match="connection error"):
            await store.ensure_collections()

    @pytest.mark.anyio
    async def test_ensure_collections_propagates_create_collection_error(self) -> None:
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = _make_async_client()
        client.collection_exists.return_value = False
        client.create_collection.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="server error",
            content=b"",
            headers={},  # type: ignore[arg-type]
        )
        store = AsyncQdrantVectorStore(client)
        with pytest.raises(UnexpectedResponse):
            await store.ensure_collections()



@pytest.mark.unit
class TestAsyncConcurrentUpsert:
    @pytest.mark.anyio
    async def test_concurrent_upserts_do_not_interleave(self) -> None:
        import asyncio

        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        chunks1 = [_make_chunk(original_doc_id=f"doc_1_{i}") for i in range(50)]
        chunks2 = [_make_chunk(original_doc_id=f"doc_2_{i}") for i in range(50)]
        embeddings1 = _make_embeddings(n=50)
        embeddings2 = _make_embeddings(n=50)
        await asyncio.gather(
            store.upsert("pokeapi", chunks1, embeddings1),
            store.upsert("pokeapi", chunks2, embeddings2),
        )
        assert client.upsert.call_count == 2
        first_call_points = client.upsert.call_args_list[0][1]["points"]
        second_call_points = client.upsert.call_args_list[1][1]["points"]
        assert len(first_call_points) == 50
        assert len(second_call_points) == 50


@pytest.mark.unit
class TestAsyncVectorStoreProtocolCompliance:
    def test_satisfies_async_vector_store_protocol(self) -> None:
        from src.retrieval.protocols import AsyncVectorStoreProtocol

        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        assert isinstance(store, AsyncVectorStoreProtocol)


@pytest.mark.unit
class TestAsyncSearchTopKValidation:
    @pytest.mark.anyio
    async def test_search_raises_on_zero_top_k(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        with pytest.raises(ValueError, match="top_k"):
            await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=0)

    @pytest.mark.anyio
    async def test_search_raises_on_negative_top_k(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        with pytest.raises(ValueError, match="top_k"):
            await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=-1)

    @pytest.mark.anyio
    async def test_search_raises_on_excessive_top_k(self) -> None:
        client = _make_async_client()
        store = AsyncQdrantVectorStore(client)
        with pytest.raises(ValueError, match="top_k"):
            await store.search("pokeapi", [0.1] * 1024, {1: 0.5}, top_k=1001)
