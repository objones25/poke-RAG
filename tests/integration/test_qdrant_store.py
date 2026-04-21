"""Integration tests for QdrantVectorStore.

Tests validate the full argument shapes sent to the real qdrant-client API,
not just that mocks were called. Uses MagicMock client throughout.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.retrieval.types import EmbeddingOutput
from src.retrieval.vector_store import QdrantVectorStore
from src.types import RetrievedChunk, VectorIndexError


@pytest.mark.integration
class TestEnsureCollectionsIntegration:
    """Test ensure_collections creates collections with correct configs."""

    def test_vector_config_includes_dense_vector_params(self) -> None:
        """ensure_collections creates dense vector config with size=1024, COSINE distance."""
        mock_client = MagicMock()
        store = QdrantVectorStore(mock_client)

        store.ensure_collections()

        # Verify at least one call was made
        assert mock_client.create_collection.called
        first_call = mock_client.create_collection.call_args_list[0]

        # Extract vectors_config argument
        vectors_config = first_call.kwargs.get("vectors_config")
        assert vectors_config is not None
        assert "dense" in vectors_config

        dense_config = vectors_config["dense"]
        assert isinstance(dense_config, VectorParams)
        assert dense_config.size == 1024
        assert dense_config.distance == Distance.COSINE

    def test_sparse_vector_config_included(self) -> None:
        """ensure_collections includes sparse_vectors_config with SparseVectorParams."""
        mock_client = MagicMock()
        store = QdrantVectorStore(mock_client)

        store.ensure_collections()

        first_call = mock_client.create_collection.call_args_list[0]

        # Extract sparse_vectors_config argument
        sparse_config = first_call.kwargs.get("sparse_vectors_config")
        assert sparse_config is not None
        assert "sparse" in sparse_config

        sparse_params = sparse_config["sparse"]
        assert isinstance(sparse_params, SparseVectorParams)
        assert isinstance(sparse_params.index, SparseIndexParams)

    def test_suppresses_unexpected_response_409(self) -> None:
        """ensure_collections suppresses UnexpectedResponse(409) without raising."""
        mock_client = MagicMock()
        # Make create_collection raise 409 (already exists)
        unexpected_resp = UnexpectedResponse(
            status_code=409, reason_phrase="Conflict", content=b"", headers={}  # type: ignore[arg-type]
        )
        mock_client.create_collection.side_effect = unexpected_resp
        store = QdrantVectorStore(mock_client)

        # Should not raise; contextlib.suppress handles it
        store.ensure_collections()

        # Verify all 3 collections were attempted (even though all failed with 409)
        assert mock_client.create_collection.call_count == 3

    def test_three_collections_created(self) -> None:
        """ensure_collections creates exactly 3 collections (bulbapedia, pokeapi, smogon)."""
        mock_client = MagicMock()
        store = QdrantVectorStore(mock_client)

        store.ensure_collections()

        assert mock_client.create_collection.call_count == 3

        # Verify all 3 sources are in the calls
        call_names = {
            call_args.kwargs["collection_name"]
            for call_args in mock_client.create_collection.call_args_list
        }
        assert call_names == {"bulbapedia", "pokeapi", "smogon"}


@pytest.mark.integration
class TestUpsertIntegration:
    """Test upsert batches points and constructs correct payloads."""

    def test_batches_over_100_points(self) -> None:
        """upsert(150 chunks) calls client.upsert twice (100 + 50)."""
        mock_client = MagicMock()
        store = QdrantVectorStore(mock_client)

        # Create 150 chunks
        chunks = [
            RetrievedChunk(
                text=f"chunk {i}",
                score=0.9,
                source="pokeapi",
                entity_name="Bulbasaur",
                entity_type="pokemon",
                chunk_index=i,
                original_doc_id="doc_1",
            )
            for i in range(150)
        ]

        # Create embeddings with 150 entries (one sparse dict per document)
        embeddings = EmbeddingOutput(
            dense=[[0.1] * 1024 for _ in range(150)],
            sparse=[{i % 10: 0.5} for i in range(150)],
        )

        store.upsert("pokeapi", chunks, embeddings)

        # Should be called twice: 100 points, then 50 points
        assert mock_client.upsert.call_count == 2

        # First call: 100 points
        first_call = mock_client.upsert.call_args_list[0]
        assert len(first_call.kwargs["points"]) == 100

        # Second call: 50 points
        second_call = mock_client.upsert.call_args_list[1]
        assert len(second_call.kwargs["points"]) == 50

    def test_exactly_100_points_is_one_batch(self) -> None:
        """upsert(100 chunks) calls client.upsert exactly once."""
        mock_client = MagicMock()
        store = QdrantVectorStore(mock_client)

        chunks = [
            RetrievedChunk(
                text=f"chunk {i}",
                score=0.9,
                source="pokeapi",
                entity_name="Bulbasaur",
                entity_type="pokemon",
                chunk_index=i,
                original_doc_id="doc_1",
            )
            for i in range(100)
        ]

        embeddings = EmbeddingOutput(
            dense=[[0.1] * 1024 for _ in range(100)],
            sparse=[{i % 10: 0.5} for i in range(100)],
        )

        store.upsert("pokeapi", chunks, embeddings)

        assert mock_client.upsert.call_count == 1
        assert len(mock_client.upsert.call_args_list[0].kwargs["points"]) == 100

    def test_uuid_is_deterministic(self) -> None:
        """Same original_doc_id + chunk_index produces same UUID on separate calls."""
        mock_client = MagicMock()
        store = QdrantVectorStore(mock_client)

        chunk = RetrievedChunk(
            text="test",
            score=0.9,
            source="pokeapi",
            entity_name="Bulbasaur",
            entity_type="pokemon",
            chunk_index=5,
            original_doc_id="doc_ABC",
        )

        embeddings = EmbeddingOutput(
            dense=[[0.1] * 1024],
            sparse=[{1: 0.5}],
        )

        # First upsert
        store.upsert("pokeapi", [chunk], embeddings)
        first_uuid = mock_client.upsert.call_args_list[0].kwargs["points"][0].id

        # Reset mock
        mock_client.reset_mock()

        # Second upsert with same chunk
        store.upsert("pokeapi", [chunk], embeddings)
        second_uuid = mock_client.upsert.call_args_list[0].kwargs["points"][0].id

        # UUIDs should be identical
        assert first_uuid == second_uuid

        # Verify it's a v5 UUID based on the namespace and doc info
        expected_uuid = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.original_doc_id}:{chunk.chunk_index}")
        )
        assert first_uuid == expected_uuid

    def test_entity_type_in_payload(self) -> None:
        """Payload includes entity_type field from chunk."""
        mock_client = MagicMock()
        store = QdrantVectorStore(mock_client)

        chunk = RetrievedChunk(
            text="test",
            score=0.9,
            source="pokeapi",
            entity_name="Pikachu",
            entity_type="pokemon",
            chunk_index=0,
            original_doc_id="doc_1",
        )

        embeddings = EmbeddingOutput(
            dense=[[0.1] * 1024],
            sparse=[{1: 0.5}],
        )

        store.upsert("pokeapi", [chunk], embeddings)

        point = mock_client.upsert.call_args_list[0].kwargs["points"][0]
        assert point.payload["entity_type"] == "pokemon"

    def test_sparse_vector_indices_and_values_match(self) -> None:
        """SparseVector indices and values correctly extracted from sparse dict."""
        mock_client = MagicMock()
        store = QdrantVectorStore(mock_client)

        chunk = RetrievedChunk(
            text="test",
            score=0.9,
            source="pokeapi",
            entity_name="Bulbasaur",
            entity_type="pokemon",
            chunk_index=0,
            original_doc_id="doc_1",
        )

        # Create sparse embedding with known keys/values
        sparse_dict = {10: 0.7, 20: 0.3, 5: 0.9}

        embeddings = EmbeddingOutput(
            dense=[[0.1] * 1024],
            sparse=[sparse_dict],
        )

        store.upsert("pokeapi", [chunk], embeddings)

        point = mock_client.upsert.call_args_list[0].kwargs["points"][0]
        sparse_vector = point.vector["sparse"]

        assert isinstance(sparse_vector, SparseVector)

        # Convert to sets to ignore order
        assert set(sparse_vector.indices) == {10, 20, 5}
        assert set(sparse_vector.values) == {0.7, 0.3, 0.9}

    def test_dense_vector_length_preserved(self) -> None:
        """Dense vector with 1024 dimensions preserved in point."""
        mock_client = MagicMock()
        store = QdrantVectorStore(mock_client)

        chunk = RetrievedChunk(
            text="test",
            score=0.9,
            source="pokeapi",
            entity_name="Bulbasaur",
            entity_type="pokemon",
            chunk_index=0,
            original_doc_id="doc_1",
        )

        dense_vector = [0.5] * 1024

        embeddings = EmbeddingOutput(
            dense=[dense_vector],
            sparse=[{1: 0.5}],
        )

        store.upsert("pokeapi", [chunk], embeddings)

        point = mock_client.upsert.call_args_list[0].kwargs["points"][0]
        assert len(point.vector["dense"]) == 1024
        assert point.vector["dense"] == dense_vector


@pytest.mark.integration
class TestSearchIntegration:
    """Test search constructs correct query arguments."""

    def test_fusion_query_object_not_bare_enum(self) -> None:
        """search() passes FusionQuery(fusion=Fusion.RRF), not bare Fusion.RRF."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        query_dense = [0.1] * 1024
        query_sparse = {5: 0.8}

        store.search("pokeapi", query_dense, query_sparse, top_k=5)

        # Get the query argument passed to query_points
        call_args = mock_client.query_points.call_args
        query_arg = call_args.kwargs["query"]

        # Must be FusionQuery object, not bare Fusion enum
        assert isinstance(query_arg, FusionQuery)
        assert query_arg.fusion == Fusion.RRF

    def test_prefetch_limits_are_top_k_times_two(self) -> None:
        """Both prefetch limits are set to top_k * 2."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        query_dense = [0.1] * 1024
        query_sparse = {5: 0.8}
        top_k = 5

        store.search("pokeapi", query_dense, query_sparse, top_k=top_k)

        call_args = mock_client.query_points.call_args
        prefetches = call_args.kwargs["prefetch"]

        assert len(prefetches) == 2
        assert prefetches[0].limit == top_k * 2
        assert prefetches[1].limit == top_k * 2

    def test_dense_prefetch_uses_correct_vector_name(self) -> None:
        """Dense prefetch has using='dense'."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        query_dense = [0.1] * 1024
        query_sparse = {5: 0.8}

        store.search("pokeapi", query_dense, query_sparse, top_k=5)

        call_args = mock_client.query_points.call_args
        prefetches = call_args.kwargs["prefetch"]

        dense_prefetch = prefetches[0]
        assert dense_prefetch.using == "dense"

    def test_sparse_prefetch_is_sparse_vector_type(self) -> None:
        """Sparse prefetch query is SparseVector instance."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        query_dense = [0.1] * 1024
        query_sparse = {5: 0.8, 10: 0.2}

        store.search("pokeapi", query_dense, query_sparse, top_k=5)

        call_args = mock_client.query_points.call_args
        prefetches = call_args.kwargs["prefetch"]

        sparse_prefetch = prefetches[1]
        assert isinstance(sparse_prefetch.query, SparseVector)
        assert sparse_prefetch.using == "sparse"

    def test_entity_name_filter_is_field_condition(self) -> None:
        """When entity_name given, query_filter is Filter with FieldCondition."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        query_dense = [0.1] * 1024
        query_sparse = {5: 0.8}
        entity_name = "Pikachu"

        store.search("pokeapi", query_dense, query_sparse, top_k=5, entity_name=entity_name)

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]

        assert isinstance(query_filter, Filter)
        assert query_filter.must is not None
        assert isinstance(query_filter.must, list)
        assert len(query_filter.must) == 1

        field_condition = query_filter.must[0]
        assert isinstance(field_condition, FieldCondition)
        assert field_condition.key == "entity_name"
        assert isinstance(field_condition.match, MatchValue)
        assert field_condition.match.value == entity_name

    def test_query_filter_none_when_entity_name_not_given(self) -> None:
        """When entity_name is None, query_filter is None."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        query_dense = [0.1] * 1024
        query_sparse = {5: 0.8}

        store.search("pokeapi", query_dense, query_sparse, top_k=5, entity_name=None)

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]

        assert query_filter is None

    def test_payload_entity_type_read_correctly(self) -> None:
        """Point payload with entity_type='pokemon' returns chunk with entity_type='pokemon'."""
        mock_client = MagicMock()

        # Create a mock point with payload
        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.score = 0.95
        mock_point.payload = {
            "text": "Mewtwo is a legendary Pokemon",
            "source": "pokeapi",
            "entity_name": "Mewtwo",
            "entity_type": "pokemon",
            "chunk_index": 0,
            "original_doc_id": "doc_1",
        }

        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        results = store.search("pokeapi", [0.1] * 1024, {5: 0.8}, top_k=5)

        assert len(results) == 1
        assert results[0].entity_type == "pokemon"

    def test_missing_text_key_raises_vector_index_error(self) -> None:
        """Point payload without 'text' key raises VectorIndexError."""
        mock_client = MagicMock()

        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.score = 0.95
        mock_point.payload = {
            # Missing "text" key
            "source": "pokeapi",
            "entity_name": "Pikachu",
            "entity_type": "pokemon",
            "chunk_index": 0,
            "original_doc_id": "doc_1",
        }

        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        with pytest.raises(VectorIndexError, match="Malformed payload"):
            store.search("pokeapi", [0.1] * 1024, {5: 0.8}, top_k=5)

    def test_missing_chunk_index_key_raises_vector_index_error(self) -> None:
        """Point payload without 'chunk_index' key raises VectorIndexError."""
        mock_client = MagicMock()

        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.score = 0.95
        mock_point.payload = {
            "text": "Pikachu is electric",
            "source": "pokeapi",
            "entity_name": "Pikachu",
            "entity_type": "pokemon",
            # Missing "chunk_index" key
            "original_doc_id": "doc_1",
        }

        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        with pytest.raises(VectorIndexError, match="Malformed payload"):
            store.search("pokeapi", [0.1] * 1024, {5: 0.8}, top_k=5)

    def test_multiple_results_returned(self) -> None:
        """search() returns multiple RetrievedChunks when multiple points in response."""
        mock_client = MagicMock()

        mock_points = []
        for i in range(3):
            mock_point = MagicMock()
            mock_point.id = f"id_{i}"
            mock_point.score = 0.9 - (i * 0.05)
            mock_point.payload = {
                "text": f"chunk {i}",
                "source": "pokeapi",
                "entity_name": "Pikachu",
                "entity_type": "pokemon",
                "chunk_index": i,
                "original_doc_id": "doc_1",
            }
            mock_points.append(mock_point)

        mock_response = MagicMock()
        mock_response.points = mock_points
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        results = store.search("pokeapi", [0.1] * 1024, {5: 0.8}, top_k=5)

        assert len(results) == 3
        assert results[0].text == "chunk 0"
        assert results[1].text == "chunk 1"
        assert results[2].text == "chunk 2"
        assert results[0].score == 0.9
        assert results[1].score == 0.85

    def test_limit_parameter_passed_correctly(self) -> None:
        """search() passes top_k as limit to query_points."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        query_dense = [0.1] * 1024
        query_sparse = {5: 0.8}
        top_k = 10

        store.search("pokeapi", query_dense, query_sparse, top_k=top_k)

        call_args = mock_client.query_points.call_args
        assert call_args.kwargs["limit"] == top_k

    def test_collection_name_passed_correctly(self) -> None:
        """search() passes collection name to query_points."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(mock_client)

        store.search("bulbapedia", [0.1] * 1024, {5: 0.8}, top_k=5)

        call_args = mock_client.query_points.call_args
        assert call_args.kwargs["collection_name"] == "bulbapedia"
