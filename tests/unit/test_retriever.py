"""Unit tests for src/retrieval/retriever.py — RED until implemented."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.retriever import Retriever
from src.retrieval.types import EmbeddingOutput
from src.types import RetrievalError, RetrievalResult, RetrievedChunk, Source
from tests.conftest import make_chunk


def _make_embedder(dense_dim: int = 1024) -> MagicMock:
    mock = MagicMock()
    mock.encode.return_value = EmbeddingOutput(
        dense=[[0.1] * dense_dim],
        sparse=[{1: 0.5, 2: 0.3}],
    )
    return mock


def _make_vector_store(chunks: list[RetrievedChunk] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.search.return_value = chunks if chunks is not None else [make_chunk()]
    return mock


def _make_reranker(chunks: list[RetrievedChunk] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.rerank.return_value = chunks if chunks is not None else [make_chunk()]
    return mock


@pytest.mark.unit
class TestRetrieverRetrieve:
    def test_returns_retrieval_result(self) -> None:
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
        )
        result = retriever.retrieve("what type is Bulbasaur?")
        assert isinstance(result, RetrievalResult)

    def test_result_contains_query(self) -> None:
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
        )
        result = retriever.retrieve("my query")
        assert result.query == "my query"

    def test_result_documents_are_retrieved_chunks(self) -> None:
        chunks = [make_chunk(chunk_index=0), make_chunk(chunk_index=1)]
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=_make_reranker(chunks),
        )
        result = retriever.retrieve("query")
        assert all(isinstance(d, RetrievedChunk) for d in result.documents)

    def test_calls_embedder_with_query(self) -> None:
        embedder = _make_embedder()
        retriever = Retriever(
            embedder=embedder,
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
        )
        retriever.retrieve("my query text")
        embedder.encode.assert_called_once_with(["my query text"])

    def test_searches_all_three_sources_by_default(self) -> None:
        vector_store = _make_vector_store()
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=vector_store,
            reranker=_make_reranker(),
        )
        retriever.retrieve("query")
        sources_searched: set[Source] = {
            call[1]["collection"] for call in vector_store.search.call_args_list
        }
        assert sources_searched == {"bulbapedia", "pokeapi", "smogon"}

    def test_searches_only_requested_sources(self) -> None:
        vector_store = _make_vector_store()
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=vector_store,
            reranker=_make_reranker(),
        )
        retriever.retrieve("query", sources=["pokeapi"])
        sources_searched = {call[1]["collection"] for call in vector_store.search.call_args_list}
        assert sources_searched == {"pokeapi"}

    def test_passes_top_k_to_reranker(self) -> None:
        reranker = _make_reranker()
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=reranker,
        )
        retriever.retrieve("query", top_k=3)
        call_kwargs = reranker.rerank.call_args[1]
        assert call_kwargs["top_k"] == 3

    def test_reranker_receives_query(self) -> None:
        reranker = _make_reranker()
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=reranker,
        )
        retriever.retrieve("specific query")
        call_args = reranker.rerank.call_args[0]
        assert call_args[0] == "specific query"

    def test_documents_match_reranker_output(self) -> None:
        reranked = [make_chunk(text="reranked result", chunk_index=0)]
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=_make_reranker(reranked),
        )
        result = retriever.retrieve("query")
        assert result.documents[0].text == "reranked result"

    def test_raises_retrieval_error_when_embedder_fails(self) -> None:
        embedder = MagicMock()
        embedder.encode.side_effect = RuntimeError("embed failed")
        retriever = Retriever(
            embedder=embedder,
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
        )
        with pytest.raises(RetrievalError):
            retriever.retrieve("query")

    def test_raises_retrieval_error_when_vector_store_fails(self) -> None:
        vector_store = MagicMock()
        vector_store.search.side_effect = RuntimeError("db down")
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=vector_store,
            reranker=_make_reranker(),
        )
        with pytest.raises(RetrievalError):
            retriever.retrieve("query")

    def test_raises_retrieval_error_when_no_chunks_found(self) -> None:
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(chunks=[]),
            reranker=_make_reranker(chunks=[]),
        )
        with pytest.raises(RetrievalError):
            retriever.retrieve("query")

    def test_raises_retrieval_error_before_reranker_when_no_candidates(self) -> None:
        reranker = _make_reranker()
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(chunks=[]),
            reranker=reranker,
        )
        with pytest.raises(RetrievalError):
            retriever.retrieve("query")
        reranker.rerank.assert_not_called()

    def test_raises_retrieval_error_when_reranker_fails(self) -> None:
        reranker = MagicMock()
        reranker.rerank.side_effect = RuntimeError("model crashed")
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=reranker,
        )
        with pytest.raises(RetrievalError):
            retriever.retrieve("query")

    def test_dense_vector_passed_to_search(self) -> None:
        dense = [0.42] * 1024
        embedder = MagicMock()
        embedder.encode.return_value = EmbeddingOutput(dense=[dense], sparse=[{1: 0.1}])
        vector_store = _make_vector_store()
        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            reranker=_make_reranker(),
        )
        retriever.retrieve("query")
        call_kwargs = vector_store.search.call_args_list[0][1]
        assert call_kwargs["query_dense"] == dense

    def test_sparse_vector_passed_to_search(self) -> None:
        sparse = {10: 0.7, 20: 0.3}
        embedder = MagicMock()
        embedder.encode.return_value = EmbeddingOutput(dense=[[0.1] * 1024], sparse=[sparse])
        vector_store = _make_vector_store()
        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            reranker=_make_reranker(),
        )
        retriever.retrieve("query")
        call_kwargs = vector_store.search.call_args_list[0][1]
        assert call_kwargs["query_sparse"] == sparse

    def test_result_documents_is_tuple(self) -> None:
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
        )
        result = retriever.retrieve("query")
        assert isinstance(result.documents, tuple)

    def test_raises_embedding_error_when_embedder_returns_empty_dense(self) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = EmbeddingOutput(dense=[], sparse=[])
        retriever = Retriever(
            embedder=embedder,
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
        )
        from src.types import EmbeddingError

        with pytest.raises(EmbeddingError):
            retriever.retrieve("query")

    def test_raises_embedding_error_when_embedder_returns_empty_sparse(self) -> None:
        embedder = MagicMock()
        embedder.encode.return_value = EmbeddingOutput(dense=[[0.1] * 1024], sparse=[])
        retriever = Retriever(
            embedder=embedder,
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
        )
        from src.types import EmbeddingError

        with pytest.raises(EmbeddingError):
            retriever.retrieve("query")

    def test_entity_name_passed_to_search(self) -> None:
        vector_store = _make_vector_store()
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=vector_store,
            reranker=_make_reranker(),
        )
        retriever.retrieve("query", entity_name="Pikachu")
        for call in vector_store.search.call_args_list:
            assert call[1]["entity_name"] == "Pikachu"

    def test_entity_name_none_by_default_in_search(self) -> None:
        vector_store = _make_vector_store()
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=vector_store,
            reranker=_make_reranker(),
        )
        retriever.retrieve("query")
        for call in vector_store.search.call_args_list:
            assert call[1]["entity_name"] is None

    def test_reranker_exception_propagates_as_retrieval_error(self) -> None:
        reranker = MagicMock()
        reranker.rerank.side_effect = RuntimeError("reranker crashed")
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=reranker,
        )
        with pytest.raises(RetrievalError):
            retriever.retrieve("query")


@pytest.mark.unit
class TestRetrieverProtocolCompliance:
    def test_satisfies_retriever_protocol(self) -> None:
        from src.retrieval.protocols import RetrieverProtocol

        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
        )
        assert isinstance(retriever, RetrieverProtocol)


@pytest.mark.unit
class TestRetrieverExceptionHandling:
    """Test that Retriever wraps exceptions with __cause__ and uses specific exception types."""

    def test_retrieve_wraps_embedding_error_with_cause(self) -> None:
        """Embedder raising EmbeddingError → RetrievalError with __cause__ set."""
        from src.types import EmbeddingError

        embedder = MagicMock()
        embedder.encode.side_effect = EmbeddingError("Model failed")
        retriever = Retriever(
            embedder=embedder,
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
        )
        try:
            retriever.retrieve("query")
            raise AssertionError("Should have raised RetrievalError")
        except RetrievalError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, EmbeddingError)

    def test_retrieve_wraps_runtime_error_from_embedder_with_cause(self) -> None:
        """Embedder raising RuntimeError → RetrievalError with __cause__ set."""
        embedder = MagicMock()
        embedder.encode.side_effect = RuntimeError("OOM")
        retriever = Retriever(
            embedder=embedder,
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
        )
        try:
            retriever.retrieve("query")
            raise AssertionError("Should have raised RetrievalError")
        except RetrievalError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, RuntimeError)

    def test_retrieve_wraps_vector_store_error_with_cause(self) -> None:
        """Vector store raising Exception → RetrievalError with __cause__ set."""
        vector_store = MagicMock()
        vector_store.search.side_effect = RuntimeError("DB connection failed")
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=vector_store,
            reranker=_make_reranker(),
        )
        try:
            retriever.retrieve("query")
            raise AssertionError("Should have raised RetrievalError")
        except RetrievalError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, RuntimeError)

    def test_retrieve_wraps_reranker_error_with_cause(self) -> None:
        """Reranker raising Exception → RetrievalError with __cause__ set."""
        reranker = MagicMock()
        reranker.rerank.side_effect = ValueError("Invalid chunks")
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=reranker,
        )
        try:
            retriever.retrieve("query")
            raise AssertionError("Should have raised RetrievalError")
        except RetrievalError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)


def _make_transformer(return_value: str = "transformed query") -> MagicMock:
    mock = MagicMock()
    mock.transform.return_value = return_value
    return mock


@pytest.mark.unit
class TestRetrieverQueryTransformer:
    def test_transformer_is_called_before_embedding(self) -> None:
        transformer = _make_transformer("hyde doc text")
        embedder = _make_embedder()
        retriever = Retriever(
            embedder=embedder,
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
            query_transformer=transformer,
        )
        retriever.retrieve("original query")
        transformer.transform.assert_called_once_with("original query")
        embedder.encode.assert_called_once_with(["hyde doc text"])

    def test_transformer_output_used_for_embedding(self) -> None:
        transformer = _make_transformer("hypothetical pokemon document")
        embedder = _make_embedder()
        retriever = Retriever(
            embedder=embedder,
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
            query_transformer=transformer,
        )
        retriever.retrieve("query")
        embedder.encode.assert_called_once_with(["hypothetical pokemon document"])

    def test_original_query_used_for_reranker(self) -> None:
        transformer = _make_transformer("transformed text")
        reranker = _make_reranker()
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=reranker,
            query_transformer=transformer,
        )
        retriever.retrieve("original query")
        call_args = reranker.rerank.call_args[0]
        assert call_args[0] == "original query"

    def test_original_query_preserved_in_result(self) -> None:
        transformer = _make_transformer("some expanded text")
        retriever = Retriever(
            embedder=_make_embedder(),
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
            query_transformer=transformer,
        )
        result = retriever.retrieve("original query")
        assert result.query == "original query"

    def test_no_transformer_uses_query_directly(self) -> None:
        embedder = _make_embedder()
        retriever = Retriever(
            embedder=embedder,
            vector_store=_make_vector_store(),
            reranker=_make_reranker(),
        )
        retriever.retrieve("plain query")
        embedder.encode.assert_called_once_with(["plain query"])
