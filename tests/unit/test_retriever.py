"""Unit tests for src/retrieval/retriever.py — RED until implemented."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.retriever import Retriever
from src.retrieval.types import EmbeddingOutput
from src.types import RetrievalError, RetrievalResult, RetrievedChunk, Source


def _make_chunk(
    text: str = "some text",
    score: float = 0.9,
    source: str = "pokeapi",
    pokemon_name: str | None = "Bulbasaur",
    idx: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        source=source,  # type: ignore[arg-type]
        pokemon_name=pokemon_name,
        chunk_index=idx,
        original_doc_id=f"doc_{idx}",
    )


def _make_embedder(dense_dim: int = 1024) -> MagicMock:
    mock = MagicMock()
    mock.encode.return_value = EmbeddingOutput(
        dense=[[0.1] * dense_dim],
        sparse=[{1: 0.5, 2: 0.3}],
    )
    return mock


def _make_vector_store(chunks: list[RetrievedChunk] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.search.return_value = chunks if chunks is not None else [_make_chunk()]
    return mock


def _make_reranker(chunks: list[RetrievedChunk] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.rerank.return_value = chunks if chunks is not None else [_make_chunk()]
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
        chunks = [_make_chunk(idx=0), _make_chunk(idx=1)]
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
        sources_searched = {
            call[1]["collection"] for call in vector_store.search.call_args_list
        }
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
        reranked = [_make_chunk(text="reranked result", idx=0)]
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
        embedder.encode.return_value = EmbeddingOutput(
            dense=[[0.1] * 1024], sparse=[sparse]
        )
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
