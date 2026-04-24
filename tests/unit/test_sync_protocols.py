"""Unit tests for structural subtyping of sync protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from unittest.mock import MagicMock

import pytest

from src.generation.protocols import GeneratorProtocol, PromptBuilderProtocol
from src.retrieval.protocols import (
    EmbedderProtocol,
    QueryRouterProtocol,
    QueryTransformerProtocol,
    RerankerProtocol,
    RetrieverProtocol,
    VectorStoreProtocol,
)
from src.types import GenerationResult, RetrievalResult, RetrievedChunk, Source


@pytest.mark.unit
class TestEmbedderProtocolCompliance:
    def test_mock_with_encode_method_satisfies_protocol(self) -> None:
        mock = MagicMock()
        mock.encode = MagicMock()
        assert isinstance(mock, EmbedderProtocol)

    def test_object_without_encode_does_not_satisfy_protocol(self) -> None:
        class NotEmbedder:
            pass

        assert not isinstance(NotEmbedder(), EmbedderProtocol)

    def test_object_with_wrong_method_name_does_not_satisfy(self) -> None:
        class BadEmbedder:
            def embedding(self, texts: list[str]) -> None:
                pass

        assert not isinstance(BadEmbedder(), EmbedderProtocol)


@pytest.mark.unit
class TestVectorStoreProtocolCompliance:
    def test_mock_with_all_required_methods_satisfies_protocol(self) -> None:
        mock = MagicMock()
        mock.ensure_collections = MagicMock()
        mock.upsert = MagicMock()
        mock.search = MagicMock()
        assert isinstance(mock, VectorStoreProtocol)

    def test_object_without_ensure_collections_does_not_satisfy(self) -> None:
        class BadVectorStore:
            def upsert(self) -> None:
                pass

            def search(self) -> None:
                pass

        assert not isinstance(BadVectorStore(), VectorStoreProtocol)

    def test_object_without_upsert_does_not_satisfy(self) -> None:
        class BadVectorStore:
            def ensure_collections(self) -> None:
                pass

            def search(self) -> None:
                pass

        assert not isinstance(BadVectorStore(), VectorStoreProtocol)

    def test_object_without_search_does_not_satisfy(self) -> None:
        class BadVectorStore:
            def ensure_collections(self) -> None:
                pass

            def upsert(self) -> None:
                pass

        assert not isinstance(BadVectorStore(), VectorStoreProtocol)


@pytest.mark.unit
class TestRerankerProtocolCompliance:
    def test_mock_with_rerank_method_satisfies_protocol(self) -> None:
        mock = MagicMock()
        mock.rerank = MagicMock()
        assert isinstance(mock, RerankerProtocol)

    def test_object_without_rerank_does_not_satisfy_protocol(self) -> None:
        class NotReranker:
            pass

        assert not isinstance(NotReranker(), RerankerProtocol)

    def test_object_with_wrong_method_name_does_not_satisfy(self) -> None:
        class BadReranker:
            def rank(self) -> None:
                pass

        assert not isinstance(BadReranker(), RerankerProtocol)


@pytest.mark.unit
class TestQueryTransformerProtocolCompliance:
    def test_mock_with_transform_method_satisfies_protocol(self) -> None:
        mock = MagicMock()
        mock.transform = MagicMock(return_value="transformed query")
        assert isinstance(mock, QueryTransformerProtocol)

    def test_object_without_transform_does_not_satisfy_protocol(self) -> None:
        class NotTransformer:
            pass

        assert not isinstance(NotTransformer(), QueryTransformerProtocol)

    def test_object_with_wrong_method_name_does_not_satisfy(self) -> None:
        class BadTransformer:
            def convert(self, query: str) -> str:
                return query

        assert not isinstance(BadTransformer(), QueryTransformerProtocol)


@pytest.mark.unit
class TestRetrieverProtocolCompliance:
    def test_mock_with_retrieve_method_satisfies_protocol(self) -> None:
        mock = MagicMock()
        mock.retrieve = MagicMock()
        assert isinstance(mock, RetrieverProtocol)

    def test_object_without_retrieve_does_not_satisfy_protocol(self) -> None:
        class NotRetriever:
            pass

        assert not isinstance(NotRetriever(), RetrieverProtocol)

    def test_object_with_wrong_method_name_does_not_satisfy(self) -> None:
        class BadRetriever:
            def search(self, query: str) -> None:
                pass

        assert not isinstance(BadRetriever(), RetrieverProtocol)


@pytest.mark.unit
class TestQueryRouterProtocolCompliance:
    def test_mock_with_route_method_satisfies_protocol(self) -> None:
        mock = MagicMock()
        mock.route = MagicMock(return_value=["pokeapi"])
        assert isinstance(mock, QueryRouterProtocol)

    def test_object_without_route_does_not_satisfy_protocol(self) -> None:
        class NotRouter:
            pass

        assert not isinstance(NotRouter(), QueryRouterProtocol)

    def test_object_with_wrong_method_name_does_not_satisfy(self) -> None:
        class BadRouter:
            def classify(self, query: str) -> list[Source]:
                return ["pokeapi"]

        assert not isinstance(BadRouter(), QueryRouterProtocol)


@pytest.mark.unit
class TestPromptBuilderProtocolCompliance:
    def test_callable_object_with_correct_signature_satisfies_protocol(self) -> None:
        mock = MagicMock()
        mock.return_value = "prompt text"
        assert isinstance(mock, PromptBuilderProtocol)

    def test_lambda_callable_satisfies_protocol(self) -> None:
        fn = lambda query, chunks: "prompt"  # noqa: E731
        assert isinstance(fn, PromptBuilderProtocol)

    def test_object_without_call_method_does_not_satisfy(self) -> None:
        class NotCallable:
            pass

        assert not isinstance(NotCallable(), PromptBuilderProtocol)


@pytest.mark.unit
class TestGeneratorProtocolCompliance:
    def test_mock_with_generate_method_satisfies_protocol(self) -> None:
        mock = MagicMock()
        mock.generate = MagicMock()
        assert isinstance(mock, GeneratorProtocol)

    def test_object_without_generate_does_not_satisfy_protocol(self) -> None:
        class NotGenerator:
            pass

        assert not isinstance(NotGenerator(), GeneratorProtocol)

    def test_object_with_wrong_method_name_does_not_satisfy(self) -> None:
        class BadGenerator:
            def run(self, query: str, chunks: tuple[RetrievedChunk, ...]) -> GenerationResult:
                return GenerationResult(
                    answer="test",
                    sources_used=("pokeapi",),
                    model_name="test",
                    num_chunks_used=1,
                )

        assert not isinstance(BadGenerator(), GeneratorProtocol)


@pytest.mark.unit
class TestProtocolMethodSignatureMatching:
    def test_embedder_protocol_requires_correct_argument_names(self) -> None:
        @runtime_checkable
        class StrictEmbedderProtocol(Protocol):
            def encode(self, texts: list[str]) -> None:
                pass

        class CorrectEmbedder:
            def encode(self, texts: list[str]) -> None:
                pass

        class WrongArgName:
            def encode(self, documents: list[str]) -> None:
                pass

        assert isinstance(CorrectEmbedder(), StrictEmbedderProtocol)
        assert isinstance(WrongArgName(), StrictEmbedderProtocol)

    def test_vector_store_search_keyword_args_signature(self) -> None:
        class MockVectorStore:
            def ensure_collections(self) -> None:
                pass

            def upsert(
                self, collection: Source, documents: list[RetrievedChunk], embeddings: object
            ) -> None:
                pass

            def search(
                self,
                collection: Source,
                query_dense: list[float],
                query_sparse: dict[int, float],
                top_k: int,
                entity_name: str | None = None,
            ) -> list[RetrievedChunk]:
                return []

        assert isinstance(MockVectorStore(), VectorStoreProtocol)

    def test_retriever_protocol_keyword_only_args(self) -> None:
        class MockRetriever:
            def retrieve(
                self,
                query: str,
                *,
                top_k: int = 5,
                sources: list[Source] | None = None,
                entity_name: str | None = None,
            ) -> RetrievalResult:
                return RetrievalResult(documents=(), query=query)

        assert isinstance(MockRetriever(), RetrieverProtocol)

    def test_query_router_returns_list_of_sources(self) -> None:
        class MockRouter:
            def route(self, query: str) -> list[Source]:
                return ["pokeapi"]

        assert isinstance(MockRouter(), QueryRouterProtocol)
