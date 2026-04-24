"""Unit tests for src/api/dependencies.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from src.pipeline.rag_pipeline import AsyncRAGPipeline, RAGPipeline
from src.retrieval.query_transformer import (
    HyDETransformer,
    MultiDraftHyDETransformer,
)


@pytest.mark.unit
class TestGetPipeline:
    def test_returns_pipeline_when_initialized(self) -> None:
        from src.api.dependencies import get_pipeline

        app = FastAPI()
        mock_pipeline = MagicMock()
        app.state.pipeline = mock_pipeline

        request = MagicMock()
        request.app = app

        result = get_pipeline(request)
        assert result is mock_pipeline

    def test_raises_runtime_error_when_not_initialized(self) -> None:
        from src.api.dependencies import get_pipeline

        app = FastAPI()  # no pipeline set on state
        request = MagicMock()
        request.app = app

        with pytest.raises(RuntimeError, match="not initialized"):
            get_pipeline(request)


@pytest.mark.unit
class TestGetAsyncPipeline:
    def test_returns_async_pipeline_when_initialized(self) -> None:
        from src.api.dependencies import get_async_pipeline

        app = FastAPI()
        mock_async_pipeline = MagicMock(spec=AsyncRAGPipeline)
        app.state.async_pipeline = mock_async_pipeline

        request = MagicMock()
        request.app = app

        result = get_async_pipeline(request)
        assert result is mock_async_pipeline

    def test_raises_runtime_error_when_not_initialized(self) -> None:
        from src.api.dependencies import get_async_pipeline

        app = FastAPI()  # no async_pipeline set on state
        request = MagicMock()
        request.app = app

        with pytest.raises(RuntimeError, match="Async pipeline not initialized"):
            get_async_pipeline(request)


@pytest.mark.unit
class TestBuildPipelineRoutingEnabled:
    def test_routing_enabled_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.api.dependencies import build_pipeline
        from src.retrieval.query_router import QueryRouter

        monkeypatch.setenv("ROUTING_ENABLED", "true")
        monkeypatch.setenv("HYDE_ENABLED", "false")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        with patch("src.api.dependencies.BGEEmbedder") as mock_embedder_cls, \
             patch("src.api.dependencies.BGEReranker") as mock_reranker_cls, \
             patch("src.api.dependencies.QdrantClient") as mock_client_cls, \
             patch("src.api.dependencies.ModelLoader") as mock_loader_cls, \
             patch("src.api.dependencies.Inferencer") as mock_inferencer_cls, \
             patch("src.api.dependencies.Generator") as mock_generator_cls:

            mock_embedder_instance = MagicMock()
            mock_embedder_cls.from_pretrained.return_value = mock_embedder_instance

            mock_reranker_instance = MagicMock()
            mock_reranker_cls.from_pretrained.return_value = mock_reranker_instance

            mock_client_instance = MagicMock()
            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.ensure_collections = MagicMock()

            mock_loader_instance = MagicMock()
            mock_loader_cls.return_value = mock_loader_instance
            mock_loader_instance.load = MagicMock()
            mock_loader_instance.get_model.return_value = MagicMock()
            mock_loader_instance.get_tokenizer.return_value = MagicMock()

            mock_inferencer_instance = MagicMock()
            mock_inferencer_cls.return_value = mock_inferencer_instance

            mock_generator_instance = MagicMock()
            mock_generator_cls.return_value = mock_generator_instance

            pipeline, loader, client = build_pipeline()

            assert isinstance(pipeline, RAGPipeline)
            assert pipeline._query_router is not None
            assert isinstance(pipeline._query_router, QueryRouter)

    def test_routing_enabled_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.api.dependencies import build_pipeline

        monkeypatch.setenv("ROUTING_ENABLED", "false")
        monkeypatch.setenv("HYDE_ENABLED", "false")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        with patch("src.api.dependencies.BGEEmbedder") as mock_embedder_cls, \
             patch("src.api.dependencies.BGEReranker") as mock_reranker_cls, \
             patch("src.api.dependencies.QdrantClient") as mock_client_cls, \
             patch("src.api.dependencies.ModelLoader") as mock_loader_cls, \
             patch("src.api.dependencies.Inferencer") as mock_inferencer_cls, \
             patch("src.api.dependencies.Generator") as mock_generator_cls:

            mock_embedder_instance = MagicMock()
            mock_embedder_cls.from_pretrained.return_value = mock_embedder_instance

            mock_reranker_instance = MagicMock()
            mock_reranker_cls.from_pretrained.return_value = mock_reranker_instance

            mock_client_instance = MagicMock()
            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.ensure_collections = MagicMock()

            mock_loader_instance = MagicMock()
            mock_loader_cls.return_value = mock_loader_instance
            mock_loader_instance.load = MagicMock()
            mock_loader_instance.get_model.return_value = MagicMock()
            mock_loader_instance.get_tokenizer.return_value = MagicMock()

            mock_inferencer_instance = MagicMock()
            mock_inferencer_cls.return_value = mock_inferencer_instance

            mock_generator_instance = MagicMock()
            mock_generator_cls.return_value = mock_generator_instance

            pipeline, loader, client = build_pipeline()

            assert isinstance(pipeline, RAGPipeline)
            assert pipeline._query_router is None


@pytest.mark.unit
class TestBuildPipelineHyDEEnabled:
    def test_hyde_enabled_single_draft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.api.dependencies import build_pipeline

        monkeypatch.setenv("HYDE_ENABLED", "true")
        monkeypatch.setenv("HYDE_NUM_DRAFTS", "1")
        monkeypatch.setenv("ROUTING_ENABLED", "false")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        with patch("src.api.dependencies.BGEEmbedder") as mock_embedder_cls, \
             patch("src.api.dependencies.BGEReranker") as mock_reranker_cls, \
             patch("src.api.dependencies.QdrantClient") as mock_client_cls, \
             patch("src.api.dependencies.ModelLoader") as mock_loader_cls, \
             patch("src.api.dependencies.Inferencer") as mock_inferencer_cls, \
             patch("src.api.dependencies.Generator") as mock_generator_cls, \
             patch("src.api.dependencies.HyDETransformer") as mock_hyde_cls:

            mock_embedder_instance = MagicMock()
            mock_embedder_cls.from_pretrained.return_value = mock_embedder_instance

            mock_reranker_instance = MagicMock()
            mock_reranker_cls.from_pretrained.return_value = mock_reranker_instance

            mock_client_instance = MagicMock()
            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.ensure_collections = MagicMock()

            mock_loader_instance = MagicMock()
            mock_loader_cls.return_value = mock_loader_instance
            mock_loader_instance.load = MagicMock()
            mock_loader_instance.get_model.return_value = MagicMock()
            mock_loader_instance.get_tokenizer.return_value = MagicMock()

            mock_inferencer_instance = MagicMock()
            mock_inferencer_cls.return_value = mock_inferencer_instance

            mock_generator_instance = MagicMock()
            mock_generator_cls.return_value = mock_generator_instance

            mock_hyde_instance = MagicMock(spec=HyDETransformer)
            mock_hyde_cls.return_value = mock_hyde_instance

            build_pipeline()

            mock_hyde_cls.assert_called_once()

    def test_hyde_enabled_multi_draft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.api.dependencies import build_pipeline

        monkeypatch.setenv("HYDE_ENABLED", "true")
        monkeypatch.setenv("HYDE_NUM_DRAFTS", "3")
        monkeypatch.setenv("ROUTING_ENABLED", "false")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        with patch("src.api.dependencies.BGEEmbedder") as mock_embedder_cls, \
             patch("src.api.dependencies.BGEReranker") as mock_reranker_cls, \
             patch("src.api.dependencies.QdrantClient") as mock_client_cls, \
             patch("src.api.dependencies.ModelLoader") as mock_loader_cls, \
             patch("src.api.dependencies.Inferencer") as mock_inferencer_cls, \
             patch("src.api.dependencies.Generator") as mock_generator_cls, \
             patch("src.api.dependencies.MultiDraftHyDETransformer") as mock_multi_hyde_cls:

            mock_embedder_instance = MagicMock()
            mock_embedder_cls.from_pretrained.return_value = mock_embedder_instance

            mock_reranker_instance = MagicMock()
            mock_reranker_cls.from_pretrained.return_value = mock_reranker_instance

            mock_client_instance = MagicMock()
            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.ensure_collections = MagicMock()

            mock_loader_instance = MagicMock()
            mock_loader_cls.return_value = mock_loader_instance
            mock_loader_instance.load = MagicMock()
            mock_loader_instance.get_model.return_value = MagicMock()
            mock_loader_instance.get_tokenizer.return_value = MagicMock()

            mock_inferencer_instance = MagicMock()
            mock_inferencer_cls.return_value = mock_inferencer_instance

            mock_generator_instance = MagicMock()
            mock_generator_cls.return_value = mock_generator_instance

            mock_multi_hyde_instance = MagicMock(spec=MultiDraftHyDETransformer)
            mock_multi_hyde_cls.return_value = mock_multi_hyde_instance

            build_pipeline()

            mock_multi_hyde_cls.assert_called_once()

    def test_hyde_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.api.dependencies import build_pipeline

        monkeypatch.setenv("HYDE_ENABLED", "false")
        monkeypatch.setenv("ROUTING_ENABLED", "false")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        with patch("src.api.dependencies.BGEEmbedder") as mock_embedder_cls, \
             patch("src.api.dependencies.BGEReranker") as mock_reranker_cls, \
             patch("src.api.dependencies.QdrantClient") as mock_client_cls, \
             patch("src.api.dependencies.ModelLoader") as mock_loader_cls, \
             patch("src.api.dependencies.Inferencer") as mock_inferencer_cls, \
             patch("src.api.dependencies.Generator") as mock_generator_cls:

            mock_embedder_instance = MagicMock()
            mock_embedder_cls.from_pretrained.return_value = mock_embedder_instance

            mock_reranker_instance = MagicMock()
            mock_reranker_cls.from_pretrained.return_value = mock_reranker_instance

            mock_client_instance = MagicMock()
            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.ensure_collections = MagicMock()

            mock_loader_instance = MagicMock()
            mock_loader_cls.return_value = mock_loader_instance
            mock_loader_instance.load = MagicMock()
            mock_loader_instance.get_model.return_value = MagicMock()
            mock_loader_instance.get_tokenizer.return_value = MagicMock()

            mock_inferencer_instance = MagicMock()
            mock_inferencer_cls.return_value = mock_inferencer_instance

            mock_generator_instance = MagicMock()
            mock_generator_cls.return_value = mock_generator_instance

            pipeline, _, _ = build_pipeline()

            assert isinstance(pipeline, RAGPipeline)
