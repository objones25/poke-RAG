"""Integration tests for the FastAPI app lifespan context manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestAppLifespan:
    def test_pipeline_initialized_on_startup(self, mocker) -> None:
        """On startup, app.state.pipeline should be set to the built pipeline."""
        from src.api.app import app

        mock_pipeline = MagicMock()
        mock_loader = MagicMock()
        mock_client = MagicMock()

        build_rv = (mock_pipeline, mock_loader, mock_client)
        with (
            patch("src.api.app.build_pipeline", return_value=build_rv),
            TestClient(app),
        ):
            assert hasattr(app.state, "pipeline")
            assert app.state.pipeline is mock_pipeline

    def test_loader_unload_called_on_shutdown(self, mocker) -> None:
        """On shutdown, loader.unload() should be called."""
        from src.api.app import app

        mock_pipeline = MagicMock()
        mock_loader = MagicMock()
        mock_client = MagicMock()

        build_rv = (mock_pipeline, mock_loader, mock_client)
        with patch("src.api.app.build_pipeline", return_value=build_rv):
            with TestClient(app):
                pass

            mock_loader.unload.assert_called_once()

    def test_loky_executor_shutdown_called_on_shutdown(self, mocker) -> None:
        """On shutdown, loky executor shutdown(wait=True) should be attempted."""
        from src.api.app import app

        mock_pipeline = MagicMock()
        mock_loader = MagicMock()
        mock_client = MagicMock()
        mock_executor = MagicMock()

        build_rv = (mock_pipeline, mock_loader, mock_client)
        # Patch at the module level where it's imported in app.py lifespan
        with (
            patch("src.api.app.build_pipeline", return_value=build_rv),
            patch("joblib.externals.loky.get_reusable_executor", return_value=mock_executor),
        ):
            with TestClient(app):
                pass

            mock_executor.shutdown.assert_called_once_with(wait=True)

    def test_build_pipeline_failure_raises_runtime_error(self, mocker) -> None:
        """If build_pipeline raises, RuntimeError with 'Failed to initialize' should be raised."""
        from src.api.app import app

        with (
            patch(
                "src.api.app.build_pipeline",
                side_effect=ValueError("Config missing"),
            ),
            pytest.raises(RuntimeError, match="Failed to initialize"),
            TestClient(app),
        ):
            pass

    def test_health_endpoint_accessible_after_startup(self, mocker) -> None:
        """After successful startup, health endpoint should be accessible."""
        from src.api.app import app

        mock_pipeline = MagicMock()
        mock_loader = MagicMock()
        mock_client = MagicMock()

        build_rv = (mock_pipeline, mock_loader, mock_client)
        with (
            patch("src.api.app.build_pipeline", return_value=build_rv),
            TestClient(app) as client,
        ):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

    def test_query_endpoint_fails_gracefully_if_pipeline_not_set(self, mocker) -> None:
        """If pipeline initialization fails, query endpoint should return 500."""
        from src.api.app import app

        with (
            patch(
                "src.api.app.build_pipeline",
                side_effect=RuntimeError("Build failed"),
            ),
            pytest.raises(RuntimeError),
            TestClient(app),
        ):
            pass


@pytest.mark.integration
class TestAsyncLifespan:
    """B2: async mode must reuse the AsyncQdrantClient from build_async_pipeline,
    not create a redundant sync QdrantClient."""

    def test_async_mode_does_not_instantiate_sync_qdrant_client(self, monkeypatch) -> None:
        from src.api.app import app

        monkeypatch.setenv("ASYNC_PIPELINE_ENABLED", "true")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        mock_pipeline = MagicMock()
        mock_loader = MagicMock()
        mock_async_client = AsyncMock()

        build_rv = (mock_pipeline, mock_loader, mock_async_client)
        with (
            patch("src.api.app.build_async_pipeline", return_value=build_rv),
            patch("src.api.app.QdrantClient") as mock_sync_cls,
            TestClient(app),
        ):
            mock_sync_cls.assert_not_called()

    def test_async_mode_stores_async_client_in_app_state(self, monkeypatch) -> None:
        from src.api.app import app

        monkeypatch.setenv("ASYNC_PIPELINE_ENABLED", "true")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        mock_pipeline = MagicMock()
        mock_loader = MagicMock()
        mock_async_client = AsyncMock()

        build_rv = (mock_pipeline, mock_loader, mock_async_client)
        with (
            patch("src.api.app.build_async_pipeline", return_value=build_rv),
            patch("src.api.app.QdrantClient"),
            TestClient(app),
        ):
            assert app.state.qdrant_client is mock_async_client

    def test_stats_uses_async_client_directly_in_async_mode(self, monkeypatch) -> None:
        from src.api.app import app

        monkeypatch.setenv("ASYNC_PIPELINE_ENABLED", "true")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
        monkeypatch.setenv("STATS_API_KEY", "test-key")

        mock_pipeline = MagicMock()
        mock_loader = MagicMock()
        mock_async_client = AsyncMock()
        mock_async_client.get_collections.return_value = MagicMock(collections=[])

        build_rv = (mock_pipeline, mock_loader, mock_async_client)
        with (
            patch("src.api.app.build_async_pipeline", return_value=build_rv),
            patch("src.api.app.QdrantClient"),
            TestClient(app) as c,
        ):
            response = c.get("/stats", headers={"Authorization": "Bearer test-key"})

        assert response.status_code == 200
        # get_collections must be called directly on the async client (awaited), not via to_thread
        mock_async_client.get_collections.assert_called_once()
