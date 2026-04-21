"""Integration tests for the FastAPI app lifespan context manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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

        with patch(
            "src.api.app.build_pipeline",
            side_effect=ValueError("Config missing"),
        ), pytest.raises(RuntimeError, match="Failed to initialize"), TestClient(app):
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

        with patch(
            "src.api.app.build_pipeline",
            side_effect=RuntimeError("Build failed"),
        ), pytest.raises(RuntimeError), TestClient(app):
            pass
