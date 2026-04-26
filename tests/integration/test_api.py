"""Integration tests for the FastAPI HTTP layer."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.pipeline.types import PipelineResult
from src.types import RetrievalError


def _make_result(**overrides: Any) -> PipelineResult:
    defaults: dict[str, Any] = {
        "answer": "Pikachu is Electric-type.",
        "sources_used": ("pokeapi",),
        "num_chunks_used": 3,
        "model_name": "google/gemma-4-E4B-it",
        "query": "What type is Pikachu?",
        "confidence_score": 0.5,
    }
    defaults.update(overrides)
    return PipelineResult(**defaults)


@pytest.fixture()
def mock_pipeline(mocker):
    pipeline = mocker.MagicMock()
    loader = mocker.MagicMock()
    mock_qdrant_client = mocker.MagicMock()
    mocker.patch("src.api.app.build_pipeline", return_value=(pipeline, loader, mock_qdrant_client))
    return pipeline


@pytest.fixture()
def client(mock_pipeline, monkeypatch):
    # Disable rate limiting for integration tests to avoid flaky test behavior
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    with TestClient(app) as c:
        yield c


@pytest.mark.integration
class TestAPIResponseFormat:
    def test_query_response_content_type_is_json(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result()
        response = client.post("/query", json={"query": "What type is Pikachu?"})
        assert "application/json" in response.headers["content-type"]

    def test_health_response_content_type_is_json(self, client) -> None:
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_503_response_has_detail_key(self, client, mock_pipeline) -> None:
        mock_pipeline.query.side_effect = RetrievalError("index unavailable")
        response = client.post("/query", json={"query": "Pikachu?"})
        assert response.json()["detail"] == "Retrieval service unavailable"

    def test_422_pydantic_error_is_json(self, client) -> None:
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422
        assert "application/json" in response.headers["content-type"]

    def test_422_pipeline_value_error_has_detail(self, client, mock_pipeline) -> None:
        mock_pipeline.query.side_effect = ValueError("bad pipeline input")
        response = client.post("/query", json={"query": "valid query"})
        assert response.status_code == 422
        assert response.json()["detail"] == "Invalid input"


@pytest.mark.integration
class TestAPIRouting:
    def test_get_on_query_returns_405(self, client) -> None:
        response = client.get("/query")
        assert response.status_code == 405

    def test_openapi_schema_available(self, client) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_unknown_route_returns_404(self, client) -> None:
        response = client.get("/nonexistent")
        assert response.status_code == 404


@pytest.mark.integration
class TestQueryNormalisation:
    def test_leading_whitespace_stripped_before_pipeline(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result()
        client.post("/query", json={"query": "  What type is Pikachu?"})
        args, _ = mock_pipeline.query.call_args
        assert args[0] == "What type is Pikachu?"

    def test_trailing_whitespace_stripped_before_pipeline(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result()
        client.post("/query", json={"query": "What type is Pikachu?  "})
        args, _ = mock_pipeline.query.call_args
        assert args[0] == "What type is Pikachu?"

    def test_whitespace_only_query_returns_422(self, client, mock_pipeline) -> None:
        mock_pipeline.query.side_effect = ValueError("query must not be empty or whitespace-only")
        response = client.post("/query", json={"query": "   "})
        assert response.status_code == 422

    def test_confidence_score_in_response(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result(confidence_score=0.87)
        response = client.post("/query", json={"query": "What type is Pikachu?"}).json()
        assert response["confidence_score"] == 0.87


@pytest.fixture()
def mock_async_pipeline_dep(mocker):
    from unittest.mock import AsyncMock as _AsyncMock

    pipeline = _AsyncMock()
    loader = mocker.MagicMock()
    mock_async_client = _AsyncMock()
    mocker.patch(
        "src.api.app.build_async_pipeline",
        return_value=(pipeline, loader, mock_async_client),
    )
    return pipeline


@pytest.fixture()
def async_client(mock_async_pipeline_dep, monkeypatch, mocker):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("ASYNC_PIPELINE_ENABLED", "true")
    mocker.patch("src.api.app.QdrantClient")
    with TestClient(app) as c:
        yield c


@pytest.mark.integration
class TestAsyncPipelineWiring:
    def test_async_path_returns_200(self, async_client, mock_async_pipeline_dep) -> None:
        mock_async_pipeline_dep.query.return_value = _make_result()
        response = async_client.post("/query", json={"query": "What type is Pikachu?"})
        assert response.status_code == 200

    def test_async_path_calls_async_query(self, async_client, mock_async_pipeline_dep) -> None:
        mock_async_pipeline_dep.query.return_value = _make_result()
        async_client.post("/query", json={"query": "What type is Pikachu?"})
        mock_async_pipeline_dep.query.assert_called_once()

    def test_async_path_response_contains_answer(
        self, async_client, mock_async_pipeline_dep
    ) -> None:
        mock_async_pipeline_dep.query.return_value = _make_result()
        response = async_client.post("/query", json={"query": "What type is Pikachu?"}).json()
        assert response["answer"] == "Pikachu is Electric-type."

    def test_async_path_confidence_score_in_response(
        self, async_client, mock_async_pipeline_dep
    ) -> None:
        mock_async_pipeline_dep.query.return_value = _make_result(confidence_score=0.92)
        response = async_client.post("/query", json={"query": "What type is Pikachu?"}).json()
        assert response["confidence_score"] == 0.92

    def test_async_path_retrieval_error_returns_503(
        self, async_client, mock_async_pipeline_dep
    ) -> None:
        from src.types import RetrievalError

        mock_async_pipeline_dep.query.side_effect = RetrievalError("no docs")
        response = async_client.post("/query", json={"query": "test"})
        assert response.status_code == 503


@pytest.mark.integration
class TestStatsAuth:
    """S1: /stats must require authentication; endpoint must be inaccessible without a key."""

    def test_stats_returns_403_when_no_api_key_configured(
        self, client, monkeypatch
    ) -> None:
        monkeypatch.delenv("STATS_API_KEY", raising=False)
        response = client.get("/stats")
        assert response.status_code == 403

    def test_stats_returns_401_with_wrong_key(self, client, monkeypatch) -> None:
        monkeypatch.setenv("STATS_API_KEY", "correct-secret")
        response = client.get("/stats", headers={"Authorization": "Bearer wrong-key"})
        assert response.status_code == 401

    def test_stats_returns_401_with_missing_auth_header(
        self, client, monkeypatch
    ) -> None:
        monkeypatch.setenv("STATS_API_KEY", "correct-secret")
        response = client.get("/stats")
        assert response.status_code == 401


@pytest.mark.unit
class TestCORSOriginsHelper:
    """S2: _compute_cors_origins must not default to wildcard when env var is unset."""

    def test_returns_empty_list_when_env_var_is_none(self) -> None:
        from src.api.app import _compute_cors_origins

        origins, allow_credentials = _compute_cors_origins(None)
        assert origins == []
        assert allow_credentials is False

    def test_returns_empty_list_when_env_var_is_empty_string(self) -> None:
        from src.api.app import _compute_cors_origins

        origins, allow_credentials = _compute_cors_origins("")
        assert origins == []
        assert allow_credentials is False

    def test_returns_wildcard_list_when_explicitly_star(self) -> None:
        from src.api.app import _compute_cors_origins

        origins, allow_credentials = _compute_cors_origins("*")
        assert origins == ["*"]
        assert allow_credentials is False

    def test_returns_specific_origins_when_configured(self) -> None:
        from src.api.app import _compute_cors_origins

        origins, allow_credentials = _compute_cors_origins(
            "https://example.com,https://api.example.com"
        )
        assert origins == ["https://example.com", "https://api.example.com"]
        assert allow_credentials is True

    def test_strips_whitespace_from_origins(self) -> None:
        from src.api.app import _compute_cors_origins

        origins, _ = _compute_cors_origins("  https://a.com , https://b.com  ")
        assert origins == ["https://a.com", "https://b.com"]
