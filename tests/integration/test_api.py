"""Integration tests for the FastAPI HTTP layer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.pipeline.types import PipelineResult
from src.types import RetrievalError


def _make_result(**overrides: object) -> PipelineResult:
    defaults: dict[str, object] = {
        "answer": "Pikachu is Electric-type.",
        "sources_used": ("pokeapi",),
        "num_chunks_used": 3,
        "model_name": "google/gemma-2-2b-it",
        "query": "What type is Pikachu?",
        "confidence_score": None,
    }
    defaults.update(overrides)
    return PipelineResult(**defaults)  # type: ignore[arg-type]


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
