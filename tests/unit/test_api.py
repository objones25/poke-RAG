"""Unit tests for the FastAPI API layer."""

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
    }
    defaults.update(overrides)
    return PipelineResult(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def mock_pipeline(mocker):
    pipeline = mocker.MagicMock()
    loader = mocker.MagicMock()
    mocker.patch("src.api.app.build_pipeline", return_value=(pipeline, loader))
    return pipeline


@pytest.fixture()
def client(mock_pipeline):
    with TestClient(app) as c:
        yield c


@pytest.mark.unit
class TestHealthEndpoint:
    def test_health_returns_ok(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.unit
class TestQueryEndpoint:
    def test_happy_path_returns_200(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result()
        response = client.post("/query", json={"query": "What type is Pikachu?"})
        assert response.status_code == 200

    def test_response_shape_matches_pipeline_result(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result(
            sources_used=("bulbapedia", "pokeapi"),
            num_chunks_used=5,
        )
        body = client.post("/query", json={"query": "What type is Pikachu?"}).json()
        assert body["answer"] == "Pikachu is Electric-type."
        assert body["sources_used"] == ["bulbapedia", "pokeapi"]
        assert body["num_chunks_used"] == 5
        assert body["model_name"] == "google/gemma-2-2b-it"
        assert body["query"] == "What type is Pikachu?"

    def test_retrieval_error_returns_503(self, client, mock_pipeline) -> None:
        mock_pipeline.query.side_effect = RetrievalError("index unavailable")
        response = client.post("/query", json={"query": "What type is Pikachu?"})
        assert response.status_code == 503

    def test_retrieval_error_detail_in_response(self, client, mock_pipeline) -> None:
        mock_pipeline.query.side_effect = RetrievalError("index unavailable")
        response = client.post("/query", json={"query": "What type is Pikachu?"})
        assert "index unavailable" in response.json()["detail"]

    def test_empty_query_returns_422(self, client) -> None:
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

    def test_missing_query_field_returns_422(self, client) -> None:
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_whitespace_query_returns_422(self, client, mock_pipeline) -> None:
        mock_pipeline.query.side_effect = ValueError("query must not be empty or whitespace-only")
        response = client.post("/query", json={"query": "   "})
        assert response.status_code == 422

    def test_sources_filter_passed_to_pipeline(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result()
        client.post("/query", json={"query": "Stats?", "sources": ["pokeapi"]})
        _, kwargs = mock_pipeline.query.call_args
        assert kwargs["sources"] == ["pokeapi"]

    def test_sources_none_by_default(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result()
        client.post("/query", json={"query": "Stats?"})
        _, kwargs = mock_pipeline.query.call_args
        assert kwargs["sources"] is None

    def test_invalid_source_returns_422(self, client) -> None:
        response = client.post("/query", json={"query": "Stats?", "sources": ["wikipedia"]})
        assert response.status_code == 422

    def test_query_text_forwarded_to_pipeline(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result()
        client.post("/query", json={"query": "How fast is Jolteon?"})
        args, _ = mock_pipeline.query.call_args
        assert args[0] == "How fast is Jolteon?"
