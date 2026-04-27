"""Integration tests for the POST /query/stream SSE endpoint."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.types import RetrievalError


async def _async_gen(*tokens: str) -> AsyncGenerator[str, None]:
    for token in tokens:
        yield token


@pytest.fixture()
def mock_async_pipeline(mocker):
    pipeline = MagicMock()
    loader = MagicMock()
    # async_qdrant_client.close() is awaited in the lifespan teardown
    async_qdrant_client = AsyncMock()
    mocker.patch(
        "src.api.app.build_async_pipeline",
        return_value=(pipeline, loader, async_qdrant_client),
    )
    return pipeline


@pytest.fixture()
def client(mock_async_pipeline, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("ASYNC_PIPELINE_ENABLED", "true")
    with TestClient(app) as c:
        yield c


def _parse_sse_events(text: str) -> list[dict[str, Any]]:
    """Parse SSE stream text into a list of data dicts."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            raw = line[len("data: ") :]
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                events.append({"raw": raw})
    return events


@pytest.mark.integration
class TestStreamEndpointBasic:
    def test_stream_endpoint_returns_200(self, client, mock_async_pipeline):
        mock_async_pipeline.stream_query.return_value = _async_gen("hello")
        response = client.post("/query/stream", json={"query": "What is Pikachu?"})
        assert response.status_code == 200

    def test_stream_endpoint_content_type_is_event_stream(self, client, mock_async_pipeline):
        mock_async_pipeline.stream_query.return_value = _async_gen("hello")
        response = client.post("/query/stream", json={"query": "What is Pikachu?"})
        assert "text/event-stream" in response.headers["content-type"]

    def test_stream_endpoint_yields_tokens_as_sse_events(self, client, mock_async_pipeline):
        mock_async_pipeline.stream_query.return_value = _async_gen("Pika", "chu")
        response = client.post("/query/stream", json={"query": "What is Pikachu?"})
        events = _parse_sse_events(response.text)
        tokens = [e["token"] for e in events if "token" in e]
        assert tokens == ["Pika", "chu"]

    def test_stream_endpoint_empty_query_returns_422(self, client, mock_async_pipeline):
        response = client.post("/query/stream", json={"query": ""})
        assert response.status_code == 422

    def test_stream_endpoint_missing_query_returns_422(self, client, mock_async_pipeline):
        response = client.post("/query/stream", json={})
        assert response.status_code == 422


@pytest.mark.integration
class TestStreamEndpointErrorHandling:
    def test_retrieval_error_closes_stream_with_error_event(self, client, mock_async_pipeline):
        async def _raise():
            raise RetrievalError("index unavailable")
            yield  # make it an async generator

        mock_async_pipeline.stream_query.return_value = _raise()
        response = client.post("/query/stream", json={"query": "What is Pikachu?"})
        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        error_events = [e for e in events if "error" in e]
        assert len(error_events) >= 1

    def test_done_event_sent_after_all_tokens(self, client, mock_async_pipeline):
        mock_async_pipeline.stream_query.return_value = _async_gen("hello", " world")
        response = client.post("/query/stream", json={"query": "test"})
        events = _parse_sse_events(response.text)
        assert events[-1] == {"done": True}


@pytest.mark.integration
class TestStreamEndpointAsyncGuard:
    """B1: /query/stream must return 501 when async pipeline is disabled."""

    def test_stream_returns_501_when_async_disabled(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("ASYNC_PIPELINE_ENABLED", "false")
        mock_pipeline = MagicMock()
        mock_loader = MagicMock()
        mock_client = MagicMock()
        build_rv = (mock_pipeline, mock_loader, mock_client)
        with (
            patch("src.api.app.build_pipeline", return_value=build_rv),
            TestClient(app) as c,
        ):
            response = c.post("/query/stream", json={"query": "What is Pikachu?"})
        assert response.status_code == 501

    def test_stream_returns_501_detail_mentions_env_var(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("ASYNC_PIPELINE_ENABLED", "false")
        mock_pipeline = MagicMock()
        mock_loader = MagicMock()
        mock_client = MagicMock()
        build_rv = (mock_pipeline, mock_loader, mock_client)
        with (
            patch("src.api.app.build_pipeline", return_value=build_rv),
            TestClient(app) as c,
        ):
            response = c.post("/query/stream", json={"query": "What is Pikachu?"})
        detail = response.json()["detail"]
        assert "ASYNC_PIPELINE_ENABLED" in detail
