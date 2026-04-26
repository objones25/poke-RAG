"""Unit tests for the FastAPI API layer."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.generation.exceptions import GenerationError
from src.pipeline.types import PipelineResult
from src.types import RetrievalError


def _make_result(**overrides: object) -> PipelineResult:
    defaults: dict[str, object] = {
        "answer": "Pikachu is Electric-type.",
        "sources_used": ("pokeapi",),
        "num_chunks_used": 3,
        "model_name": "google/gemma-4-E4B-it",
        "query": "What type is Pikachu?",
        "confidence_score": 0.85,
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
def client(mock_pipeline):
    with TestClient(app) as c:
        yield c


@pytest.mark.unit
class TestStatsEndpoint:
    def test_stats_returns_200_with_valid_key(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STATS_API_KEY", "test-key")
        response = client.get("/stats", headers={"Authorization": "Bearer test-key"})
        assert response.status_code == 200

    def test_stats_returns_dict_with_collections(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STATS_API_KEY", "test-key")
        response = client.get("/stats", headers={"Authorization": "Bearer test-key"})
        result = response.json()
        assert isinstance(result, dict)


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
        assert body["model_name"] == "google/gemma-4-E4B-it"
        assert body["query"] == "What type is Pikachu?"

    def test_retrieval_error_returns_503(self, client, mock_pipeline) -> None:
        mock_pipeline.query.side_effect = RetrievalError("index unavailable")
        response = client.post("/query", json={"query": "What type is Pikachu?"})
        assert response.status_code == 503

    def test_retrieval_error_detail_in_response(self, client, mock_pipeline) -> None:
        mock_pipeline.query.side_effect = RetrievalError("index unavailable")
        response = client.post("/query", json={"query": "What type is Pikachu?"})
        assert response.json()["detail"] == "Retrieval service unavailable"

    def test_generation_error_returns_503(self, client, mock_pipeline) -> None:
        mock_pipeline.query.side_effect = GenerationError("inference failed")
        response = client.post("/query", json={"query": "What type is Pikachu?"})
        assert response.status_code == 503

    def test_generation_error_detail_in_response(self, client, mock_pipeline) -> None:
        mock_pipeline.query.side_effect = GenerationError("inference failed")
        response = client.post("/query", json={"query": "What type is Pikachu?"})
        assert response.json()["detail"] == "Generation service unavailable"

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

    def test_confidence_score_in_response(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result(confidence_score=0.92)
        response = client.post("/query", json={"query": "What type is Pikachu?"}).json()
        assert response["confidence_score"] == 0.92

    def test_confidence_score_in_response_default(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result(confidence_score=0.75)
        response = client.post("/query", json={"query": "What type is Pikachu?"}).json()
        assert response["confidence_score"] == 0.75

    def test_entity_name_forwarded_to_pipeline(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result()
        client.post("/query", json={"query": "Stats?", "entity_name": "Pikachu"})
        _, kwargs = mock_pipeline.query.call_args
        assert kwargs["entity_name"] == "Pikachu"

    def test_entity_name_none_by_default(self, client, mock_pipeline) -> None:
        mock_pipeline.query.return_value = _make_result()
        client.post("/query", json={"query": "Stats?"})
        _, kwargs = mock_pipeline.query.call_args
        assert kwargs["entity_name"] is None


@pytest.mark.unit
class TestSecurityHeaders:
    def test_security_headers_present_on_health(self, client) -> None:
        response = client.get("/health")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "strict-origin-when-cross-origin" in response.headers["Referrer-Policy"]

    def test_content_security_policy_present(self, client) -> None:
        response = client.get("/health")
        assert "Content-Security-Policy" in response.headers


@pytest.mark.unit
class TestBodySizeLimitMiddleware:
    def test_returns_413_when_body_too_large(self) -> None:
        from fastapi import FastAPI

        from src.api.app import BodySizeLimitMiddleware

        test_app = FastAPI()
        test_app.add_middleware(BodySizeLimitMiddleware, max_bytes=10)

        @test_app.post("/upload")
        async def _upload() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            response = c.post("/upload", content=b"a" * 11)
            assert response.status_code == 413

    def test_allows_request_within_limit(self) -> None:
        from fastapi import FastAPI

        from src.api.app import BodySizeLimitMiddleware

        test_app = FastAPI()
        test_app.add_middleware(BodySizeLimitMiddleware, max_bytes=100)

        @test_app.post("/upload")
        async def _upload() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            response = c.post("/upload", content=b"small body")
            assert response.status_code == 200


@pytest.mark.unit
class TestRateLimitMiddleware:
    def test_returns_429_after_limit_exceeded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        test_app = FastAPI()
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=2)

        @test_app.post("/query")
        async def _query() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            c.post("/query")
            c.post("/query")
            response = c.post("/query")
            assert response.status_code == 429

    def test_allows_requests_within_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        test_app = FastAPI()
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=5)

        @test_app.post("/query")
        async def _query() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            for _ in range(5):
                response = c.post("/query")
                assert response.status_code == 200

    def test_rate_limit_disabled_with_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
        test_app = FastAPI()
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=1)

        @test_app.post("/query")
        async def _query() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            c.post("/query")
            c.post("/query")
            response = c.post("/query")
            assert response.status_code == 200  # No rate limit enforced

    def test_rate_limit_disabled_with_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "0")
        test_app = FastAPI()
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=1)

        @test_app.post("/query")
        async def _query() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            c.post("/query")
            response = c.post("/query")
            assert response.status_code == 200  # No rate limit enforced

    def test_rate_limit_disabled_with_no(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "no")
        test_app = FastAPI()
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=1)

        @test_app.post("/query")
        async def _query() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            c.post("/query")
            response = c.post("/query")
            assert response.status_code == 200  # No rate limit enforced

    def test_rate_limit_enabled_with_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
        test_app = FastAPI()
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=1)

        @test_app.post("/query")
        async def _query() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            c.post("/query")
            response = c.post("/query")
            assert response.status_code == 429  # Rate limit enforced

    def test_rate_limit_enabled_with_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "yes")
        test_app = FastAPI()
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=1)

        @test_app.post("/query")
        async def _query() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            c.post("/query")
            response = c.post("/query")
            assert response.status_code == 429  # Rate limit enforced

    def test_rate_limit_invalid_value_raises_valueerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import FastAPI

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "maybe")
        test_app = FastAPI()
        with pytest.raises(ValueError, match="RATE_LIMIT_ENABLED"):
            RateLimitMiddleware(app=test_app)


@pytest.mark.unit
class TestGetClientIp:
    def test_returns_client_host_without_proxy(self) -> None:
        from src.api.app import _get_client_ip

        request = MagicMock()
        request.client.host = "1.2.3.4"
        request.headers.get.return_value = ""
        assert _get_client_ip(request, trusted_proxy_count=0) == "1.2.3.4"

    def test_returns_xff_last_entry_with_one_trusted_proxy(self) -> None:
        from src.api.app import _get_client_ip

        request = MagicMock()
        request.headers.get.return_value = "evil-spoof, 5.6.7.8"
        assert _get_client_ip(request, trusted_proxy_count=1) == "5.6.7.8"

    def test_falls_back_to_client_host_when_xff_too_short(self) -> None:
        from src.api.app import _get_client_ip

        request = MagicMock()
        request.client.host = "9.10.11.12"
        request.headers.get.return_value = ""
        assert _get_client_ip(request, trusted_proxy_count=1) == "9.10.11.12"

    def test_returns_client_host_when_no_client(self) -> None:
        from src.api.app import _get_client_ip

        request = MagicMock()
        request.client = None
        request.headers.get.return_value = ""
        assert _get_client_ip(request, trusted_proxy_count=0) == "unknown"


@pytest.mark.unit
class TestBodySizeLimitInvalidContentLength:
    def test_returns_400_on_invalid_content_length(self) -> None:
        from fastapi import FastAPI

        from src.api.app import BodySizeLimitMiddleware

        test_app = FastAPI()
        test_app.add_middleware(BodySizeLimitMiddleware, max_bytes=100)

        @test_app.post("/upload")
        async def _upload() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            response = c.post("/upload", headers={"Content-Length": "abc"})
            assert response.status_code == 400
            assert "Invalid Content-Length" in response.json()["detail"]

    def test_returns_400_on_float_content_length(self) -> None:
        from fastapi import FastAPI

        from src.api.app import BodySizeLimitMiddleware

        test_app = FastAPI()
        test_app.add_middleware(BodySizeLimitMiddleware, max_bytes=100)

        @test_app.post("/upload")
        async def _upload() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            response = c.post("/upload", headers={"Content-Length": "12.5"})
            assert response.status_code == 400


@pytest.mark.unit
class TestGetClientIpXFFSpoofing:
    def test_falls_back_to_client_host_on_xff_too_many_ips(self) -> None:
        from src.api.app import _get_client_ip

        request = MagicMock()
        request.client.host = "1.2.3.4"
        # More IPs than trusted_proxy_count + 1 should trigger fallback
        request.headers.get.return_value = "fake1, fake2, fake3, fake4, 5.6.7.8"
        assert _get_client_ip(request, trusted_proxy_count=1) == "1.2.3.4"


@pytest.mark.unit
class TestXFFUndeclaredProxyWarning:
    """S4: Warn once when XFF header is present but TRUSTED_PROXY_COUNT=0."""

    def test_warning_emitted_when_xff_present_and_no_proxy_declared(
        self, mock_pipeline, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from src.api.app import RateLimitMiddleware

        # Reset the one-time flag so this test is hermetic
        RateLimitMiddleware._xff_warning_issued = False

        with (
            caplog.at_level(logging.WARNING, logger="src.api.app"),
            TestClient(app, raise_server_exceptions=False) as c,
        ):
            c.get("/health", headers={"X-Forwarded-For": "1.2.3.4"})

        assert any(
            "X-Forwarded-For" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        ), "S4: expected WARNING about XFF present but TRUSTED_PROXY_COUNT=0"

    def test_warning_emitted_only_once(
        self, mock_pipeline, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from src.api.app import RateLimitMiddleware

        RateLimitMiddleware._xff_warning_issued = False

        with (
            caplog.at_level(logging.WARNING, logger="src.api.app"),
            TestClient(app, raise_server_exceptions=False) as c,
        ):
            c.get("/health", headers={"X-Forwarded-For": "1.2.3.4"})
            c.get("/health", headers={"X-Forwarded-For": "5.6.7.8"})

        xff_warnings = [
            r for r in caplog.records
            if "X-Forwarded-For" in r.message and r.levelno == logging.WARNING
        ]
        assert len(xff_warnings) == 1, "S4: warning must be emitted exactly once"


@pytest.mark.unit
class TestQueryEndpointTimeout:
    def test_query_timeout_returns_504(self, client, mock_pipeline) -> None:
        mock_pipeline.query.side_effect = TimeoutError()
        response = client.post("/query", json={"query": "What type is Pikachu?"})
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()


@pytest.mark.unit
class TestSecurityHeadersHSTS:
    def test_hsts_header_present_when_https_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI

        from src.api.app import SecurityHeadersMiddleware

        monkeypatch.setenv("HTTPS_ENABLED", "true")
        test_app = FastAPI()
        test_app.add_middleware(SecurityHeadersMiddleware)

        @test_app.get("/health")
        async def _health() -> dict[str, str]:
            return {"status": "ok"}

        with TestClient(test_app) as c:
            response = c.get("/health")
            assert "Strict-Transport-Security" in response.headers
            assert "max-age=31536000" in response.headers["Strict-Transport-Security"]

    def test_hsts_header_absent_when_https_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI

        from src.api.app import SecurityHeadersMiddleware

        monkeypatch.setenv("HTTPS_ENABLED", "false")
        test_app = FastAPI()
        test_app.add_middleware(SecurityHeadersMiddleware)

        @test_app.get("/health")
        async def _health() -> dict[str, str]:
            return {"status": "ok"}

        with TestClient(test_app) as c:
            response = c.get("/health")
            assert "Strict-Transport-Security" not in response.headers


@pytest.mark.unit
class TestHSTSStartupWarning:
    """S5: Warn at startup when HSTS is disabled (HTTPS_ENABLED not true)."""

    def test_warning_emitted_when_https_not_enabled(
        self, mock_pipeline, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        monkeypatch.delenv("HTTPS_ENABLED", raising=False)

        with (
            caplog.at_level(logging.WARNING, logger="src.api.app"),
            TestClient(app, raise_server_exceptions=False),
        ):
            pass

        assert any(
            "HSTS" in r.message or "HTTPS_ENABLED" in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        ), "S5: expected startup WARNING about HSTS being disabled"

    def test_no_warning_when_https_enabled(
        self, mock_pipeline, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        monkeypatch.setenv("HTTPS_ENABLED", "true")

        with (
            caplog.at_level(logging.WARNING, logger="src.api.app"),
            TestClient(app, raise_server_exceptions=False),
        ):
            pass

        hsts_warnings = [
            r for r in caplog.records
            if ("HSTS" in r.message or "HTTPS_ENABLED" in r.message)
            and r.levelno == logging.WARNING
        ]
        assert hsts_warnings == [], "S5: no HSTS warning expected when HTTPS_ENABLED=true"


@pytest.mark.unit
class TestHealthEndpointMetadata:
    def test_health_has_response_model_and_status_code(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.unit
class TestStatsApiKeyTiming:
    """Test that stats endpoint uses constant-time comparison for API key."""

    def test_stats_uses_hmac_compare_digest(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that hmac.compare_digest is used (not plain ==) for timing-safe comparison."""
        from unittest.mock import patch

        monkeypatch.setenv("STATS_API_KEY", "my-secret-key")

        with patch("src.api.app.hmac.compare_digest") as mock_compare:
            mock_compare.return_value = False
            response = client.get("/stats", headers={"Authorization": "Bearer wrong-key"})
            assert response.status_code == 401
            mock_compare.assert_called_once()

    def test_stats_always_calls_compare_digest(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that compare_digest is called even for malformed auth headers (constant-time)."""
        from unittest.mock import patch

        monkeypatch.setenv("STATS_API_KEY", "my-secret-key")

        with patch("src.api.app.hmac.compare_digest") as mock_compare:
            mock_compare.return_value = False
            # Send malformed header (no "Bearer " prefix)
            response = client.get("/stats", headers={"Authorization": "notabearer"})
            assert response.status_code == 401
            # compare_digest MUST be called even for malformed headers to prevent timing attacks
            mock_compare.assert_called_once()

    def test_stats_rejects_malformed_auth_header(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Malformed auth header (not starting with 'Bearer ') should return 401."""
        monkeypatch.setenv("STATS_API_KEY", "my-secret-key")
        response = client.get("/stats", headers={"Authorization": "notabearer"})
        assert response.status_code == 401

    def test_stats_accepts_correct_api_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stats with correct API key in Bearer token should return 200."""
        monkeypatch.setenv("STATS_API_KEY", "my-secret-key")
        response = client.get("/stats", headers={"Authorization": "Bearer my-secret-key"})
        assert response.status_code == 200

    def test_stats_rejects_wrong_api_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stats with wrong API key should return 401."""
        monkeypatch.setenv("STATS_API_KEY", "my-secret-key")
        response = client.get("/stats", headers={"Authorization": "Bearer wrong-key"})
        assert response.status_code == 401

    def test_stats_returns_403_when_no_api_key_configured(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stats must return 403 when STATS_API_KEY is not configured (S1 fix)."""
        monkeypatch.delenv("STATS_API_KEY", raising=False)
        response = client.get("/stats")
        assert response.status_code == 403


@pytest.mark.unit
class TestStatsEndpointRateLimit:
    """Test that /stats endpoint is rate-limited."""

    def test_stats_rate_limit_returns_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stats endpoint should be rate-limited like /query."""
        from fastapi import FastAPI

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        test_app = FastAPI()
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=2)

        @test_app.get("/stats")
        async def _stats() -> dict[str, bool]:
            return {"pokeapi": True}

        with TestClient(test_app) as c:
            c.get("/stats")
            c.get("/stats")
            response = c.get("/stats")
            assert response.status_code == 429


@pytest.mark.unit
class TestRateLimitMiddlewareTrustedProxyValidation:
    def test_trusted_proxy_count_non_integer_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import FastAPI

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("TRUSTED_PROXY_COUNT", "abc")
        test_app = FastAPI()
        with pytest.raises(ValueError, match="TRUSTED_PROXY_COUNT.*non-negative integer"):
            RateLimitMiddleware(app=test_app)

    def test_trusted_proxy_count_negative_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import FastAPI

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("TRUSTED_PROXY_COUNT", "-1")
        test_app = FastAPI()
        with pytest.raises(ValueError, match="TRUSTED_PROXY_COUNT.*non-negative integer"):
            RateLimitMiddleware(app=test_app)


@pytest.mark.unit
class TestRateLimitMiddlewareXForwardedForInjection:
    def test_xff_extraction_with_trusted_proxy_count_one(self) -> None:
        from src.api.app import _get_client_ip

        request = MagicMock()
        request.client.host = "socket_ip"
        request.headers.get.return_value = "client,proxy"
        result = _get_client_ip(request, trusted_proxy_count=1)
        assert result == "proxy"

    def test_xff_extraction_with_trusted_proxy_count_two_exact_ips(self) -> None:
        from src.api.app import _get_client_ip

        request = MagicMock()
        request.client.host = "socket_ip"
        request.headers.get.return_value = "client,proxy1,proxy2"
        result = _get_client_ip(request, trusted_proxy_count=2)
        assert result == "proxy1"

    def test_xff_fallback_when_too_many_ips(self) -> None:
        from src.api.app import _get_client_ip

        request = MagicMock()
        request.client.host = "socket_ip"
        request.headers.get.return_value = "fake1,fake2,fake3,real_client,proxy"
        result = _get_client_ip(request, trusted_proxy_count=1)
        assert result == "socket_ip"


@pytest.mark.unit
class TestRateLimitMiddlewareCapacityEviction:
    def test_ip_eviction_at_capacity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        from fastapi import FastAPI

        test_app = FastAPI()
        middleware = RateLimitMiddleware(app=test_app, requests_per_minute=1000)

        small_capacity = 10
        for i in range(small_capacity):
            ip = f"192.168.0.{i}"
            middleware.request_times[ip] = [1.0]

        assert len(middleware.request_times) == small_capacity
        first_ip = list(middleware.request_times.keys())[0]

        for i in range(small_capacity, small_capacity + 2):
            ip = f"192.168.0.{i + 100}"
            if len(middleware.request_times) >= small_capacity:
                middleware.request_times.popitem(last=False)
            middleware.request_times[ip] = [1.0]

        new_first = list(middleware.request_times.keys())[0]
        assert first_ip != new_first


@pytest.mark.unit
class TestRateLimitMiddlewareLRUEviction:
    """B8: dispatch() must call move_to_end() for existing IPs so FIFO eviction cannot
    purge recently-active rate-limited IPs and let them bypass the limit."""

    @staticmethod
    def _get_app_module():
        import importlib

        return importlib.import_module("src.api.app")

    def test_existing_ip_moved_to_tail_on_dispatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """dispatch() in the existing-IP branch must call move_to_end() so that
        recently-active IPs migrate to the tail and are not the first victims of eviction.

        RED before fix: victim stays at HEAD (insertion order).
        GREEN after fix: dispatch() calls move_to_end() → victim at TAIL.
        """
        import asyncio
        from unittest.mock import MagicMock

        from src.api.app import RateLimitMiddleware

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")

        middleware = RateLimitMiddleware(
            app=MagicMock(),
            requests_per_minute=100,
            trusted_proxy_count=0,
        )

        # victim inserted first (HEAD), two more IPs push it away from TAIL.
        middleware.request_times["victim"] = [time.time()]
        middleware.request_times["other-1"] = [time.time()]
        middleware.request_times["other-2"] = [time.time()]

        # Call actual dispatch() for victim — existing-IP branch should call move_to_end().
        request = MagicMock()
        request.url.path = "/query"
        request.client.host = "victim"
        request.headers.get.return_value = ""

        async def call_next(req: object) -> object:
            from starlette.responses import Response

            return Response(status_code=200)

        asyncio.run(middleware.dispatch(request, call_next))

        keys = list(middleware.request_times.keys())
        assert keys[-1] == "victim", (
            f"B8: dispatch() must call move_to_end() for existing IPs — got order: {keys}"
        )

    def test_rate_limited_ip_stays_blocked_after_capacity_flood(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B8 core: victim must stay rate-limited after FIFO eviction would reset its state.

        Sequence (CAPACITY=3, RPM=2):
          1. victim: 2 requests (fills budget) — inserted first, at HEAD.
          2. flood-0, flood-1 added to reach CAPACITY; victim still at HEAD.
          3. victim: 3rd request → 429. FIX calls move_to_end() → victim to TAIL.
          4. flood-2 triggers eviction. BUG evicts victim (HEAD); FIX evicts flood-0.
          5. victim: 4th request. BUG → 200 (bypass); FIX → 429 (still blocked).

        Uses trusted_proxy_count=1 + X-Forwarded-For so each header IP is tracked
        separately (TestClient always presents the same socket IP otherwise).
        """
        app_module = self._get_app_module()
        CAPACITY = 3
        RPM = 2
        monkeypatch.setattr(app_module, "_MAX_TRACKED_IPS", CAPACITY)
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.app import RateLimitMiddleware

        test_app = FastAPI()
        test_app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=RPM,
            trusted_proxy_count=1,  # enables X-Forwarded-For so each test IP is distinct
        )

        @test_app.post("/query")
        async def _query() -> dict[str, bool]:
            return {"ok": True}

        def xff(ip: str) -> dict[str, str]:
            return {"X-Forwarded-For": ip}

        with TestClient(test_app) as c:
            # Step 1: victim fills its rate-limit budget (inserted first → HEAD)
            for _ in range(RPM):
                c.post("/query", headers=xff("victim"))

            # Step 2: flood-0, flood-1 fill dict to CAPACITY (victim still at HEAD)
            c.post("/query", headers=xff("flood-0"))
            c.post("/query", headers=xff("flood-1"))

            # Step 3: victim's 3rd request is blocked; FIX moves it to TAIL here
            rate_limited = c.post("/query", headers=xff("victim"))
            assert rate_limited.status_code == 429, "pre-condition: victim must be rate-limited"

            # Step 4: flood-2 triggers eviction
            # BUG: HEAD = victim → evicted. FIX: HEAD = flood-0 → evicted, victim survives.
            c.post("/query", headers=xff("flood-2"))

            # Step 5: victim's request must still be blocked
            after_eviction = c.post("/query", headers=xff("victim"))
            assert after_eviction.status_code == 429, (
                "B8: rate-limited IP bypassed limit after FIFO eviction — "
                "dispatch() must call move_to_end() in the existing-IP branch"
            )


@pytest.mark.unit
class TestBodySizeLimitMiddlewareEdgeCases:
    def test_content_length_exactly_at_limit(self) -> None:
        from fastapi import FastAPI

        from src.api.app import BodySizeLimitMiddleware

        test_app = FastAPI()
        test_app.add_middleware(BodySizeLimitMiddleware, max_bytes=65536)

        @test_app.post("/upload")
        async def _upload() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            response = c.post("/upload", headers={"Content-Length": "65536"})
            assert response.status_code == 200

    def test_content_length_one_byte_over_limit(self) -> None:
        from fastapi import FastAPI

        from src.api.app import BodySizeLimitMiddleware

        test_app = FastAPI()
        test_app.add_middleware(BodySizeLimitMiddleware, max_bytes=65536)

        @test_app.post("/upload")
        async def _upload() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            response = c.post("/upload", headers={"Content-Length": "65537"})
            assert response.status_code == 413

    def test_content_length_zero(self) -> None:
        from fastapi import FastAPI

        from src.api.app import BodySizeLimitMiddleware

        test_app = FastAPI()
        test_app.add_middleware(BodySizeLimitMiddleware, max_bytes=100)

        @test_app.post("/upload")
        async def _upload() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            response = c.post("/upload", headers={"Content-Length": "0"})
            assert response.status_code == 200

    def test_content_length_negative(self) -> None:
        from fastapi import FastAPI

        from src.api.app import BodySizeLimitMiddleware

        test_app = FastAPI()
        test_app.add_middleware(BodySizeLimitMiddleware, max_bytes=100)

        @test_app.post("/upload")
        async def _upload() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(test_app) as c:
            response = c.post("/upload", headers={"Content-Length": "-1"})
            assert response.status_code == 413


@pytest.mark.unit
class TestGlobalExceptionHandlerStackTrace:
    """S8: global exception handler must only log full stack traces at DEBUG level."""

    def test_exc_info_not_logged_at_error_level(self, monkeypatch) -> None:
        import logging
        from unittest.mock import patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.app import global_exception_handler

        test_app = FastAPI()
        test_app.add_exception_handler(Exception, global_exception_handler)

        @test_app.get("/boom")
        async def _boom() -> None:
            raise RuntimeError("intentional test error")

        log_calls: list[dict] = []
        original_error = logging.Logger.error

        def capture_error(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def]
            log_calls.append({"msg": msg, "kwargs": kwargs})
            original_error(self, msg, *args, **kwargs)

        app_logger = logging.getLogger("src.api.app")
        original_level = app_logger.level
        try:
            app_logger.setLevel(logging.WARNING)
            with (
                patch.object(logging.Logger, "error", capture_error),
                TestClient(test_app, raise_server_exceptions=False) as c,
            ):
                c.get("/boom")
        finally:
            app_logger.setLevel(original_level)

        exc_info_calls = [
            call for call in log_calls if call["kwargs"].get("exc_info")
        ]
        assert exc_info_calls == [], (
            "S8: exc_info=True must not be logged when logger level is above DEBUG"
        )

    def test_exc_info_logged_at_debug_level(self, monkeypatch) -> None:
        import logging
        from unittest.mock import patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.app import global_exception_handler

        test_app = FastAPI()
        test_app.add_exception_handler(Exception, global_exception_handler)

        @test_app.get("/boom")
        async def _boom() -> None:
            raise RuntimeError("intentional test error")

        log_calls: list[dict] = []
        original_error = logging.Logger.error

        def capture_error(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def]
            log_calls.append({"msg": msg, "kwargs": kwargs})
            original_error(self, msg, *args, **kwargs)

        app_logger = logging.getLogger("src.api.app")
        original_level = app_logger.level
        try:
            app_logger.setLevel(logging.DEBUG)
            with (
                patch.object(logging.Logger, "error", capture_error),
                TestClient(test_app, raise_server_exceptions=False) as c,
            ):
                c.get("/boom")
        finally:
            app_logger.setLevel(original_level)

        exc_info_calls = [
            call for call in log_calls if call["kwargs"].get("exc_info")
        ]
        assert exc_info_calls, (
            "S8: exc_info=True must be logged when logger level is DEBUG"
        )


@pytest.mark.unit
class TestQueryAuditLogging:
    """S11: /query must emit an INFO-level audit log with entity_name and sources."""

    def test_audit_log_emitted_with_entity_name_and_sources(
        self, mock_pipeline, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        mock_pipeline.query.return_value = _make_result(
            query="What moves does Pikachu learn?",
            sources_used=("pokeapi",),
        )
        with (
            caplog.at_level(logging.INFO, logger="src.api.app"),
            TestClient(app, raise_server_exceptions=False) as c,
        ):
            c.post(
                "/query",
                json={
                    "query": "What moves does Pikachu learn?",
                    "entity_name": "Pikachu",
                    "sources": ["pokeapi"],
                },
            )

        assert any(
            "Pikachu" in r.message and "pokeapi" in r.message
            for r in caplog.records
            if r.levelno == logging.INFO
        ), "S11: expected INFO audit log containing entity_name and sources"

    def test_audit_log_emitted_without_optional_fields(
        self, mock_pipeline, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        mock_pipeline.query.return_value = _make_result(
            query="What type is Bulbasaur?",
        )
        with (
            caplog.at_level(logging.INFO, logger="src.api.app"),
            TestClient(app, raise_server_exceptions=False) as c,
        ):
            c.post("/query", json={"query": "What type is Bulbasaur?"})

        assert any(
            "query" in r.message.lower()
            for r in caplog.records
            if r.levelno == logging.INFO
        ), "S11: expected INFO audit log for query without entity_name/sources"
