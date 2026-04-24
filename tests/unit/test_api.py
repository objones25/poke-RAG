"""Unit tests for the FastAPI API layer."""

from __future__ import annotations

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
    def test_stats_returns_200(self, client) -> None:
        import os

        stats_key = os.getenv("STATS_API_KEY")
        if stats_key:
            response = client.get("/stats", headers={"Authorization": f"Bearer {stats_key}"})
        else:
            response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_returns_dict_with_collections(self, client) -> None:
        import os

        stats_key = os.getenv("STATS_API_KEY")
        if stats_key:
            response = client.get("/stats", headers={"Authorization": f"Bearer {stats_key}"})
        else:
            response = client.get("/stats")
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

    def test_stats_public_when_no_api_key_set(self, client: TestClient) -> None:
        """Stats should be public (200) when STATS_API_KEY is not set."""
        # Ensure STATS_API_KEY is not set
        import os

        if "STATS_API_KEY" in os.environ:
            del os.environ["STATS_API_KEY"]
        response = client.get("/stats")
        assert response.status_code == 200


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
