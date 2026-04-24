from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from qdrant_client import AsyncQdrantClient, QdrantClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.api.dependencies import (
    build_async_pipeline,
    build_pipeline,
    get_async_pipeline,
    get_pipeline,
)
from src.api.models import QueryRequest, QueryResponse
from src.api.query_parser import parse_query
from src.config import Settings, _parse_bool
from src.generation.exceptions import GenerationError
from src.pipeline.rag_pipeline import AsyncRAGPipeline, RAGPipeline
from src.types import RetrievalError
from src.utils.logging import setup_logging

load_dotenv()  # populate os.environ before lifespan runs

_LOG = logging.getLogger(__name__)

_MAX_TRACKED_IPS = 10_000
_MAX_BODY_BYTES = 64 * 1024  # 64 KB — far above any valid query payload


def _get_client_ip(request: Request, trusted_proxy_count: int) -> str:
    """Return the real client IP, honouring X-Forwarded-For when behind trusted proxies."""
    if trusted_proxy_count > 0:
        xff = request.headers.get("X-Forwarded-For", "")
        ips = [ip.strip() for ip in xff.split(",") if ip.strip()]
        # Accept exactly trusted_proxy_count or trusted_proxy_count+1 IPs.
        # More IPs than that means an attacker injected extra entries — fall back to socket IP.
        if trusted_proxy_count <= len(ips) <= trusted_proxy_count + 1:
            return ips[-trusted_proxy_count]
    return request.client.host if request.client else "unknown"


class LatencyTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds X-Response-Time-Ms header with elapsed time in milliseconds."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiter for /query endpoint: 20 requests per minute per IP."""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 20,
        trusted_proxy_count: int | None = None,
    ) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        if trusted_proxy_count is not None:
            self.trusted_proxy_count = trusted_proxy_count
        else:
            raw = os.getenv("TRUSTED_PROXY_COUNT", "0")
            try:
                parsed = int(raw)
                if parsed < 0:
                    raise ValueError("must be non-negative")
            except ValueError:
                raise ValueError(
                    f"TRUSTED_PROXY_COUNT must be a non-negative integer, got: {raw!r}"
                ) from None
            self.trusted_proxy_count = parsed
        self.request_times: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self.enabled = _parse_bool(os.getenv("RATE_LIMIT_ENABLED"), "RATE_LIMIT_ENABLED", True)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not self.enabled or request.url.path == "/health":
            return await call_next(request)

        client_ip = _get_client_ip(request, self.trusted_proxy_count)
        now = time.time()
        window_start = now - 60

        async with self._lock:
            if client_ip in self.request_times:
                self.request_times[client_ip] = [
                    t for t in self.request_times[client_ip] if t > window_start
                ]
            else:
                # Evict oldest entry if at capacity
                if len(self.request_times) >= _MAX_TRACKED_IPS:
                    self.request_times.popitem(last=False)
                self.request_times[client_ip] = []

            if len(self.request_times[client_ip]) >= self.requests_per_minute:
                _LOG.warning("Rate limit exceeded for IP: %s", client_ip)
                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

            self.request_times[client_ip].append(now)

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds security headers to every response."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        if _parse_bool(os.getenv("HTTPS_ENABLED"), "HTTPS_ENABLED", False):
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects requests whose Content-Length exceeds max_bytes."""

    def __init__(self, app: ASGIApp, max_bytes: int = _MAX_BODY_BYTES) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        content_length = request.headers.get("Content-Length")
        if content_length is not None:
            try:
                size = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400, content={"detail": "Invalid Content-Length header"}
                )
            if size < 0 or size > self.max_bytes:
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging()
    loader = None
    async_qdrant_client: AsyncQdrantClient | None = None
    try:
        settings = Settings.from_env()
        if settings.async_pipeline_enabled:
            async_pipeline, loader, async_qdrant_client = build_async_pipeline()
            app.state.async_pipeline = async_pipeline
            api_key_str = (
                None
                if settings.qdrant_api_key is None
                else settings.qdrant_api_key.get_secret_value()
            )
            client: QdrantClient = QdrantClient(url=settings.qdrant_url, api_key=api_key_str)
        else:
            pipeline, loader, client = build_pipeline()
            app.state.pipeline = pipeline
        app.state.qdrant_client = client
        app.state.settings = settings
    except Exception as exc:
        _LOG.error("Failed to initialize RAG pipeline: %s", exc)
        raise RuntimeError("Failed to initialize pipeline. Check server logs for details.") from exc
    try:
        yield
    finally:
        if loader is not None:
            loader.unload()
        if async_qdrant_client is not None:
            await async_qdrant_client.close()
        try:
            from joblib.externals.loky import get_reusable_executor  # type: ignore[import-untyped]

            get_reusable_executor().shutdown(wait=True)
        except Exception as exc:
            _LOG.warning("Loky executor shutdown error: %s", exc, exc_info=True)


app = FastAPI(title="poke-RAG", lifespan=lifespan)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins == "*":
    origins = ["*"]
    _allow_credentials = False
else:
    origins = [origin.strip() for origin in allowed_origins.split(",")]
    _allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.add_middleware(LatencyTrackingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BodySizeLimitMiddleware)


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
    _LOG.error("Retrieval error: %s", exc, exc_info=True)
    return JSONResponse(status_code=503, content={"detail": "Retrieval service unavailable"})


@app.exception_handler(GenerationError)
async def generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
    _LOG.error("Generation error: %s", exc, exc_info=True)
    return JSONResponse(status_code=503, content={"detail": "Generation service unavailable"})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    _LOG.warning("Validation error: %s", exc)
    return JSONResponse(status_code=422, content={"detail": "Invalid input"})


@app.exception_handler(TimeoutError)
async def timeout_error_handler(request: Request, exc: TimeoutError) -> JSONResponse:
    _LOG.warning("Request timed out: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=504, content={"detail": "Request timed out"})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    _LOG.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
async def stats(request: Request) -> dict[str, bool]:
    stats_api_key = os.getenv("STATS_API_KEY")
    if stats_api_key:
        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {stats_api_key}"
        if not hmac.compare_digest(auth.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Unauthorized")
    client: QdrantClient | None = getattr(request.app.state, "qdrant_client", None)
    if client is None:
        raise RuntimeError("Qdrant client not initialized")
    collections = await asyncio.to_thread(client.get_collections)
    return {col.name: True for col in collections.collections}


@app.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    request: Request,
) -> QueryResponse:
    parsed = parse_query(body.query)
    settings: Settings = request.app.state.settings
    if settings.async_pipeline_enabled:
        async_pipeline: AsyncRAGPipeline = get_async_pipeline(request)
        result = await asyncio.wait_for(
            async_pipeline.query(parsed, sources=body.sources, entity_name=body.entity_name),
            timeout=settings.query_timeout_seconds,
        )
    else:
        sync_pipeline: RAGPipeline = get_pipeline(request)
        result = await asyncio.wait_for(
            asyncio.to_thread(
                sync_pipeline.query, parsed, sources=body.sources, entity_name=body.entity_name
            ),
            timeout=settings.query_timeout_seconds,
        )
    return QueryResponse(
        answer=result.answer,
        sources_used=list(result.sources_used),
        num_chunks_used=result.num_chunks_used,
        model_name=result.model_name,
        query=result.query,
        confidence_score=result.confidence_score,
    )
