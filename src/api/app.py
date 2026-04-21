from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from qdrant_client import QdrantClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.api.dependencies import build_pipeline, get_pipeline
from src.api.models import QueryRequest, QueryResponse
from src.api.query_parser import parse_query
from src.pipeline.rag_pipeline import RAGPipeline
from src.types import RetrievalError
from src.utils.logging import setup_logging

load_dotenv()  # populate os.environ before lifespan runs

_LOG = logging.getLogger(__name__)


class LatencyTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds X-Response-Time-Ms header with elapsed time in milliseconds."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiter for /query endpoint: 20 requests per minute per IP."""

    def __init__(self, app: ASGIApp, requests_per_minute: int = 20) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_times: dict[str, list[float]] = defaultdict(list)
        # Allow disabling rate limiting via environment variable (useful for testing)
        self.enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() != "false"

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if not self.enabled or request.url.path != "/query":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60

        self.request_times[client_ip] = [
            t for t in self.request_times[client_ip] if t > window_start
        ]

        if len(self.request_times[client_ip]) >= self.requests_per_minute:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

        self.request_times[client_ip].append(now)

        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging()
    try:
        pipeline, loader, client = build_pipeline()
        app.state.pipeline = pipeline
        app.state.qdrant_client = client
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize RAG pipeline: {exc}") from exc
    try:
        yield
    finally:
        loader.unload()
        try:
            from joblib.externals.loky import get_reusable_executor  # type: ignore[import-untyped]

            get_reusable_executor().shutdown(wait=True)
        except Exception as exc:
            _LOG.warning("Loky executor shutdown error: %s", exc)


app = FastAPI(title="poke-RAG", lifespan=lifespan)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins == "*":
    origins = ["*"]
else:
    origins = [origin.strip() for origin in allowed_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LatencyTrackingMiddleware)
app.add_middleware(RateLimitMiddleware)


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
    _LOG.error("Retrieval error: %s", exc, exc_info=True)
    return JSONResponse(status_code=503, content={"detail": "Retrieval service unavailable"})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    _LOG.warning("Validation error: %s", exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats(request: Request) -> dict[str, bool]:
    client: QdrantClient | None = getattr(request.app.state, "qdrant_client", None)
    if client is None:
        raise RuntimeError("Qdrant client not initialized")
    collections = client.get_collections()
    return {col.name: True for col in collections.collections}


@app.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),  # noqa: B008
) -> QueryResponse:
    result = pipeline.query(parse_query(body.query), sources=body.sources)
    return QueryResponse(
        answer=result.answer,
        sources_used=list(result.sources_used),
        num_chunks_used=result.num_chunks_used,
        model_name=result.model_name,
        query=result.query,
        confidence_score=result.confidence_score,
    )
