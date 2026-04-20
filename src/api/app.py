from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from src.api.dependencies import build_pipeline, get_pipeline
from src.api.models import QueryRequest, QueryResponse
from src.pipeline.rag_pipeline import RAGPipeline
from src.types import RetrievalError
from src.utils.logging import setup_logging

load_dotenv()  # populate os.environ before lifespan runs


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging()
    app.state.pipeline = build_pipeline()
    yield


app = FastAPI(title="poke-RAG", lifespan=lifespan)


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request, exc: RetrievalError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    result = pipeline.query(body.query, sources=body.sources)
    return QueryResponse(
        answer=result.answer,
        sources_used=list(result.sources_used),
        num_chunks_used=result.num_chunks_used,
        model_name=result.model_name,
        query=result.query,
    )
