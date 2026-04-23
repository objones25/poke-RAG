from __future__ import annotations

import logging

from fastapi import Request
from qdrant_client import QdrantClient

from src.config import Settings
from src.generation.generator import Generator
from src.generation.inference import Inferencer
from src.generation.loader import ModelLoader
from src.generation.models import GenerationConfig
from src.generation.prompt_builder import build_prompt
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.query_router import QueryRouter
from src.retrieval.query_transformer import HyDETransformer
from src.retrieval.reranker import BGEReranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import QdrantVectorStore

_LOG = logging.getLogger(__name__)


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return pipeline


def build_pipeline() -> tuple[RAGPipeline, ModelLoader, QdrantClient]:
    settings = Settings.from_env()

    embedder = BGEEmbedder.from_pretrained(model_name=settings.embed_model, device=settings.device)
    reranker = BGEReranker.from_pretrained(model_name=settings.rerank_model, device=settings.device)

    api_key_str = (
        None if settings.qdrant_api_key is None else settings.qdrant_api_key.get_secret_value()
    )
    client = QdrantClient(url=settings.qdrant_url, api_key=api_key_str)
    vector_store = QdrantVectorStore(client)
    vector_store.ensure_collections()

    gen_config = GenerationConfig(
        model_id=settings.gen_model,
        temperature=settings.temperature,
        max_new_tokens=settings.max_new_tokens,
        top_p=settings.top_p,
        do_sample=settings.do_sample,
    )
    loader = ModelLoader(
        config=gen_config,
        device=settings.device,
        lora_adapter_path=settings.lora_adapter_path,
    )
    loader.load()
    inferencer = Inferencer(
        model=loader.get_model(),
        processor=loader.get_tokenizer(),
        config=gen_config,
    )

    if settings.hyde_enabled:
        query_transformer = HyDETransformer(inferencer, max_new_tokens=settings.hyde_max_tokens)
        _LOG.info(
            "HyDE enabled: query transformer active (max_new_tokens=%d)", settings.hyde_max_tokens
        )
    else:
        query_transformer = None
        _LOG.info("HyDE disabled: queries embedded directly without transformation")
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        query_transformer=query_transformer,
    )

    generator = Generator(
        loader=loader, prompt_builder=build_prompt, inferencer=inferencer, config=gen_config
    )

    if settings.routing_enabled:
        query_router = QueryRouter()
        _LOG.info("Query routing enabled: keyword-based source routing active")
    else:
        query_router = None
        _LOG.info("Query routing disabled: all sources searched for every query")

    pipeline = RAGPipeline(retriever=retriever, generator=generator, query_router=query_router)
    return pipeline, loader, client
