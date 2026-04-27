from __future__ import annotations

import logging

from fastapi import Request
from qdrant_client import AsyncQdrantClient, QdrantClient

from src.config import Settings
from src.generation.generator import Generator
from src.generation.inference import Inferencer
from src.generation.loader import ModelLoader
from src.generation.models import GenerationConfig
from src.generation.prompt_builder import build_prompt
from src.pipeline.rag_pipeline import AsyncRAGPipeline, RAGPipeline
from src.retrieval.cache import LocalLRUCache, RedisCache
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.knowledge_refiner import KnowledgeRefiner
from src.retrieval.protocols import CacheProtocol
from src.retrieval.query_router import QueryRouter
from src.retrieval.query_transformer import HyDETransformer, MultiDraftHyDETransformer
from src.retrieval.reranker import BGEReranker
from src.retrieval.retriever import AsyncRetriever, Retriever
from src.retrieval.vector_store import AsyncQdrantVectorStore, QdrantVectorStore

_LOG = logging.getLogger(__name__)


def _build_cache(settings: Settings) -> CacheProtocol | None:
    """Instantiate the configured cache backend, or return None if caching is disabled."""
    if not settings.cache_enabled:
        _LOG.info("Cache disabled")
        return None

    if settings.redis_url:
        try:
            password = (
                settings.redis_password.get_secret_value()
                if settings.redis_password is not None
                else None
            )
            cache: CacheProtocol = RedisCache(
                redis_url=settings.redis_url,
                username=settings.redis_username,
                password=password,
            )
            _LOG.info("Cache enabled: Redis at %s", settings.redis_url)
            return cache
        except Exception:
            _LOG.warning(
                "Redis cache init failed; falling back to LocalLRUCache", exc_info=True
            )

    cache = LocalLRUCache(maxsize=settings.cache_max_size)
    _LOG.info("Cache enabled: LocalLRUCache (maxsize=%d)", settings.cache_max_size)
    return cache


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return pipeline


def get_async_pipeline(request: Request) -> AsyncRAGPipeline:
    pipeline: AsyncRAGPipeline | None = getattr(request.app.state, "async_pipeline", None)
    if pipeline is None:
        raise RuntimeError("Async pipeline not initialized")
    return pipeline


def build_pipeline() -> tuple[RAGPipeline, ModelLoader, QdrantClient]:
    settings = Settings.from_env()

    embedder = BGEEmbedder.from_pretrained(
        model_name=settings.embed_model,
        device=settings.device,
        colbert_enabled=settings.colbert_enabled,
    )
    reranker = BGEReranker.from_pretrained(model_name=settings.rerank_model, device=settings.device)

    api_key_str = (
        None if settings.qdrant_api_key is None else settings.qdrant_api_key.get_secret_value()
    )
    client = QdrantClient(url=settings.qdrant_url, api_key=api_key_str)
    vector_store = QdrantVectorStore(client, colbert_enabled=settings.colbert_enabled)
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

    query_transformer: MultiDraftHyDETransformer | HyDETransformer | None
    if settings.hyde_enabled:
        if settings.hyde_num_drafts > 1:
            query_transformer = MultiDraftHyDETransformer(
                inferencer,
                embedder,
                num_drafts=settings.hyde_num_drafts,
                max_new_tokens=settings.hyde_max_tokens,
            )
            _LOG.info(
                "Multi-draft HyDE enabled: %d drafts, max_new_tokens=%d",
                settings.hyde_num_drafts,
                settings.hyde_max_tokens,
            )
        else:
            query_transformer = HyDETransformer(inferencer, max_new_tokens=settings.hyde_max_tokens)
            _LOG.info(
                "HyDE enabled: query transformer active (max_new_tokens=%d)",
                settings.hyde_max_tokens,
            )
        if settings.hyde_confidence_threshold is not None:
            _LOG.info(
                "HyDE confidence gating enabled: threshold=%.3f", settings.hyde_confidence_threshold
            )
    else:
        query_transformer = None
        _LOG.info("HyDE disabled: queries embedded directly without transformation")
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        query_transformer=query_transformer,
        hyde_confidence_threshold=settings.hyde_confidence_threshold,
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

    if settings.refiner_enabled:
        knowledge_refiner: KnowledgeRefiner | None = KnowledgeRefiner(
            reranker,
            upper_threshold=settings.refiner_upper_threshold,
            lower_threshold=settings.refiner_lower_threshold,
            strip_threshold=settings.refiner_strip_threshold,
        )
        _LOG.info(
            "KnowledgeRefiner enabled: upper=%.1f lower=%.1f strip=%.1f",
            settings.refiner_upper_threshold,
            settings.refiner_lower_threshold,
            settings.refiner_strip_threshold,
        )
    else:
        knowledge_refiner = None
        _LOG.info("KnowledgeRefiner disabled")

    cache = _build_cache(settings)

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        query_router=query_router,
        knowledge_refiner=knowledge_refiner,
        cache=cache,
    )
    return pipeline, loader, client


def build_async_pipeline() -> tuple[AsyncRAGPipeline, ModelLoader, AsyncQdrantClient]:
    settings = Settings.from_env()

    embedder = BGEEmbedder.from_pretrained(
        model_name=settings.embed_model,
        device=settings.device,
        colbert_enabled=settings.colbert_enabled,
    )
    reranker = BGEReranker.from_pretrained(model_name=settings.rerank_model, device=settings.device)

    api_key_str = (
        None if settings.qdrant_api_key is None else settings.qdrant_api_key.get_secret_value()
    )
    async_client = AsyncQdrantClient(url=settings.qdrant_url, api_key=api_key_str)
    async_vector_store = AsyncQdrantVectorStore(
        async_client, colbert_enabled=settings.colbert_enabled
    )

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

    query_transformer: MultiDraftHyDETransformer | HyDETransformer | None
    if settings.hyde_enabled:
        if settings.hyde_num_drafts > 1:
            query_transformer = MultiDraftHyDETransformer(
                inferencer,
                embedder,
                num_drafts=settings.hyde_num_drafts,
                max_new_tokens=settings.hyde_max_tokens,
            )
            _LOG.info(
                "Multi-draft HyDE enabled: %d drafts, max_new_tokens=%d",
                settings.hyde_num_drafts,
                settings.hyde_max_tokens,
            )
        else:
            query_transformer = HyDETransformer(inferencer, max_new_tokens=settings.hyde_max_tokens)
            _LOG.info(
                "HyDE enabled: query transformer active (max_new_tokens=%d)",
                settings.hyde_max_tokens,
            )
        if settings.hyde_confidence_threshold is not None:
            _LOG.info(
                "HyDE confidence gating enabled: threshold=%.3f", settings.hyde_confidence_threshold
            )
    else:
        query_transformer = None
        _LOG.info("HyDE disabled: queries embedded directly without transformation")

    async_retriever = AsyncRetriever(
        embedder=embedder,
        vector_store=async_vector_store,
        reranker=reranker,
        query_transformer=query_transformer,
        hyde_confidence_threshold=settings.hyde_confidence_threshold,
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

    if settings.refiner_enabled:
        knowledge_refiner: KnowledgeRefiner | None = KnowledgeRefiner(
            reranker,
            upper_threshold=settings.refiner_upper_threshold,
            lower_threshold=settings.refiner_lower_threshold,
            strip_threshold=settings.refiner_strip_threshold,
        )
        _LOG.info(
            "KnowledgeRefiner enabled: upper=%.1f lower=%.1f strip=%.1f",
            settings.refiner_upper_threshold,
            settings.refiner_lower_threshold,
            settings.refiner_strip_threshold,
        )
    else:
        knowledge_refiner = None
        _LOG.info("KnowledgeRefiner disabled")

    cache = _build_cache(settings)

    async_pipeline = AsyncRAGPipeline(
        retriever=async_retriever,
        generator=generator,
        query_router=query_router,
        knowledge_refiner=knowledge_refiner,
        cache=cache,
    )
    return async_pipeline, loader, async_client
