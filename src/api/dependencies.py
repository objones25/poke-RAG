from __future__ import annotations

from fastapi import Request
from qdrant_client import QdrantClient

from src.config import Settings
from src.generation.generator import Generator
from src.generation.inference import Inferencer
from src.generation.loader import ModelLoader
from src.generation.models import GenerationConfig, TokenizerConfig
from src.generation.prompt_builder import build_prompt
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.reranker import BGEReranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import QdrantVectorStore


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return pipeline


def build_pipeline() -> tuple[RAGPipeline, ModelLoader, QdrantClient]:
    settings = Settings.from_env()

    embedder = BGEEmbedder.from_pretrained(model_name=settings.embed_model, device=settings.device)
    reranker = BGEReranker.from_pretrained(model_name=settings.rerank_model, device=settings.device)

    api_key_str = None if settings.qdrant_api_key is None else str(settings.qdrant_api_key)
    client = QdrantClient(url=settings.qdrant_url, api_key=api_key_str)
    vector_store = QdrantVectorStore(client)
    vector_store.ensure_collections()
    retriever = Retriever(embedder=embedder, vector_store=vector_store, reranker=reranker)

    gen_config = GenerationConfig(
        model_id=settings.gen_model,
        temperature=settings.temperature,
        max_new_tokens=settings.max_new_tokens,
        top_p=settings.top_p,
        do_sample=settings.do_sample,
    )
    tok_config = TokenizerConfig(
        max_length=settings.tokenizer_max_length,
        return_tensors=settings.return_tensors,
        truncation=settings.truncation,
    )
    loader = ModelLoader(config=gen_config, device=settings.device)
    loader.load()
    inferencer = Inferencer(
        model=loader.get_model(),
        tokenizer=loader.get_tokenizer(),
        config=gen_config,
        tokenizer_config=tok_config,
    )
    generator = Generator(
        loader=loader, prompt_builder=build_prompt, inferencer=inferencer, config=gen_config
    )

    return RAGPipeline(retriever=retriever, generator=generator), loader, client
