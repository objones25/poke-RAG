from __future__ import annotations

import os

import torch
from fastapi import Request
from qdrant_client import QdrantClient

from src.generation.generator import Generator
from src.generation.inference import Inferencer
from src.generation.loader import ModelLoader
from src.generation.models import GenerationConfig
from src.generation.prompt_builder import build_prompt
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.reranker import BGEReranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import QdrantVectorStore

_DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
_DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
_DEFAULT_GEN_MODEL = "google/gemma-4-E4B-it"


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return pipeline


def build_pipeline() -> RAGPipeline:
    qdrant_url = os.environ["QDRANT_URL"]
    embed_model = os.getenv("EMBED_MODEL", _DEFAULT_EMBED_MODEL)
    rerank_model = os.getenv("RERANK_MODEL", _DEFAULT_RERANK_MODEL)
    gen_model = os.getenv("GEN_MODEL", _DEFAULT_GEN_MODEL)

    use_fp16 = torch.cuda.is_available()

    embedder = BGEEmbedder.from_pretrained(model_name=embed_model, use_fp16=use_fp16)
    reranker = BGEReranker.from_pretrained(model_name=rerank_model, use_fp16=use_fp16)

    client = QdrantClient(url=qdrant_url)
    vector_store = QdrantVectorStore(client)

    retriever = Retriever(embedder=embedder, vector_store=vector_store, reranker=reranker)

    gen_config = GenerationConfig(model_id=gen_model)
    loader = ModelLoader(config=gen_config)
    loader.load()
    inferencer = Inferencer(
        model=loader.get_model(),
        tokenizer=loader.get_tokenizer(),
        config=gen_config,
    )
    generator = Generator(loader=loader, prompt_builder=build_prompt, inferencer=inferencer)

    return RAGPipeline(retriever=retriever, generator=generator)
