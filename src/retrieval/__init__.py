"""Retrieval subsystem — public API."""
from src.retrieval.chunker import (
    chunk_bulbapedia_doc,
    chunk_file,
    chunk_pokeapi_line,
    chunk_smogon_line,
)
from src.retrieval.context_assembler import ContextAssembler
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.protocols import (
    EmbedderProtocol,
    RerankerProtocol,
    RetrieverProtocol,
    VectorStoreProtocol,
)
from src.retrieval.reranker import BGEReranker
from src.retrieval.retriever import Retriever
from src.retrieval.types import EmbeddingOutput
from src.retrieval.vector_store import QdrantVectorStore

__all__ = [
    "BGEEmbedder",
    "BGEReranker",
    "ContextAssembler",
    "EmbeddingOutput",
    "EmbedderProtocol",
    "QdrantVectorStore",
    "RerankerProtocol",
    "RetrieverProtocol",
    "Retriever",
    "VectorStoreProtocol",
    "chunk_bulbapedia_doc",
    "chunk_file",
    "chunk_pokeapi_line",
    "chunk_smogon_line",
]
