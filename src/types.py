from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Source = Literal["bulbapedia", "pokeapi", "smogon"]


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    source: Source
    pokemon_name: str | None
    chunk_index: int
    original_doc_id: str


@dataclass(frozen=True)
class RetrievalResult:
    documents: tuple[RetrievedChunk, ...]
    query: str


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    sources_used: tuple[str, ...]
    model_name: str
    num_chunks_used: int


class RetrievalError(Exception):
    """Raised when retrieval fails. Generator must never be called when this is raised."""


class EmbeddingError(RetrievalError):
    """Raised when the embedding model fails."""


class VectorIndexError(RetrievalError):
    """Raised when the vector index is unavailable or returns no results."""
