from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Entity names: alphanumeric with hyphens, underscores, apostrophes, spaces
_ENTITY_NAME_RE = re.compile(r"^[a-zA-Z0-9 '_\-]+$")


class QueryRequest(BaseModel):
    """Query payload for the RAG endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question about Pokémon",
    )
    sources: list[Literal["bulbapedia", "pokeapi", "smogon"]] | None = Field(
        default=None,
        description="Restrict retrieval to specific sources; omit to search all",
    )
    entity_name: str | None = Field(
        default=None,
        max_length=50,
        description="Optional Pokémon or entity name to filter results (e.g. 'Pikachu')",
    )

    @field_validator("entity_name")
    @classmethod
    def validate_entity_name(cls, v: str | None) -> str | None:
        if v is not None and not _ENTITY_NAME_RE.match(v):
            raise ValueError(
                "entity_name may only contain letters, digits, spaces, "
                "hyphens, underscores, and apostrophes"
            )
        return v


class QueryResponse(BaseModel):
    """Response from the RAG endpoint."""

    answer: str = Field(..., description="Generated answer grounded in retrieved context")
    sources_used: list[Literal["bulbapedia", "pokeapi", "smogon"]] = Field(
        ..., description="Sources that contributed chunks to the answer"
    )
    num_chunks_used: int = Field(
        ..., description="Number of context chunks passed to the generator"
    )
    model_name: str = Field(..., description="Name of the generation model used")
    query: str = Field(..., description="The parsed query that was processed")
    confidence_score: float = Field(
        ...,
        description="Sigmoid of the top-ranked chunk's reranker score",
    )
