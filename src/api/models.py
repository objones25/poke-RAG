from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    sources: list[Literal["bulbapedia", "pokeapi", "smogon"]] | None = None
    entity_name: str | None = Field(default=None, max_length=200)


class QueryResponse(BaseModel):
    answer: str
    sources_used: list[Literal["bulbapedia", "pokeapi", "smogon"]]
    num_chunks_used: int
    model_name: str
    query: str
    confidence_score: float | None = None
