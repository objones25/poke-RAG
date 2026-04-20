from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    sources: list[Literal["bulbapedia", "pokeapi", "smogon"]] | None = None


class QueryResponse(BaseModel):
    answer: str
    sources_used: list[str]
    num_chunks_used: int
    model_name: str
    query: str
