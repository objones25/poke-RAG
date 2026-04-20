"""Shared pytest fixtures for the poke-RAG test suite."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.types import EntityType, GenerationResult, RetrievalResult, RetrievedChunk, Source


def make_chunk(
    text: str = "some text",
    score: float = 0.9,
    source: Source = "pokeapi",
    entity_name: str | None = "Bulbasaur",
    entity_type: EntityType | None = "pokemon",
    chunk_index: int = 0,
    original_doc_id: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        source=source,
        entity_name=entity_name,
        entity_type=entity_type,
        chunk_index=chunk_index,
        original_doc_id=(original_doc_id if original_doc_id is not None else f"doc_{chunk_index}"),
    )


@pytest.fixture(name="make_chunk")
def make_chunk_fixture() -> Callable[..., RetrievedChunk]:
    return make_chunk


@pytest.fixture
def make_retrieval_result() -> Callable[..., RetrievalResult]:
    def _factory(
        chunks: tuple[RetrievedChunk, ...] = (),
        query: str = "test query",
    ) -> RetrievalResult:
        return RetrievalResult(documents=chunks, query=query)

    return _factory


@pytest.fixture
def make_generation_result() -> Callable[..., GenerationResult]:
    def _factory(
        answer: str = "Pikachu is Electric-type.",
        sources_used: tuple[Source, ...] = ("pokeapi",),
        model_name: str = "google/gemma-4-E4B-it",
        num_chunks_used: int = 1,
    ) -> GenerationResult:
        return GenerationResult(
            answer=answer,
            sources_used=sources_used,
            model_name=model_name,
            num_chunks_used=num_chunks_used,
        )

    return _factory
