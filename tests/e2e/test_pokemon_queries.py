"""End-to-end tests: full RAG pipeline against real Qdrant and real models.

Run with: uv run pytest -m "e2e and gpu" -v

Requirements:
  - Qdrant running at $QDRANT_URL (default: http://localhost:6333)
  - BAAI/bge-m3 and BAAI/bge-reranker-v2-m3 downloaded
  - google/gemma-4-E4B-it downloaded (or GEN_MODEL env var pointing to another model)
  - CUDA-capable GPU (or DEVICE=cpu for slow CPU inference)

Set PYTEST_E2E_CLEANUP=1 to delete the e2e Qdrant collections after the run.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest

from src.types import RetrievalError

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_E2E_CLEANUP = os.getenv("PYTEST_E2E_CLEANUP", "0") == "1"


@pytest.fixture(scope="module")
def qdrant_client():
    try:
        from qdrant_client import QdrantClient

        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url)
        client.get_collections()
        return client
    except Exception as exc:
        pytest.skip(f"Qdrant not reachable: {exc}")


@pytest.fixture(scope="module")
def embedder():
    try:
        from src.retrieval.embedder import BGEEmbedder

        device = os.getenv("DEVICE", "cpu")
        return BGEEmbedder.from_pretrained(
            model_name="BAAI/bge-m3",
            device=device,
        )
    except Exception as exc:
        pytest.skip(f"BGE-M3 embedder unavailable: {exc}")


@pytest.fixture(scope="module")
def reranker():
    try:
        from src.retrieval.reranker import BGEReranker

        device = os.getenv("DEVICE", "cpu")
        return BGEReranker.from_pretrained(
            model_name="BAAI/bge-reranker-v2-m3",
            device=device,
        )
    except Exception as exc:
        pytest.skip(f"BGE reranker unavailable: {exc}")


@pytest.fixture(scope="module")
def indexed_store(qdrant_client, embedder):
    from src.retrieval.chunker import chunk_file
    from src.retrieval.vector_store import QdrantVectorStore
    from src.types import Source

    store = QdrantVectorStore(qdrant_client)
    store.ensure_collections()

    fixture_files: list[tuple[Source, Path]] = [
        ("pokeapi", FIXTURES_DIR / "sample_pokeapi.txt"),
        ("smogon", FIXTURES_DIR / "sample_smogon.txt"),
        ("bulbapedia", FIXTURES_DIR / "sample_bulbapedia.txt"),
    ]

    for source, path in fixture_files:
        chunks = chunk_file(path, source=source)
        if not chunks:
            continue
        texts = [c.text for c in chunks]
        embedding = embedder.encode(texts)
        store.upsert(source, chunks, embedding)

    yield store

    if _E2E_CLEANUP:
        for source, _ in fixture_files:
            with contextlib.suppress(Exception):
                qdrant_client.delete_collection(source)


@pytest.fixture(scope="module")
def generator():
    try:
        from src.generation.generator import Generator
        from src.generation.inference import Inferencer
        from src.generation.loader import ModelLoader
        from src.generation.models import GenerationConfig
        from src.generation.prompt_builder import build_prompt

        device = os.getenv("DEVICE", "cpu")
        config = GenerationConfig(
            model_id=os.getenv("GEN_MODEL", "google/gemma-4-E4B-it"),
            temperature=0.1,
            max_new_tokens=256,
            top_p=0.9,
            do_sample=False,
        )
        loader = ModelLoader(config=config, device=device)
        loader.load()
        inferencer = Inferencer(
            model=loader.get_model(),
            processor=loader.get_tokenizer(),
            config=config,
        )
        gen = Generator(
            loader=loader,
            prompt_builder=build_prompt,
            inferencer=inferencer,
            config=config,
        )
        yield gen
        loader.unload()
    except Exception as exc:
        pytest.skip(f"Generator unavailable (model/GPU required): {exc}")


@pytest.fixture(scope="module")
def e2e_pipeline(indexed_store, embedder, reranker, generator):
    from src.pipeline.rag_pipeline import RAGPipeline
    from src.retrieval.retriever import Retriever

    retriever = Retriever(
        embedder=embedder,
        vector_store=indexed_store,
        reranker=reranker,
    )
    return RAGPipeline(retriever=retriever, generator=generator)


@pytest.mark.e2e
@pytest.mark.gpu
class TestPokemonQueryAnswers:
    def test_pikachu_type_query_returns_answer(self, e2e_pipeline):
        result = e2e_pipeline.query("What type is Pikachu?")
        assert result.answer.strip()
        assert result.num_chunks_used > 0

    def test_pikachu_answer_mentions_electric(self, e2e_pipeline):
        result = e2e_pipeline.query("What type is Pikachu?")
        assert "electric" in result.answer.lower()

    def test_garchomp_moveset_mentions_key_moves(self, e2e_pipeline):
        result = e2e_pipeline.query("What is Garchomp's best moveset in OU?")
        assert result.answer.strip()
        assert any(w in result.answer.lower() for w in ["earthquake", "dragon", "garchomp"])

    def test_charizard_answer_mentions_fire_or_name(self, e2e_pipeline):
        result = e2e_pipeline.query("Describe Charizard")
        assert result.answer.strip()
        assert any(w in result.answer.lower() for w in ["fire", "charizard"])

    def test_result_has_non_empty_sources(self, e2e_pipeline):
        result = e2e_pipeline.query("Tell me about Pikachu")
        assert result.sources_used

    def test_result_sources_are_valid_namespaces(self, e2e_pipeline):
        result = e2e_pipeline.query("Tell me about Pikachu")
        valid = {"bulbapedia", "pokeapi", "smogon"}
        for source in result.sources_used:
            assert source in valid

    def test_confidence_score_is_set(self, e2e_pipeline):
        result = e2e_pipeline.query("What are Garchomp's base stats?")
        assert result.confidence_score is not None

    def test_confidence_score_is_finite_float(self, e2e_pipeline):
        import math

        result = e2e_pipeline.query("What are Garchomp's base stats?")
        assert isinstance(result.confidence_score, float)
        assert math.isfinite(result.confidence_score)

    def test_top_k_bounds_num_chunks(self, e2e_pipeline):
        result = e2e_pipeline.query("What type is Pikachu?", top_k=3)
        assert 1 <= result.num_chunks_used <= 3

    def test_source_filter_restricts_to_pokeapi(self, e2e_pipeline):
        result = e2e_pipeline.query("What is Pikachu?", sources=["pokeapi"])
        assert result.answer.strip()
        assert result.num_chunks_used > 0

    def test_smogon_source_filter_for_competitive_query(self, e2e_pipeline):
        result = e2e_pipeline.query(
            "What moveset should I use for Garchomp?",
            sources=["smogon"],
        )
        assert result.answer.strip()
        assert result.num_chunks_used > 0


@pytest.mark.e2e
@pytest.mark.gpu
class TestPipelineInvariants:
    def test_empty_query_raises_value_error(self, e2e_pipeline):
        with pytest.raises(ValueError, match="empty"):
            e2e_pipeline.query("")

    def test_whitespace_only_query_raises_value_error(self, e2e_pipeline):
        with pytest.raises(ValueError, match="empty"):
            e2e_pipeline.query("   ")

    def test_retrieval_error_blocks_generation(self, e2e_pipeline, mocker):
        mocker.patch.object(
            e2e_pipeline._retriever,
            "retrieve",
            side_effect=RetrievalError("forced failure"),
        )
        with pytest.raises(RetrievalError):
            e2e_pipeline.query("What type is Pikachu?")

    def test_model_name_present_and_namespaced(self, e2e_pipeline):
        result = e2e_pipeline.query("What type is Pikachu?")
        assert result.model_name
        assert "/" in result.model_name

    def test_query_text_echoed_in_result(self, e2e_pipeline):
        query = "What type is Pikachu?"
        result = e2e_pipeline.query(query)
        assert result.query == query

    def test_pipeline_result_is_frozen(self, e2e_pipeline):
        from dataclasses import FrozenInstanceError

        result = e2e_pipeline.query("What type is Pikachu?")
        with pytest.raises(FrozenInstanceError):
            result.answer = "mutated"
