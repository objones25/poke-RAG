"""Integration tests for the full generation pipeline.

Uses a mocked model that returns deterministic output — no GPU required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.generation.generator import Generator
from src.generation.inference import Inferencer
from src.generation.loader import ModelLoader
from src.generation.models import GenerationConfig
from src.generation.prompt_builder import build_prompt
from src.types import GenerationResult, RetrievedChunk


@pytest.fixture()
def config() -> GenerationConfig:
    return GenerationConfig(model_id="google/gemma-4-E4B-it")


@pytest.fixture()
def sample_chunks() -> tuple[RetrievedChunk, ...]:
    return (
        RetrievedChunk(
            text="Charizard is a Fire/Flying-type Pokémon introduced in Generation I.",
            score=0.95,
            source="bulbapedia",
            pokemon_name="Charizard",
            chunk_index=0,
            original_doc_id="bulba_charizard_001",
        ),
        RetrievedChunk(
            text="Charizard @ Choice Specs — In OU, Charizard is a powerful special attacker.",
            score=0.82,
            source="smogon",
            pokemon_name="Charizard",
            chunk_index=0,
            original_doc_id="smogon_charizard_001",
        ),
        RetrievedChunk(
            text="charizard hp:78 attack:84 defense:78 sp_atk:109 sp_def:85 speed:100",
            score=0.71,
            source="pokeapi",
            pokemon_name="Charizard",
            chunk_index=0,
            original_doc_id="pokeapi_charizard_001",
        ),
    )


@pytest.fixture()
def generator(config: GenerationConfig) -> Generator:
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    encoded = MagicMock()
    encoded.to = MagicMock(return_value=encoded)
    encoded.__getitem__ = MagicMock(return_value=MagicMock())
    fake_tokenizer.return_value = encoded
    fake_model.device = "cpu"
    fake_model.generate.return_value = MagicMock()
    fake_tokenizer.decode.return_value = (
        "Charizard is a Fire/Flying-type Pokémon with 109 Special Attack."
    )

    loader = ModelLoader(config=config)
    with (
        patch(
            "src.generation.loader.AutoModelForImageTextToText.from_pretrained",
            return_value=fake_model,
        ),
        patch(
            "src.generation.loader.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ),
    ):
        loader.load()

    inferencer = Inferencer(loader.get_model(), loader.get_tokenizer(), config)
    return Generator(
        loader=loader, prompt_builder=build_prompt, inferencer=inferencer, config=config
    )


@pytest.mark.integration
class TestGenerationPipelineIntegration:
    def test_returns_generation_result(
        self, generator: Generator, sample_chunks: tuple[RetrievedChunk, ...]
    ) -> None:
        result = generator.generate("What type is Charizard?", sample_chunks)
        assert isinstance(result, GenerationResult)

    def test_answer_is_non_empty_string(
        self, generator: Generator, sample_chunks: tuple[RetrievedChunk, ...]
    ) -> None:
        result = generator.generate("What type is Charizard?", sample_chunks)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_sources_used_contains_all_three_sources(
        self, generator: Generator, sample_chunks: tuple[RetrievedChunk, ...]
    ) -> None:
        result = generator.generate("Tell me about Charizard.", sample_chunks)
        assert set(result.sources_used) == {"bulbapedia", "smogon", "pokeapi"}

    def test_num_chunks_used_matches_input(
        self, generator: Generator, sample_chunks: tuple[RetrievedChunk, ...]
    ) -> None:
        result = generator.generate("Tell me about Charizard.", sample_chunks)
        assert result.num_chunks_used == len(sample_chunks)

    def test_model_name_from_config(
        self, generator: Generator, sample_chunks: tuple[RetrievedChunk, ...]
    ) -> None:
        result = generator.generate("Tell me about Charizard.", sample_chunks)
        assert result.model_name == "google/gemma-4-E4B-it"

    def test_raises_on_empty_chunks(self, generator: Generator) -> None:
        with pytest.raises(ValueError, match="chunks"):
            generator.generate("What type is Charizard?", ())
