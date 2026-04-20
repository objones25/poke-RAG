from __future__ import annotations

import logging

from src.generation.inference import Inferencer
from src.generation.loader import ModelLoader
from src.generation.models import GenerationConfig
from src.generation.protocols import GeneratorProtocol, PromptBuilderProtocol
from src.types import GenerationResult, RetrievedChunk

__all__ = ["Generator", "GeneratorProtocol"]

_LOG = logging.getLogger(__name__)


class Generator:
    def __init__(
        self,
        loader: ModelLoader,
        prompt_builder: PromptBuilderProtocol,
        inferencer: Inferencer,
        config: GenerationConfig,
    ) -> None:
        self._loader = loader
        self._prompt_builder = prompt_builder
        self._inferencer = inferencer
        self._config = config

    def generate(self, query: str, chunks: tuple[RetrievedChunk, ...]) -> GenerationResult:
        if not chunks:
            raise ValueError("chunks must not be empty — retrieval must succeed before generation")

        _LOG.info(
            "Generating answer: query_len=%d chars, chunks=%d",
            len(query),
            len(chunks),
        )

        prompt = self._prompt_builder(query, chunks)
        answer = self._inferencer.infer(prompt)
        unique_sources = tuple(sorted({c.source for c in chunks}))

        _LOG.info(
            "Generation complete: answer_len=%d chars, sources=%s",
            len(answer),
            unique_sources,
        )

        return GenerationResult(
            answer=answer,
            sources_used=unique_sources,
            model_name=self._config.model_id,
            num_chunks_used=len(chunks),
        )
