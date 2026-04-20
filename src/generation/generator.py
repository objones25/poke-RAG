from __future__ import annotations

from src.generation.inference import Inferencer
from src.generation.loader import ModelLoader
from src.generation.models import GenerationConfig
from src.generation.protocols import GeneratorProtocol, PromptBuilderProtocol
from src.types import GenerationResult, RetrievedChunk

__all__ = ["Generator", "GeneratorProtocol"]


class Generator:
    def __init__(
        self,
        loader: ModelLoader,
        prompt_builder: PromptBuilderProtocol,
        inferencer: Inferencer,
        config: GenerationConfig | None = None,
    ) -> None:
        self._loader = loader
        self._prompt_builder = prompt_builder
        self._inferencer = inferencer
        self._config = config or GenerationConfig()

    def generate(self, query: str, chunks: tuple[RetrievedChunk, ...]) -> GenerationResult:
        if not chunks:
            raise ValueError("chunks must not be empty — retrieval must succeed before generation")

        prompt = self._prompt_builder(query, chunks)
        answer = self._inferencer.infer(prompt)
        unique_sources = tuple(sorted({c.source for c in chunks}))

        return GenerationResult(
            answer=answer,
            sources_used=unique_sources,
            model_name=self._config.model_id,
            num_chunks_used=len(chunks),
        )
