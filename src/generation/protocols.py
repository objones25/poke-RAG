from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from src.types import GenerationResult, RetrievedChunk


@runtime_checkable
class PromptBuilderProtocol(Protocol):
    def __call__(self, query: str, chunks: tuple[RetrievedChunk, ...]) -> str: ...


@runtime_checkable
class GeneratorProtocol(Protocol):
    def generate(self, query: str, chunks: tuple[RetrievedChunk, ...]) -> GenerationResult: ...


@runtime_checkable
class StreamingGeneratorProtocol(Protocol):
    def stream_generate(self, query: str, chunks: tuple[RetrievedChunk, ...]) -> Iterator[str]: ...
