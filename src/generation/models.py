from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerConfig:
    max_length: int = 2048
    return_tensors: str = "pt"
    truncation: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.9
    do_sample: bool = True
    model_id: str = "google/gemma-4-E4B-it"
