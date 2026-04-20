from __future__ import annotations

import os
from dataclasses import dataclass


def _detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    qdrant_api_key: str | None
    embed_model: str
    rerank_model: str
    gen_model: str
    temperature: float
    max_new_tokens: int
    top_p: float
    do_sample: bool
    tokenizer_max_length: int
    return_tensors: str
    truncation: bool
    device: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            qdrant_url=os.environ["QDRANT_URL"],
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
            rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            gen_model=os.getenv("GEN_MODEL", "google/gemma-4-E4B-it"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
            top_p=float(os.getenv("TOP_P", "0.9")),
            do_sample=os.getenv("DO_SAMPLE", "true").lower() == "true",
            tokenizer_max_length=int(os.getenv("TOKENIZER_MAX_LENGTH", "8192")),
            return_tensors=os.getenv("RETURN_TENSORS", "pt"),
            truncation=os.getenv("TRUNCATION", "true").lower() == "true",
            device=os.getenv("DEVICE", _detect_device()),
        )
