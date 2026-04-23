from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from pydantic import SecretStr

_LOG = logging.getLogger(__name__)


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
    qdrant_api_key: SecretStr | None
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
    lora_adapter_path: str | None = None
    hyde_enabled: bool = False
    hyde_max_tokens: int = 150

    @classmethod
    def from_env(cls) -> Settings:
        try:
            temperature = float(os.getenv("TEMPERATURE", "0.7"))
        except ValueError:
            raise ValueError(
                f"TEMPERATURE must be a valid float, got: {os.getenv('TEMPERATURE')!r}"
            ) from None
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"TEMPERATURE must be between 0.0 and 2.0, got: {temperature}")

        try:
            max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        except ValueError:
            raise ValueError(
                f"MAX_NEW_TOKENS must be a valid int, got: {os.getenv('MAX_NEW_TOKENS')!r}"
            ) from None
        if max_new_tokens <= 0:
            raise ValueError(f"MAX_NEW_TOKENS must be a positive integer, got: {max_new_tokens}")

        try:
            top_p = float(os.getenv("TOP_P", "0.9"))
        except ValueError:
            raise ValueError(f"TOP_P must be a valid float, got: {os.getenv('TOP_P')!r}") from None
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"TOP_P must be between 0.0 and 1.0, got: {top_p}")

        try:
            tokenizer_max_length = int(os.getenv("TOKENIZER_MAX_LENGTH", "8192"))
        except ValueError:
            tmax_val = os.getenv("TOKENIZER_MAX_LENGTH")
            msg = f"TOKENIZER_MAX_LENGTH must be a valid int, got: {tmax_val!r}"
            raise ValueError(msg) from None

        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if log_level not in valid_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of {valid_levels}, got: {os.getenv('LOG_LEVEL')!r}"
            )

        device = os.getenv("DEVICE", _detect_device())
        valid_devices = {"cpu", "cuda", "mps"}
        if device not in valid_devices:
            raise ValueError(f"DEVICE must be one of {valid_devices}, got: {device!r}")
        _LOG.info("Using device: %s", device)

        return cls(
            qdrant_url=os.environ["QDRANT_URL"],
            qdrant_api_key=SecretStr(api_key) if (api_key := os.getenv("QDRANT_API_KEY")) else None,
            embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
            rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            gen_model=os.getenv("GEN_MODEL", "google/gemma-4-E4B-it"),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            do_sample=os.getenv("DO_SAMPLE", "true").lower() == "true",
            tokenizer_max_length=tokenizer_max_length,
            return_tensors=os.getenv("RETURN_TENSORS", "pt"),
            truncation=os.getenv("TRUNCATION", "true").lower() == "true",
            device=device,
            lora_adapter_path=os.getenv("LORA_ADAPTER_PATH"),
            hyde_enabled=os.getenv("HYDE_ENABLED", "false").lower() == "true",
            hyde_max_tokens=int(os.getenv("HYDE_MAX_TOKENS", "150")),
        )
