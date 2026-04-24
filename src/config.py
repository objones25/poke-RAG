from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from pydantic import SecretStr

_LOG = logging.getLogger(__name__)


def _parse_bool(value: str | None, env_var_name: str, default: bool) -> bool:
    """Parse a boolean environment variable.

    Accepts: "true", "1", "yes" (case-insensitive) as True.
    Accepts: "false", "0", "no" (case-insensitive) as False.
    Raises ValueError for any other value.
    """
    if value is None:
        return default
    lower_val = value.lower().strip()
    if lower_val in ("true", "1", "yes"):
        return True
    if lower_val in ("false", "0", "no"):
        return False
    raise ValueError(
        f"{env_var_name} must be one of ['true', '1', 'yes', 'false', '0', 'no'], got: {value!r}"
    )


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
    query_timeout_seconds: float = 120.0
    lora_adapter_path: str | None = None
    hyde_enabled: bool = False
    hyde_max_tokens: int = 150
    hyde_num_drafts: int = 1
    hyde_confidence_threshold: float | None = None
    routing_enabled: bool = False

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
        if tokenizer_max_length <= 0:
            raise ValueError(
                f"TOKENIZER_MAX_LENGTH must be a positive integer, got: {tokenizer_max_length}"
            )

        try:
            hyde_max_tokens = int(os.getenv("HYDE_MAX_TOKENS", "150"))
        except ValueError:
            raise ValueError(
                f"HYDE_MAX_TOKENS must be a valid int, got: {os.getenv('HYDE_MAX_TOKENS')!r}"
            ) from None
        if hyde_max_tokens <= 0:
            raise ValueError(f"HYDE_MAX_TOKENS must be a positive integer, got: {hyde_max_tokens}")

        try:
            hyde_num_drafts = int(os.getenv("HYDE_NUM_DRAFTS", "1"))
        except ValueError:
            raise ValueError(
                f"HYDE_NUM_DRAFTS must be a valid int, got: {os.getenv('HYDE_NUM_DRAFTS')!r}"
            ) from None
        if hyde_num_drafts <= 0:
            raise ValueError(f"HYDE_NUM_DRAFTS must be a positive integer, got: {hyde_num_drafts}")

        try:
            query_timeout_seconds = float(os.getenv("QUERY_TIMEOUT_SECONDS", "120.0"))
        except ValueError:
            raw_timeout = os.getenv("QUERY_TIMEOUT_SECONDS")
            raise ValueError(
                f"QUERY_TIMEOUT_SECONDS must be a valid float, got: {raw_timeout!r}"
            ) from None
        if not 1.0 <= query_timeout_seconds <= 600.0:
            raise ValueError(
                f"QUERY_TIMEOUT_SECONDS must be between 1.0 and 600.0, got: {query_timeout_seconds}"
            )

        raw_threshold = os.getenv("HYDE_CONFIDENCE_THRESHOLD")
        if raw_threshold is not None:
            try:
                threshold_value = float(raw_threshold)
            except ValueError:
                raise ValueError(
                    f"HYDE_CONFIDENCE_THRESHOLD must be a valid float, got: {raw_threshold!r}"
                ) from None
            if not 0.0 <= threshold_value <= 1.0:
                raise ValueError(
                    f"HYDE_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got: {threshold_value}"
                )
            hyde_confidence_threshold: float | None = threshold_value
        else:
            hyde_confidence_threshold = None

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
            do_sample=_parse_bool(os.getenv("DO_SAMPLE"), "DO_SAMPLE", True),
            tokenizer_max_length=tokenizer_max_length,
            return_tensors=os.getenv("RETURN_TENSORS", "pt"),
            truncation=_parse_bool(os.getenv("TRUNCATION"), "TRUNCATION", True),
            device=device,
            query_timeout_seconds=query_timeout_seconds,
            lora_adapter_path=os.getenv("LORA_ADAPTER_PATH"),
            hyde_enabled=_parse_bool(os.getenv("HYDE_ENABLED"), "HYDE_ENABLED", False),
            hyde_max_tokens=hyde_max_tokens,
            hyde_num_drafts=hyde_num_drafts,
            hyde_confidence_threshold=hyde_confidence_threshold,
            routing_enabled=_parse_bool(os.getenv("ROUTING_ENABLED"), "ROUTING_ENABLED", False),
        )
