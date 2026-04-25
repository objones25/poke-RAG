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


def _parse_float_in_range(
    value: str | None,
    env_var_name: str,
    default: float,
    min_val: float,
    max_val: float,
) -> float:
    """Parse a float from environment variable within a specified range.

    Args:
        value: The string value to parse (or None to return default).
        env_var_name: The environment variable name (for error messages).
        default: Default value if value is None.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        Parsed float value within [min_val, max_val].

    Raises:
        ValueError: If value cannot be parsed as float or is outside the range.
    """
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{env_var_name} must be a valid float, got: {value!r}") from None
    if not min_val <= parsed <= max_val:
        raise ValueError(f"{env_var_name} must be between {min_val} and {max_val}, got: {parsed}")
    return parsed


def _parse_float_in_range_optional(
    value: str | None,
    env_var_name: str,
    min_val: float,
    max_val: float,
) -> float | None:
    """Parse an optional float from environment variable within a specified range.

    Args:
        value: The string value to parse (or None to return None).
        env_var_name: The environment variable name (for error messages).
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        Parsed float value within [min_val, max_val], or None if value is None.

    Raises:
        ValueError: If value cannot be parsed as float or is outside the range.
    """
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{env_var_name} must be a valid float, got: {value!r}") from None
    if not min_val <= parsed <= max_val:
        raise ValueError(f"{env_var_name} must be between {min_val} and {max_val}, got: {parsed}")
    return parsed


def _parse_int_positive(
    value: str | None,
    env_var_name: str,
    default: int,
) -> int:
    """Parse a positive integer from environment variable.

    Args:
        value: The string value to parse (or None to return default).
        env_var_name: The environment variable name (for error messages).
        default: Default value if value is None.

    Returns:
        Parsed positive integer (> 0).

    Raises:
        ValueError: If value cannot be parsed as int or is not positive.
    """
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{env_var_name} must be a valid int, got: {value!r}") from None
    if parsed <= 0:
        raise ValueError(f"{env_var_name} must be a positive integer, got: {parsed}")
    return parsed


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
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000
    redis_url: str | None = None
    redis_username: str = "default"
    redis_password: SecretStr | None = None
    async_pipeline_enabled: bool = False
    colbert_enabled: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        temperature = _parse_float_in_range(
            os.getenv("TEMPERATURE"),
            "TEMPERATURE",
            0.7,
            0.0,
            2.0,
        )

        max_new_tokens = _parse_int_positive(
            os.getenv("MAX_NEW_TOKENS"),
            "MAX_NEW_TOKENS",
            512,
        )

        top_p = _parse_float_in_range(
            os.getenv("TOP_P"),
            "TOP_P",
            0.9,
            0.0,
            1.0,
        )

        tokenizer_max_length = _parse_int_positive(
            os.getenv("TOKENIZER_MAX_LENGTH"),
            "TOKENIZER_MAX_LENGTH",
            8192,
        )

        hyde_max_tokens = _parse_int_positive(
            os.getenv("HYDE_MAX_TOKENS"),
            "HYDE_MAX_TOKENS",
            150,
        )

        hyde_num_drafts = _parse_int_positive(
            os.getenv("HYDE_NUM_DRAFTS"),
            "HYDE_NUM_DRAFTS",
            1,
        )

        query_timeout_seconds = _parse_float_in_range(
            os.getenv("QUERY_TIMEOUT_SECONDS"),
            "QUERY_TIMEOUT_SECONDS",
            120.0,
            1.0,
            600.0,
        )

        hyde_confidence_threshold = _parse_float_in_range_optional(
            os.getenv("HYDE_CONFIDENCE_THRESHOLD"),
            "HYDE_CONFIDENCE_THRESHOLD",
            0.0,
            1.0,
        )

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
            cache_enabled=_parse_bool(os.getenv("CACHE_ENABLED"), "CACHE_ENABLED", False),
            cache_ttl_seconds=_parse_int_positive(
                os.getenv("CACHE_TTL_SECONDS"), "CACHE_TTL_SECONDS", 3600
            ),
            cache_max_size=_parse_int_positive(os.getenv("CACHE_MAX_SIZE"), "CACHE_MAX_SIZE", 1000),
            redis_url=os.getenv("REDIS_URL"),
            redis_username=os.getenv("REDIS_USERNAME", "default"),
            redis_password=SecretStr(pw) if (pw := os.getenv("REDIS_PASSWORD")) else None,
            async_pipeline_enabled=_parse_bool(
                os.getenv("ASYNC_PIPELINE_ENABLED"), "ASYNC_PIPELINE_ENABLED", False
            ),
            colbert_enabled=_parse_bool(os.getenv("COLBERT_ENABLED"), "COLBERT_ENABLED", False),
        )
