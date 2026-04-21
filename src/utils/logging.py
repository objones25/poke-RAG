from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger. Reads LOG_LEVEL env var when level is not given."""
    effective = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if effective not in valid_levels:
        raise ValueError(
            f"LOG_LEVEL must be one of {valid_levels}, got: {os.getenv('LOG_LEVEL')!r}"
        )
    logging.basicConfig(
        level=effective,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
