from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    """Configure root logger with a human-readable format.

    Safe to call multiple times: re-entrant calls are no-ops if the root logger
    already has handlers (prevents duplicate log lines in test environments).

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). If not given,
            reads LOG_LEVEL env var, defaulting to INFO.

    Raises:
        ValueError: If level is not a valid logging level.
    """
    effective = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if effective not in valid_levels:
        raise ValueError(
            f"LOG_LEVEL must be one of {valid_levels}, got: {os.getenv('LOG_LEVEL')!r}"
        )

    root = logging.getLogger()
    root.setLevel(effective)

    # Re-entrant safety: if root logger already has handlers, don't add duplicates
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(effective)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
