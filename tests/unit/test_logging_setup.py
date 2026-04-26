"""Unit tests for src/utils/logging.py."""

from __future__ import annotations

import logging
import os

import pytest


@pytest.fixture(autouse=True)
def _restore_root_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Save and restore root logger state to prevent test pollution.

    Resets the root logger to NOTSET before each test to ensure setup_logging()
    is called with a clean slate, not polluted by previous tests.
    Also temporarily unsets LOG_LEVEL to ensure default behavior is tested.
    """
    root = logging.getLogger()
    original_level = root.level
    original_handlers = root.handlers[:]
    original_httpx_level = logging.getLogger("httpx").level
    original_log_level = os.getenv("LOG_LEVEL")

    # Reset to clean state before the test runs
    # Unset LOG_LEVEL so setup_logging() uses the default (INFO)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    yield
    # Restore after the test
    root.handlers.clear()
    root.handlers.extend(original_handlers)
    root.setLevel(original_level)
    logging.getLogger("httpx").setLevel(original_httpx_level)
    if original_log_level is not None:
        monkeypatch.setenv("LOG_LEVEL", original_log_level)


@pytest.mark.unit
class TestSetupLogging:
    def test_raises_value_error_for_invalid_level(self) -> None:
        from src.utils.logging import setup_logging

        with pytest.raises(ValueError, match="LOG_LEVEL"):
            setup_logging(level="INVALID")

    def test_default_level_is_info(self) -> None:
        from src.utils.logging import setup_logging

        setup_logging()
        assert logging.getLogger().level == logging.INFO

    def test_explicit_debug_level(self) -> None:
        from src.utils.logging import setup_logging

        setup_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_reentrancy_does_not_add_duplicate_handlers(self) -> None:
        from src.utils.logging import setup_logging

        setup_logging()
        count_after_first = len(logging.getLogger().handlers)
        setup_logging()
        assert len(logging.getLogger().handlers) == count_after_first

    def test_httpx_logger_suppressed(self) -> None:
        from src.utils.logging import setup_logging

        setup_logging()
        assert logging.getLogger("httpx").level >= logging.WARNING

    def test_qdrant_client_logger_suppressed(self) -> None:
        from src.utils.logging import setup_logging

        setup_logging()
        assert logging.getLogger("qdrant_client").level >= logging.WARNING
