"""Unit tests for src/utils/logging.py."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _restore_root_logger() -> None:
    """Save and restore root logger state to prevent test pollution."""
    root = logging.getLogger()
    original_level = root.level
    original_handlers = root.handlers[:]
    original_httpx_level = logging.getLogger("httpx").level
    root.handlers.clear()
    yield
    root.handlers.clear()
    root.handlers.extend(original_handlers)
    root.setLevel(original_level)
    logging.getLogger("httpx").setLevel(original_httpx_level)


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
