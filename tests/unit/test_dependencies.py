"""Unit tests for src/api/dependencies.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI


@pytest.mark.unit
class TestGetPipeline:
    def test_returns_pipeline_when_initialized(self) -> None:
        from src.api.dependencies import get_pipeline

        app = FastAPI()
        mock_pipeline = MagicMock()
        app.state.pipeline = mock_pipeline

        request = MagicMock()
        request.app = app

        result = get_pipeline(request)
        assert result is mock_pipeline

    def test_raises_runtime_error_when_not_initialized(self) -> None:
        from src.api.dependencies import get_pipeline

        app = FastAPI()  # no pipeline set on state
        request = MagicMock()
        request.app = app

        with pytest.raises(RuntimeError, match="not initialized"):
            get_pipeline(request)
