"""Unit tests for src/utils/math.py — shared math helpers."""

from __future__ import annotations

import math

import pytest

from src.utils.math import sigmoid


@pytest.mark.unit
class TestSigmoid:
    def test_zero_returns_half(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self) -> None:
        assert sigmoid(100.0) == pytest.approx(1.0, abs=1e-9)

    def test_large_negative_approaches_zero(self) -> None:
        assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-9)

    def test_positive_value(self) -> None:
        expected = 1.0 / (1.0 + math.exp(-2.0))
        assert sigmoid(2.0) == pytest.approx(expected)

    def test_negative_value(self) -> None:
        expected = math.exp(-3.0) / (1.0 + math.exp(-3.0))
        assert sigmoid(-3.0) == pytest.approx(expected)

    def test_numerically_stable_for_positive(self) -> None:
        # Numerically stable path: x >= 0 uses 1/(1+exp(-x))
        result = sigmoid(1.0)
        assert 0.5 < result < 1.0

    def test_numerically_stable_for_negative(self) -> None:
        # Numerically stable path: x < 0 uses exp(x)/(1+exp(x))
        result = sigmoid(-1.0)
        assert 0.0 < result < 0.5
