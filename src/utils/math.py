from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid: avoids overflow for both large positive and negative x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)
