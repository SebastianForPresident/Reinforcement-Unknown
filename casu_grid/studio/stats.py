"""Small, dependency-free statistics for comparing success rates (Wilson / Newcombe intervals, two-proportion z test)."""
from __future__ import annotations

import math


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def difference(x1: int, n1: int, x2: int, n2: int, z: float = 1.96) -> dict:
    """Success-rate difference p1 - p2 with a Newcombe (hybrid score) 95 % interval and a two-sided pooled two-proportion z test."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1, z)
    l2, u2 = wilson(x2, n2, z)
    d = p1 - p2
    low = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        p_value = 1.0
    else:
        zz = abs(d) / se
        p_value = math.erfc(zz / math.sqrt(2))
    return {"difference": d, "low": low, "high": high, "p_value": p_value}
