"""Small, dependency-free statistics helpers for comparing win rates."""

from __future__ import annotations

import math


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def two_proportion_z_test(
    successes_a: int, trials_a: int, successes_b: int, trials_b: int
) -> tuple[float, float]:
    """One-sided pooled two-proportion z-test of H1: rate A > rate B.

    Returns (z, p_value). With no variance (e.g. both 0% or both 100%) the
    test can't favour A, so it returns (0.0, 1.0).
    """
    if trials_a <= 0 or trials_b <= 0:
        return 0.0, 1.0
    p_a = successes_a / trials_a
    p_b = successes_b / trials_b
    pooled = (successes_a + successes_b) / (trials_a + trials_b)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / trials_a + 1.0 / trials_b))
    if se == 0.0:
        return 0.0, 1.0
    z = (p_a - p_b) / se
    return z, 0.5 * math.erfc(z / math.sqrt(2.0))
