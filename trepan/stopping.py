"""The local stopping criterion of TREPAN.

A node becomes a leaf when, with high probability, it covers instances of a
single class only.  TREPAN estimates the proportion ``p_c`` of the most common
class among the instances at the node, puts a one-sided confidence interval
around it (Hogg & Tanis, 1983) and requires

    prob(p_c < 1 - epsilon) < delta

where ``epsilon`` and ``delta`` are parameters of the algorithm.

Craven's thesis (Section 3.2.5) applies the criterion only when every instance
at the node has the same class (``p_c = 1``) and derives from the Wilson score
interval the number of instances that must then have been seen::

    m_L = z_delta^2 * (1 - epsilon) / epsilon

The node is a leaf only if it has at least ``m_L`` instances and all of them
agree.  :func:`required_sample_size` computes ``m_L``; the bound functions
below support the more permissive "interval" variant in which a node with a
sufficiently tight lower confidence bound on ``p_c`` is accepted even when a
few instances disagree.

Three interval constructions are offered: ``"wilson"`` (the thesis' score
interval), ``"exact"`` (Clopper-Pearson, from the Beta quantiles of the
binomial) and ``"normal"`` (the Wald interval, which degenerates when
``p_c = 1``).
"""

from __future__ import annotations

import math

from scipy import stats

__all__ = [
    "required_sample_size",
    "proportion_lower_bound",
    "proportion_upper_bound",
    "is_pure_enough",
    "cannot_become_pure",
]

_METHODS = ("wilson", "exact", "normal")


def _check(successes: int, n: int, delta: float, method: str) -> None:
    if n <= 0:
        raise ValueError("Sample size must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")


def required_sample_size(epsilon: float, delta: float) -> int:
    """Instances needed before a unanimous node may become a leaf.

    Solves ``1 - epsilon = lower Wilson bound`` with ``p_hat = 1`` for the
    sample size: ``m_L = z_delta^2 (1 - epsilon) / epsilon`` (thesis, 3.2.5).
    For ``epsilon = delta = 0.05`` this is 52 instances.
    """
    if not 0 < epsilon < 1 or not 0 < delta < 1:
        raise ValueError("epsilon and delta must lie in (0, 1)")
    z = stats.norm.ppf(1 - delta)
    return int(math.ceil(z * z * (1 - epsilon) / epsilon))


def proportion_lower_bound(successes: int, n: int, delta: float, method: str = "wilson") -> float:
    """One-sided lower confidence bound on a binomial proportion.

    The true proportion falls below the returned value with probability at
    most ``delta``.
    """
    _check(successes, n, delta, method)
    if successes == 0:
        return 0.0
    p = successes / n
    z = stats.norm.ppf(1 - delta)
    if method == "wilson":
        centre = p + z * z / (2 * n)
        half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, (centre - half_width) / (1 + z * z / n))
    if method == "exact":
        return float(stats.beta.ppf(delta, successes, n - successes + 1))
    return max(0.0, p - z * math.sqrt(p * (1 - p) / n))


def proportion_upper_bound(successes: int, n: int, delta: float, method: str = "wilson") -> float:
    """One-sided upper confidence bound on a binomial proportion.

    The true proportion exceeds the returned value with probability at most
    ``delta``.
    """
    _check(successes, n, delta, method)
    if successes == n:
        return 1.0
    p = successes / n
    z = stats.norm.ppf(1 - delta)
    if method == "wilson":
        centre = p + z * z / (2 * n)
        half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return min(1.0, (centre + half_width) / (1 + z * z / n))
    if method == "exact":
        return float(stats.beta.ppf(1 - delta, successes + 1, n - successes))
    return min(1.0, p + z * math.sqrt(p * (1 - p) / n))


def is_pure_enough(successes: int, n: int, epsilon: float, delta: float, method: str = "wilson") -> bool:
    """``True`` when ``prob(p_c < 1 - epsilon) < delta`` holds for the sample."""
    return proportion_lower_bound(successes, n, delta, method) >= 1 - epsilon


def cannot_become_pure(successes: int, n: int, epsilon: float, delta: float, method: str = "wilson") -> bool:
    """``True`` when the sample already shows the node is impure.

    If even the upper confidence bound on ``p_c`` is below ``1 - epsilon`` there
    is no point in querying for more instances: the node will not pass the
    purity test and should be expanded instead.
    """
    return proportion_upper_bound(successes, n, delta, method) < 1 - epsilon
