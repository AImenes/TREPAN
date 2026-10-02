"""Modelling feature distributions and drawing the oracle's query instances.

Queries to the oracle need not be complete instances: a node hands the
``DrawSample`` routine the constraints on the path from the root and gets back
complete instances that satisfy them.  To generate them TREPAN models the
*marginal* distribution of every feature from the training data (thesis,
Section 3.2.2):

* the empirical frequency distribution for discrete-valued features,
* a Gaussian kernel density estimate (Silverman, 1986) for continuous ones,
  with kernel width ``1 / sqrt(m)`` for ``m`` fitted examples (expressed here
  in units of the feature's range, since the thesis assumes inputs scaled to
  ``[0, 1]``).  The width shrinks with more data and gives smooth, near-Gaussian
  estimates when data is scarce.

The marginals are estimated *locally*: a node compares, for every feature not
constrained by a test on its path, the examples that reach it with the
examples behind the nearest ancestor model (a chi-square test for discrete
features, a Kolmogorov-Smirnov test for continuous ones, at a
Bonferroni-corrected level).  If any feature differs significantly the node
fits its own model from its examples; otherwise it reuses the ancestor's,
which rests on more data.

Instances are drawn with the thesis' ``DrawInstance`` procedure (Figure 13):
constraints that pin down individual literals (binary splits, satisfied
n-of-n tests, failed 1-of-n tests) become *hard* per-feature bounds and each
feature is sampled from its conditional distribution.  For a genuinely
disjunctive m-of-n constraint, literals are chosen at random -- with
probability proportional to how likely each is under the current conditional
distributions -- and added to the hard bounds until the constraint is
guaranteed (``m`` literals forced true, or ``n - m + 1`` forced false).  Every
drawn instance is finally verified against the constraints; the rare
violations caused by rounding are redrawn.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .splits import Constraint, Literal, satisfies_all

__all__ = [
    "FeatureBounds",
    "derive_bounds",
    "NominalDistribution",
    "KernelDensityDistribution",
    "FeatureDistributions",
    "draw_instances",
]

logger = logging.getLogger(__name__)

_INF = math.inf


@dataclass(frozen=True)
class FeatureBounds:
    """Hard constraints on one feature implied by the path.

    Continuous features are restricted to the half-open interval
    ``(lower, upper]``; nominal features to ``allowed`` minus ``excluded``
    (``allowed=None`` means every value).  Instances are hashable so that
    conditional probabilities can be cached per bound.
    """

    lower: float = -_INF
    upper: float = _INF
    allowed: frozenset | None = None
    excluded: frozenset = frozenset()

    @property
    def is_trivial(self) -> bool:
        """Whether the bounds impose no restriction at all."""
        return self.lower == -_INF and self.upper == _INF and self.allowed is None and not self.excluded

    def with_literal(self, literal: Literal) -> FeatureBounds:
        """Tighten the bounds so that ``literal`` holds."""
        if literal.op == "<=":
            return FeatureBounds(self.lower, min(self.upper, literal.value), self.allowed, self.excluded)
        if literal.op == ">":
            return FeatureBounds(max(self.lower, literal.value), self.upper, self.allowed, self.excluded)
        if literal.op == "==":
            allowed = frozenset({literal.value}) if self.allowed is None else self.allowed & {literal.value}
            return FeatureBounds(self.lower, self.upper, allowed, self.excluded)
        return FeatureBounds(self.lower, self.upper, self.allowed, self.excluded | {literal.value})

    def permits_values(self, values: np.ndarray) -> np.ndarray:
        """Boolean mask of the nominal ``values`` that satisfy the bounds."""
        mask = np.ones(len(values), dtype=bool)
        if self.allowed is not None:
            mask &= np.isin(values, list(self.allowed))
        if self.excluded:
            mask &= ~np.isin(values, list(self.excluded))
        return mask

    def contains(self, samples: np.ndarray) -> np.ndarray:
        """Boolean mask of the continuous ``samples`` inside ``(lower, upper]``."""
        return (samples > self.lower) & (samples <= self.upper)


def derive_bounds(constraints: Iterable[Constraint], n_features: int) -> list[FeatureBounds]:
    """Translate the constraints that force individual literals into per-feature bounds.

    Compound m-of-n constraints that cannot be decomposed are ignored here; see
    :func:`draw_instances` for how they are handled.
    """
    bounds = [FeatureBounds() for _ in range(n_features)]
    for constraint in constraints:
        forced = constraint.forced_literals
        if forced is None:
            continue
        for lit in forced:
            bounds[lit.feature] = bounds[lit.feature].with_literal(lit)
    return bounds


# ---------------------------------------------------------------------------
# Marginal models
# ---------------------------------------------------------------------------


@dataclass
class NominalDistribution:
    """Empirical frequency distribution of a discrete-valued feature."""

    values: np.ndarray
    probabilities: np.ndarray
    data: np.ndarray

    @classmethod
    def fit(cls, column: np.ndarray) -> NominalDistribution:
        """Estimate the value frequencies from ``column``."""
        values, counts = np.unique(column, return_counts=True)
        return cls(values=values, probabilities=counts / counts.sum(), data=np.asarray(column, dtype=float))

    def _restricted(self, bounds: FeatureBounds | None):
        if bounds is None or bounds.is_trivial:
            return self.values, self.probabilities
        mask = bounds.permits_values(self.values)
        if not mask.any():
            return self.values[:0], self.probabilities[:0]
        probabilities = self.probabilities[mask]
        return self.values[mask], probabilities / probabilities.sum()

    def sample(self, rng: np.random.Generator, size: int, bounds: FeatureBounds | None = None) -> np.ndarray:
        """Draw ``size`` values from the distribution conditioned on ``bounds``.

        Returns an empty array when no fitted value is permitted.
        """
        values, probabilities = self._restricted(bounds)
        if len(values) == 0:
            return np.empty(0, dtype=float)
        return rng.choice(values, size=size, p=probabilities)

    def prob(self, literal: Literal, bounds: FeatureBounds | None = None) -> float:
        """Probability that ``literal`` holds for a value drawn under ``bounds``."""
        values, probabilities = self._restricted(bounds)
        if len(values) == 0:
            return 0.0
        p_equal = float(probabilities[values == literal.value].sum())
        if literal.op == "==":
            return p_equal
        if literal.op == "!=":
            return 1.0 - p_equal
        if literal.op == "<=":
            return float(probabilities[values <= literal.value].sum())
        return float(probabilities[values > literal.value].sum())

    def differs_from(self, column: np.ndarray, alpha: float) -> bool:
        """Chi-square test of homogeneity between ``column`` and the fitted data."""
        categories = np.union1d(self.values, np.unique(column))
        if len(categories) < 2 or len(column) < 2:
            return False
        table = np.vstack(
            [
                np.array([np.sum(self.data == c) for c in categories]),
                np.array([np.sum(column == c) for c in categories]),
            ]
        )
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 2:
            return False
        _, p_value, _, _ = stats.chi2_contingency(table)
        return bool(p_value < alpha)


@dataclass
class KernelDensityDistribution:
    """Gaussian kernel density estimate of a continuous feature.

    Sampling from a Gaussian KDE picks one of the fitted values at random and
    adds Gaussian noise with the kernel width as standard deviation
    (Silverman, 1986, p. 143).  As in the thesis, samples are restricted to
    the range required by the conditional distribution; optionally they are
    also kept within the range observed in the training data, and features
    whose observed values are all integers keep integer samples.

    Attributes:
        data: Fitted values.
        bandwidth: Kernel standard deviation.
        support: ``(low, high)`` range samples are kept in, or ``None``.
        integer_valued: Round samples to the nearest integer.
    """

    data: np.ndarray
    bandwidth: float
    support: tuple[float, float] | None
    integer_valued: bool

    @staticmethod
    def select_bandwidth(column: np.ndarray, rule: str | float, scale: float | None = None) -> float:
        """Kernel width from Craven's, Silverman's or Scott's rule, or a fixed value.

        Args:
            column: Fitted values.
            rule: ``"craven"`` for ``scale / sqrt(m)`` (the thesis' ``1 / sqrt(m)``
                for features scaled to unit range), ``"silverman"``, ``"scott"``
                or a non-negative number.
            scale: Range of the feature in the full training set, used by the
                ``"craven"`` rule; defaults to the range of ``column``.
        """
        if isinstance(rule, (int, float)) and not isinstance(rule, bool):
            if rule < 0:
                raise ValueError("A fixed bandwidth must be non-negative")
            return float(rule)
        m = len(column)
        if m == 0:
            return 0.0
        if rule == "craven":
            if scale is None:
                scale = float(column.max() - column.min())
            return scale / math.sqrt(m)
        if m < 2:
            return 0.0
        std = float(np.std(column, ddof=1))
        if rule == "silverman":
            iqr = float(np.subtract(*np.percentile(column, [75, 25])))
            spread = min(std, iqr / 1.34) if iqr > 0 else std
            return 0.9 * spread * m ** (-1 / 5)
        if rule == "scott":
            return std * m ** (-1 / 5)
        raise ValueError(f"Unknown bandwidth rule {rule!r}; use 'craven', 'silverman', 'scott' or a number")

    @classmethod
    def fit(
        cls,
        column: np.ndarray,
        bandwidth: str | float = "craven",
        scale: float | None = None,
        support: tuple[float, float] | None = None,
    ) -> KernelDensityDistribution:
        """Fit the estimate to ``column`` (see :meth:`select_bandwidth` for the width)."""
        column = np.asarray(column, dtype=float)
        return cls(
            data=column,
            bandwidth=cls.select_bandwidth(column, bandwidth, scale),
            support=None if support is None else (float(support[0]), float(support[1])),
            integer_valued=bool(len(column) and np.all(np.mod(column, 1) == 0)),
        )

    # -- probabilities -----------------------------------------------------

    def cdf(self, t: float) -> float:
        """Cumulative distribution function of the (unrestricted) estimate."""
        if self.bandwidth <= 0:
            return float(np.mean(self.data <= t))
        return float(np.mean(stats.norm.cdf((t - self.data) / self.bandwidth)))

    def _interval(self, bounds: FeatureBounds | None) -> tuple[float, float]:
        """Effective ``(low, high]`` interval: path bounds intersected with the support."""
        low, high = -_INF, _INF
        if self.support is not None:
            low, high = np.nextafter(self.support[0], -_INF), self.support[1]
        if bounds is not None:
            low, high = max(low, bounds.lower), min(high, bounds.upper)
        return low, high

    def _prob_leq(self, t: float, low: float, high: float) -> float:
        """``P(value <= t | low < value <= high)`` taking integer rounding into account."""
        if self.integer_valued:
            t = math.floor(t) + 0.5  # rint(s) <= t  <=>  s < floor(t) + 0.5
            low = math.floor(low) + 0.5 if low > -_INF else low
            high = math.floor(high) + 0.5 if high < _INF else high
        if t <= low:
            return 0.0
        denominator = self.cdf(high) - self.cdf(low) if (low > -_INF or high < _INF) else 1.0
        if denominator <= 0:
            return 0.0
        numerator = self.cdf(min(t, high)) - (self.cdf(low) if low > -_INF else 0.0)
        return float(min(1.0, max(0.0, numerator / denominator)))

    def prob(self, literal: Literal, bounds: FeatureBounds | None = None) -> float:
        """Probability that ``literal`` holds for a value drawn under ``bounds``."""
        low, high = self._interval(bounds)
        if low >= high:
            return 0.0
        if literal.op == "<=":
            return self._prob_leq(literal.value, low, high)
        if literal.op == ">":
            return 1.0 - self._prob_leq(literal.value, low, high)
        # Equality tests on a continuous feature are unusual; estimate by simulation.
        samples = self.sample(np.random.default_rng(0), 2000, bounds)
        if len(samples) == 0:
            return 0.0
        p_equal = float(np.mean(samples == literal.value))
        return p_equal if literal.op == "==" else 1.0 - p_equal

    # -- sampling -----------------------------------------------------------

    def _sample_raw(self, rng: np.random.Generator, size: int) -> np.ndarray:
        centres = self.data[rng.integers(len(self.data), size=size)]
        samples = centres + rng.normal(0.0, self.bandwidth, size=size) if self.bandwidth > 0 else centres.copy()
        if self.integer_valued:
            samples = np.rint(samples)
        return samples

    def sample(
        self,
        rng: np.random.Generator,
        size: int,
        bounds: FeatureBounds | None = None,
        max_attempts_factor: int = 1_000,
    ) -> np.ndarray:
        """Draw ``size`` values restricted to the support and to ``bounds``.

        Values outside the interval are rejected and redrawn.  If the interval
        carries (almost) no probability mass the result may be shorter than
        ``size``, possibly empty.
        """
        if size <= 0 or len(self.data) == 0:
            return np.empty(0, dtype=float)
        low, high = self._interval(bounds)
        if low == -_INF and high == _INF:
            return self._sample_raw(rng, size)
        if low >= high:
            return np.empty(0, dtype=float)
        accepted: list[np.ndarray] = []
        n_accepted = n_generated = 0
        batch = max(size, 256)
        max_attempts = max_attempts_factor * size
        while n_accepted < size and n_generated < max_attempts:
            candidates = self._sample_raw(rng, batch)
            n_generated += batch
            kept = candidates[(candidates > low) & (candidates <= high)]
            if len(kept):
                accepted.append(kept)
                n_accepted += len(kept)
            rate = max(n_accepted / n_generated, 1e-3)
            batch = int(min(max(256, math.ceil((size - n_accepted) / rate) * 1.2), 200_000))
        if not accepted:
            return np.empty(0, dtype=float)
        return np.concatenate(accepted)[:size]

    def differs_from(self, column: np.ndarray, alpha: float) -> bool:
        """Two-sample Kolmogorov-Smirnov test between ``column`` and the fitted data."""
        if len(column) < 2 or len(self.data) < 2:
            return False
        _, p_value = stats.ks_2samp(self.data, column)
        return bool(p_value < alpha)


class FeatureDistributions:
    """The instance model of one node: a marginal distribution per feature.

    Attributes:
        distributions: One :class:`NominalDistribution` or
            :class:`KernelDensityDistribution` per feature.
        X_fit: The examples the model was fitted on.
        categorical: Indices of the nominal features.
        depth: Depth of the node that fitted the model (for reporting).
    """

    def __init__(self, distributions: Sequence[object], X_fit: np.ndarray, categorical: Iterable[int], depth: int = 0):
        self.distributions = list(distributions)
        self.X_fit = X_fit
        self.categorical = set(categorical)
        self.depth = depth

    @property
    def n_features(self) -> int:
        """Number of modelled features."""
        return len(self.distributions)

    @property
    def n_fit(self) -> int:
        """Number of examples behind the model."""
        return len(self.X_fit)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        categorical: Iterable[int] = (),
        *,
        bandwidth: str | float = "craven",
        scales: Sequence[float] | None = None,
        supports: Sequence[tuple[float, float] | None] | None = None,
        depth: int = 0,
    ) -> FeatureDistributions:
        """Fit every marginal from the given examples.

        Args:
            X: Examples, shape ``(m, n_features)``.
            categorical: Indices of the nominal features.
            bandwidth: Kernel-width rule for continuous features.
            scales: Per-feature range in the whole training set (for ``"craven"``).
            supports: Per-feature ``(low, high)`` range to keep samples in, or ``None``.
            depth: Depth of the node the model belongs to.
        """
        categorical = set(categorical)
        distributions = []
        for j in range(X.shape[1]):
            column = X[:, j]
            if j in categorical:
                distributions.append(NominalDistribution.fit(column))
            else:
                scale = None if scales is None else scales[j]
                support = None if supports is None else supports[j]
                distributions.append(KernelDensityDistribution.fit(column, bandwidth, scale, support))
        return cls(distributions, X, categorical, depth)

    @classmethod
    def fit_local(
        cls,
        X: np.ndarray,
        ancestor: FeatureDistributions,
        constrained_features: Iterable[int] = (),
        *,
        alpha: float = 0.10,
        min_examples: int = 5,
        bandwidth: str | float = "craven",
        scales: Sequence[float] | None = None,
        supports: Sequence[tuple[float, float] | None] | None = None,
        depth: int = 0,
    ) -> FeatureDistributions:
        """Decide between a local model for a node and the nearest ancestor model.

        Every feature not constrained by a test on the path is compared with
        the ancestor model's examples at level ``alpha / k`` (Bonferroni
        correction over the ``k`` tested features).  A single significant
        difference makes the node fit its own model; otherwise the ancestor's
        model object is returned unchanged.

        Args:
            X: Training examples reaching the node.
            ancestor: Model of the nearest ancestor that fitted one.
            constrained_features: Features referenced by tests on the path.
            alpha: Overall significance level of the test.
            min_examples: Fewer examples than this always inherit.
            bandwidth: Kernel-width rule for continuous features.
            scales: Per-feature range in the whole training set (for ``"craven"``).
            supports: Per-feature ``(low, high)`` range to keep samples in, or ``None``.
            depth: Depth of the node the model belongs to.
        """
        constrained = set(constrained_features)
        unconstrained = [j for j in range(ancestor.n_features) if j not in constrained]
        if len(X) < min_examples or not unconstrained:
            return ancestor
        level = alpha / len(unconstrained)
        for j in unconstrained:
            if ancestor.distributions[j].differs_from(X[:, j], level):
                return cls.fit(
                    X, ancestor.categorical, bandwidth=bandwidth, scales=scales, supports=supports, depth=depth
                )
        return ancestor

    def prob(self, literal: Literal, bounds: FeatureBounds | None = None) -> float:
        """Probability that ``literal`` holds for a value drawn under ``bounds``."""
        return self.distributions[literal.feature].prob(literal, bounds)

    def sample(self, rng: np.random.Generator, size: int, bounds: Sequence[FeatureBounds] | None = None) -> np.ndarray:
        """Draw ``size`` instances feature by feature, honouring per-feature ``bounds``.

        If some bounded feature cannot supply enough values the result has
        fewer than ``size`` rows (possibly none).
        """
        if size <= 0:
            return np.empty((0, self.n_features), dtype=float)
        columns = [
            dist.sample(rng, size, None if bounds is None else bounds[j]) for j, dist in enumerate(self.distributions)
        ]
        n = min(len(c) for c in columns) if columns else size
        if n == 0:
            return np.empty((0, self.n_features), dtype=float)
        return np.column_stack([c[:n] for c in columns])


# ---------------------------------------------------------------------------
# DrawInstance / DrawSample
# ---------------------------------------------------------------------------


def _select_forced_literals(
    compound: Sequence[Constraint],
    base_bounds: Sequence[FeatureBounds],
    distributions: FeatureDistributions,
    rng: np.random.Generator,
    cache: dict,
) -> tuple[Literal, ...] | None:
    """Choose literals that guarantee every compound constraint (thesis, Figure 13).

    For each disjunctive constraint, literals are selected one at a time with
    probability proportional to their conditional probability of holding,
    each selection tightening the hard bounds of its feature, until ``m``
    literals are forced true (satisfied outcome) or ``n - m + 1`` forced false
    (unsatisfied outcome).  Returns ``None`` if some constraint cannot be
    guaranteed under the model.
    """
    bounds = list(base_bounds)
    forced: list[Literal] = []
    for constraint in compound:
        test = constraint.test
        if constraint.outcome:
            targets, needed = list(test.literals), test.m
        else:
            targets, needed = [lit.negate() for lit in test.literals], test.n - test.m + 1
        satisfied = 0
        while satisfied < needed:
            probabilities = []
            for lit in targets:
                key = (lit, bounds[lit.feature])
                if key not in cache:
                    cache[key] = distributions.prob(lit, bounds[lit.feature])
                probabilities.append(cache[key])
            probabilities = np.array(probabilities)
            # Literals that already hold with certainty under the bounds count as satisfied.
            certain = np.flatnonzero(probabilities >= 1 - 1e-12)
            if certain.size:
                satisfied += certain.size
                targets = [lit for i, lit in enumerate(targets) if i not in set(certain)]
                continue
            total = probabilities.sum()
            if total <= 0 or not targets:
                return None
            chosen = int(rng.choice(len(targets), p=probabilities / total))
            lit = targets.pop(chosen)
            bounds[lit.feature] = bounds[lit.feature].with_literal(lit)
            forced.append(lit)
            satisfied += 1
    return tuple(sorted(forced))


def draw_instances(
    distributions: FeatureDistributions,
    constraints: Sequence[Constraint],
    n: int,
    rng: np.random.Generator,
    *,
    max_attempts: int | None = None,
) -> np.ndarray:
    """Draw ``n`` instances that satisfy ``constraints`` from the node's instance model.

    Decomposable constraints become per-feature bounds; compound m-of-n
    constraints are satisfied by forcing randomly selected literals as in the
    thesis' ``DrawInstance``.  Drawn instances are verified and any that fail
    (rounding at a threshold, for instance) are redrawn, up to
    ``max_attempts`` candidates in total.

    Returns:
        Array of shape ``(k, n_features)`` with ``k <= n``; ``k < n`` only when
        the constrained region has (almost) no mass under the model, in which
        case a warning is logged.
    """
    if n <= 0:
        return np.empty((0, distributions.n_features), dtype=float)
    if max_attempts is None:
        max_attempts = max(10_000, 100 * n)
    base_bounds = derive_bounds(constraints, distributions.n_features)
    compound = [c for c in constraints if c.forced_literals is None]
    prob_cache: dict = {}

    accepted: list[np.ndarray] = []
    n_accepted = n_generated = 0
    while n_accepted < n and n_generated < max_attempts:
        wanted = n - n_accepted
        if compound:
            patterns: Counter = Counter()
            for _ in range(wanted):
                pattern = _select_forced_literals(compound, base_bounds, distributions, rng, prob_cache)
                patterns[pattern] += 1
            parts = []
            for pattern, count in patterns.items():
                if pattern is None:
                    continue  # the model cannot guarantee the constraints for these draws
                bounds = list(base_bounds)
                for lit in pattern:
                    bounds[lit.feature] = bounds[lit.feature].with_literal(lit)
                parts.append(distributions.sample(rng, count, bounds))
            candidates = np.vstack(parts) if parts else np.empty((0, distributions.n_features))
        else:
            candidates = distributions.sample(rng, wanted, base_bounds)
        n_generated += wanted
        if len(candidates) == 0:
            break
        kept = candidates[satisfies_all(constraints, candidates)]
        if len(kept):
            accepted.append(kept)
            n_accepted += len(kept)

    if n_accepted < n:
        logger.log(
            logging.WARNING if n_accepted < n / 2 else logging.DEBUG,
            "DrawSample produced %d of %d requested instances; the constrained region has little mass under the model.",
            n_accepted,
            n,
        )
    if not accepted:
        return np.empty((0, distributions.n_features), dtype=float)
    return np.vstack(accepted)[:n]
