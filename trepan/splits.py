"""Split representations and the information-theoretic machinery used to score them.

TREPAN partitions the input space with *m-of-n* tests (Craven & Shavlik, 1995;
Murphy & Pazzani, 1991).  An m-of-n test is a Boolean expression made of ``n``
literals and an integer threshold ``m``; it is satisfied by an instance when at
least ``m`` of its ``n`` literals hold.  Every internal node of a TREPAN tree
carries one such test and has exactly two children: one for the instances that
satisfy the test and one for those that do not.

This module defines the immutable value objects used throughout the package:

* :class:`Literal` -- a single Boolean condition on one feature,
* :class:`MofNTest` -- an m-of-n expression over several literals,
* :class:`Constraint` -- an m-of-n test together with the outcome an instance
  must have on it (the path from the root to a node is a list of constraints),

the candidate-split generator (:func:`candidate_literals`) and the vectorised
scoring utilities (:func:`entropy`, :func:`information_gain`,
:func:`gain_ratio`, :func:`partition_chi2_pvalues`, :class:`LiteralBank`)
used by the split search.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "Literal",
    "MofNTest",
    "Constraint",
    "LiteralBank",
    "implies",
    "satisfies_all",
    "entropy",
    "information_gain",
    "gain_ratio",
    "partition_chi2_pvalues",
    "candidate_literals",
]

CONTINUOUS_OPS = ("<=", ">")
NOMINAL_OPS = ("==", "!=")
_NEGATION = {"<=": ">", ">": "<=", "==": "!=", "!=": "=="}


def _format_value(value: float) -> str:
    """Format a feature value compactly for human-readable output."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


@dataclass(frozen=True, order=True)
class Literal:
    """A single Boolean condition on one feature.

    Continuous features use threshold tests (``<=`` and its negation ``>``);
    nominal features use equality tests (``==`` and its negation ``!=``).

    Attributes:
        feature: Column index of the feature the literal refers to.
        op: One of ``"<="``, ``">"``, ``"=="`` or ``"!="``.
        value: Threshold (continuous) or category code (nominal).
    """

    feature: int
    op: str
    value: float

    def __post_init__(self) -> None:
        if self.op not in _NEGATION:
            raise ValueError(f"Unsupported operator {self.op!r}; expected one of {tuple(_NEGATION)}")
        # Normalise the stored types so that equal literals compare equal.
        object.__setattr__(self, "feature", int(self.feature))
        object.__setattr__(self, "value", float(self.value))

    @property
    def is_nominal(self) -> bool:
        """Whether the literal is an equality test on a nominal feature."""
        return self.op in NOMINAL_OPS

    def negate(self) -> Literal:
        """Return the literal with the opposite truth value."""
        return Literal(self.feature, _NEGATION[self.op], self.value)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the literal on every row of ``X``.

        Args:
            X: Two-dimensional array of shape ``(n_instances, n_features)``.

        Returns:
            Boolean array of shape ``(n_instances,)``.
        """
        column = X[:, self.feature]
        if self.op == "<=":
            return column <= self.value
        if self.op == ">":
            return column > self.value
        if self.op == "==":
            return column == self.value
        return column != self.value

    def describe(self, feature_names: Sequence[str] | None = None) -> str:
        """Human-readable rendering, e.g. ``petal_length <= 2.45``."""
        name = feature_names[self.feature] if feature_names is not None else f"x{self.feature}"
        return f"{name} {self.op} {_format_value(self.value)}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.describe()


def implies(a: Literal, b: Literal) -> bool:
    """Whether literal ``a`` logically implies literal ``b``.

    Only literals on the same feature can imply one another: ``x <= 1``
    implies ``x <= 2``, ``x > 2`` implies ``x > 1`` and ``colour == red``
    implies ``colour != blue``.  The thesis forbids adding to an m-of-n test a
    literal that is implied by (or implies) one already present.
    """
    if a.feature != b.feature:
        return False
    if a == b:
        return True
    if a.op == "<=" and b.op == "<=":
        return a.value <= b.value
    if a.op == ">" and b.op == ">":
        return a.value >= b.value
    if a.op == "==" and b.op == "!=":
        return a.value != b.value
    return False


@dataclass(frozen=True)
class MofNTest:
    """An m-of-n expression: satisfied when at least ``m`` of the ``n`` literals hold.

    A test with a single literal (``1-of-{a}``) is an ordinary binary split.  A
    test with ``n >= 2`` literals is called *compound* here; the thesis speaks
    of disjunctive m-of-n tests when discussing the restriction that a feature
    may not appear in two of them on the same root-to-leaf path.

    Attributes:
        m: Threshold; ``1 <= m <= n``.
        literals: The ``n`` literals, stored in a canonical sorted order so that
            two tests with the same literals compare equal.
    """

    m: int
    literals: tuple[Literal, ...]

    def __post_init__(self) -> None:
        literals = tuple(sorted(self.literals))
        if not literals:
            raise ValueError("An m-of-n test needs at least one literal")
        if not 1 <= self.m <= len(literals):
            raise ValueError(f"m must satisfy 1 <= m <= n; got m={self.m}, n={len(literals)}")
        if len(set(literals)) != len(literals):
            raise ValueError("Duplicate literal in m-of-n test")
        for lit in literals:
            if lit.negate() in literals:
                raise ValueError(f"Literal {lit} and its negation cannot both appear in one test")
        object.__setattr__(self, "m", int(self.m))
        object.__setattr__(self, "literals", literals)

    @property
    def n(self) -> int:
        """Number of literals in the test."""
        return len(self.literals)

    @property
    def is_compound(self) -> bool:
        """Whether the test combines two or more literals."""
        return self.n >= 2

    @property
    def features(self) -> frozenset[int]:
        """Indices of the features referenced by the literals."""
        return frozenset(lit.feature for lit in self.literals)

    def literal_counts(self, X: np.ndarray) -> np.ndarray:
        """Number of satisfied literals for every row of ``X``."""
        counts = np.zeros(len(X), dtype=np.int32)
        for lit in self.literals:
            counts += lit.evaluate(X)
        return counts

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Boolean array telling which rows of ``X`` satisfy the test."""
        return self.literal_counts(X) >= self.m

    def with_literal(self, literal: Literal, increment_m: bool) -> MofNTest | None:
        """Apply one of the two search operators of Murphy & Pazzani (1991).

        When ``literal`` is the negation of a literal already in the test, the
        pair is superfluous (exactly one of the two always holds) and the
        thesis' truth-preserving simplification is applied instead: both are
        dropped and ``m`` is decremented, e.g. ``2-of-{a, b, c} + not c`` gives
        ``1-of-{a, b}``.

        Args:
            literal: Literal to add to the set.
            increment_m: ``False`` for the *m-of-n+1* operator (hold the
                threshold), ``True`` for *m+1-of-n+1* (increment it).

        Returns:
            The new test, or ``None`` when the result would be degenerate
            (always true, never true, or without literals).
        """
        if literal in self.literals:
            return None
        new_m = self.m + int(increment_m)
        negation = literal.negate()
        if negation in self.literals:
            literals = tuple(lit for lit in self.literals if lit != negation)
            new_m -= 1
        else:
            literals = self.literals + (literal,)
        if not literals or not 1 <= new_m <= len(literals):
            return None
        return MofNTest(new_m, literals)

    def without_literal(self, literal: Literal, decrement_m: bool) -> MofNTest | None:
        """Drop ``literal`` from the test (used by the literal-pruning pass)."""
        literals = tuple(lit for lit in self.literals if lit != literal)
        new_m = self.m - int(decrement_m)
        if not literals or not 1 <= new_m <= len(literals):
            return None
        return MofNTest(new_m, literals)

    def describe(self, feature_names: Sequence[str] | None = None) -> str:
        """Human-readable rendering, e.g. ``2 of {a <= 1, b > 3, c == 2}``."""
        body = ", ".join(lit.describe(feature_names) for lit in self.literals)
        if self.n == 1:
            return body
        return f"{self.m} of {{{body}}}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.describe()


@dataclass(frozen=True)
class Constraint:
    """An m-of-n test paired with the outcome instances must have on it."""

    test: MofNTest
    outcome: bool

    @property
    def forced_literals(self) -> tuple[Literal, ...] | None:
        """Literals that every instance satisfying the constraint must satisfy.

        A satisfied n-of-n test forces all its literals; a failed 1-of-n test
        forces all their negations.  Other m-of-n constraints cannot be
        decomposed this way and ``None`` is returned.
        """
        if self.outcome and self.test.m == self.test.n:
            return self.test.literals
        if not self.outcome and self.test.m == 1:
            return tuple(lit.negate() for lit in self.test.literals)
        return None

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Boolean array telling which rows of ``X`` satisfy the constraint."""
        result = self.test.evaluate(X)
        return result if self.outcome else ~result

    def describe(self, feature_names: Sequence[str] | None = None) -> str:
        """Human-readable rendering, prefixing unsatisfied tests with ``not``."""
        text = self.test.describe(feature_names)
        if self.test.is_compound:
            text = f"({text})"
        return text if self.outcome else f"not {text}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.describe()


def satisfies_all(constraints: Iterable[Constraint], X: np.ndarray) -> np.ndarray:
    """Boolean mask of the rows of ``X`` that satisfy every constraint."""
    mask = np.ones(len(X), dtype=bool)
    for constraint in constraints:
        mask &= constraint.evaluate(X)
    return mask


# ---------------------------------------------------------------------------
# Information-theoretic scoring
# ---------------------------------------------------------------------------


def entropy(counts: np.ndarray) -> np.ndarray:
    """Shannon entropy (in bits) of one or many count vectors.

    Args:
        counts: Array whose last axis indexes the classes.  Rows that sum to
            zero get entropy zero.

    Returns:
        Array with the last axis removed.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, counts / np.where(total > 0, total, 1.0), 0.0)
        logp = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    return -(p * logp).sum(axis=-1)


def _weights(left_counts: np.ndarray, right_counts: np.ndarray):
    n_left = left_counts.sum(axis=-1)
    n_right = right_counts.sum(axis=-1)
    n = n_left + n_right
    with np.errstate(divide="ignore", invalid="ignore"):
        w_left = np.where(n > 0, n_left / np.where(n > 0, n, 1.0), 0.0)
        w_right = np.where(n > 0, n_right / np.where(n > 0, n, 1.0), 0.0)
    return n_left, n_right, w_left, w_right


def information_gain(left_counts: np.ndarray, right_counts: np.ndarray) -> np.ndarray:
    """Quinlan's information gain for one or many binary partitions.

    ``gain = info(S) - sum_i |S_i| / |S| * info(S_i)``; the thesis uses this
    (rather than the gain ratio) because TREPAN's tests are always binary.

    Args:
        left_counts: Class counts on the satisfied side, shape ``(..., n_classes)``.
        right_counts: Class counts on the other side, same shape.

    Returns:
        Array of gains with the class axis removed; degenerate partitions
        (one side empty) have gain zero.
    """
    left_counts = np.asarray(left_counts, dtype=float)
    right_counts = np.asarray(right_counts, dtype=float)
    _, _, w_left, w_right = _weights(left_counts, right_counts)
    gain = entropy(left_counts + right_counts) - (w_left * entropy(left_counts) + w_right * entropy(right_counts))
    return np.where(gain < 0, 0.0, gain)  # clip floating point round-off


def gain_ratio(left_counts: np.ndarray, right_counts: np.ndarray) -> np.ndarray:
    """Quinlan's gain ratio (information gain divided by the split information).

    Offered as an alternative criterion; the NeurIPS paper mentions it while
    the thesis settles on plain information gain.  A degenerate partition that
    leaves one side empty has gain ratio zero.
    """
    left_counts = np.asarray(left_counts, dtype=float)
    right_counts = np.asarray(right_counts, dtype=float)
    n_left, n_right, _, _ = _weights(left_counts, right_counts)
    gain = information_gain(left_counts, right_counts)
    split_info = entropy(np.stack([n_left, n_right], axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(split_info > 0, gain / np.where(split_info > 0, split_info, 1.0), 0.0)
    return np.where(ratio < 0, 0.0, ratio)


def partition_chi2_pvalues(reference_counts: np.ndarray, candidate_counts: np.ndarray) -> np.ndarray:
    """Two-sample chi-square test between a reference class distribution and many candidates.

    Implements the statistic used throughout the thesis (Press et al., 1992)::

        chi2 = sum_i (sqrt(m_B / m_A) * m_Ai - sqrt(m_A / m_B) * m_Bi)^2 / (m_Ai + m_Bi)

    with one degree of freedom fewer than the number of classes present (one
    more when the two samples have different sizes, following Press et al.).

    Args:
        reference_counts: Class counts of the reference sample, shape ``(n_classes,)``.
        candidate_counts: Class counts of each candidate sample, shape ``(k, n_classes)``.

    Returns:
        p-values of shape ``(k,)``; a candidate identical to the reference, or
        an empty sample, gets p-value 1.
    """
    a = np.asarray(reference_counts, dtype=float).reshape(1, -1)
    b = np.atleast_2d(np.asarray(candidate_counts, dtype=float))
    m_a = a.sum(axis=1)
    m_b = b.sum(axis=1)
    denominator = a + b
    present = denominator > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_ab = np.sqrt(np.where(m_a > 0, m_b / np.where(m_a > 0, m_a, 1.0), 0.0))[:, None]
        ratio_ba = np.sqrt(np.where(m_b > 0, m_a / np.where(m_b > 0, m_b, 1.0), 0.0))[:, None]
        terms = np.where(present, (ratio_ab * a - ratio_ba * b) ** 2 / np.where(present, denominator, 1.0), 0.0)
    statistic = terms.sum(axis=1)
    dof = present.sum(axis=1) - np.where(m_a == m_b, 1, 0)
    p_values = np.ones(len(b))
    valid = (dof > 0) & (m_a > 0) & (m_b > 0)
    p_values[valid] = stats.chi2.sf(statistic[valid], dof[valid])
    return p_values


# ---------------------------------------------------------------------------
# Candidate literals
# ---------------------------------------------------------------------------


def candidate_literals(
    X: np.ndarray,
    y: np.ndarray | None = None,
    categorical: Iterable[int] = (),
    max_thresholds: int | None = None,
) -> list[Literal]:
    """Build the candidate binary splits for the instances at a node.

    Following the thesis: a two-valued feature yields one split on its value, a
    nominal feature with more values yields one ``feature == value`` split per
    allowable value, and a continuous feature yields threshold splits halfway
    between adjacent observed values.  When labels are given, only *boundary*
    midpoints -- those between adjacent values whose instances are not all of
    one shared class -- are kept, since Fayyad & Irani (1992) proved that the
    information gain is maximised at such cut points.

    Args:
        X: Instances at the node, shape ``(n_instances, n_features)``.
        y: Class labels of the instances (optional; enables the boundary filter).
        categorical: Indices of the nominal features.
        max_thresholds: If given, continuous features with more candidate
            thresholds than this use evenly spaced quantiles instead, bounding
            the cost of the split search.

    Returns:
        Literals in their positive sense (``<=`` / ``==``) only; negations are
        generated on demand by :class:`LiteralBank`.
    """
    categorical = set(categorical)
    literals: list[Literal] = []
    for j in range(X.shape[1]):
        column = X[:, j]
        values = np.unique(column)
        if len(values) < 2:
            continue  # constant feature: nothing to split on
        if j in categorical:
            if len(values) == 2:
                literals.append(Literal(j, "==", values[0]))
            else:
                literals.extend(Literal(j, "==", v) for v in values)
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        if y is not None:
            thresholds = thresholds[_boundary_mask(column, np.asarray(y), values)]
        if max_thresholds is not None and len(thresholds) > max_thresholds:
            quantiles = np.linspace(0, 1, max_thresholds + 2)[1:-1]
            thresholds = np.unique(np.quantile(column, quantiles))
            thresholds = thresholds[thresholds < values[-1]]  # a cut at the maximum is degenerate
        literals.extend(Literal(j, "<=", t) for t in thresholds)
    return literals


def _boundary_mask(column: np.ndarray, y: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Which gaps between adjacent distinct values separate instances of different classes."""
    order = np.argsort(column, kind="stable")
    sorted_values = column[order]
    sorted_labels = y[order]
    # For every distinct value: does the group hold a single class, and which one?
    starts = np.searchsorted(sorted_values, values, side="left")
    ends = np.searchsorted(sorted_values, values, side="right")
    single_class = np.empty(len(values), dtype=object)
    for i, (s, e) in enumerate(zip(starts, ends)):
        labels = np.unique(sorted_labels[s:e])
        single_class[i] = labels[0] if len(labels) == 1 else None
    mask = np.ones(len(values) - 1, dtype=bool)
    for i in range(len(values) - 1):
        a, b = single_class[i], single_class[i + 1]
        if a is not None and b is not None and a == b:
            mask[i] = False  # both neighbouring groups are pure and of the same class
    return mask


class LiteralBank:
    """Pre-computed truth table of candidate literals over a set of instances.

    The split search evaluates thousands of literal/test combinations.  To keep
    that cheap, the truth value of every candidate literal (and its negation)
    on every instance is computed once and stored as a boolean matrix; scoring
    a test then reduces to integer additions and one matrix product with the
    one-hot class matrix.

    Attributes:
        literals: Candidate literals, positives first then their negations.
        matrix: Boolean array of shape ``(n_literals, n_instances)``.
    """

    def __init__(self, literals: Sequence[Literal], X: np.ndarray) -> None:
        positives = list(literals)
        self.n_positive = len(positives)
        self.literals: list[Literal] = positives + [lit.negate() for lit in positives]
        self.features = np.array([lit.feature for lit in self.literals], dtype=int)
        self.matrix = np.zeros((len(self.literals), len(X)), dtype=bool)
        for i, lit in enumerate(self.literals):
            self.matrix[i] = lit.evaluate(X)
        self._index = {lit: i for i, lit in enumerate(self.literals)}

    def __len__(self) -> int:
        return len(self.literals)

    @property
    def is_empty(self) -> bool:
        """Whether there are no candidate literals at all."""
        return self.n_positive == 0

    def row(self, literal: Literal) -> np.ndarray:
        """Truth values of ``literal`` on the stored instances."""
        return self.matrix[self._index[literal]]

    def score_binary_splits(self, y_onehot: np.ndarray, criterion=information_gain) -> np.ndarray:
        """Score every *positive* literal used as a binary split."""
        positive = self.matrix[: self.n_positive]
        left = positive.astype(float) @ y_onehot
        right = y_onehot.sum(axis=0) - left
        return criterion(left, right)

    def eligible_mask(self, test: MofNTest, excluded_features: Iterable[int] = ()) -> np.ndarray:
        """Which literals may be added to ``test`` by the search operators.

        A literal is ineligible when it is already in the test, when it is
        implied by or implies a literal already in the test (``x > 0.5`` next
        to ``x > 0.8``), or when its feature is excluded because an ancestor's
        compound split already uses it.  The negation of a present literal is
        eligible: adding it triggers the simplification in
        :meth:`MofNTest.with_literal`.
        """
        mask = np.ones(len(self.literals), dtype=bool)
        excluded = set(excluded_features)
        for i, lit in enumerate(self.literals):
            if lit.feature in excluded or lit in test.literals:
                mask[i] = False
                continue
            for existing in test.literals:
                if existing.feature == lit.feature and (implies(existing, lit) or implies(lit, existing)):
                    mask[i] = False
                    break
        return mask
