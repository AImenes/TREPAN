"""Construction of the m-of-n splitting test of an internal node.

``ConstructTest`` (thesis, Figure 15) scores every candidate binary test with
the information gain criterion and hands the best one as a seed to
``ConstructMofNTest`` (Figure 16), a beam search over m-of-n tests driven by
the two operators of Murphy & Pazzani (1991):

* *m-of-n+1* -- add a literal and hold the threshold, ``2-of-{a,b} => 2-of-{a,b,c}``;
* *m+1-of-n+1* -- add a literal and increment the threshold, ``2-of-{a,b,c} => 3-of-{a,b,c,d}``.

An operator application is admissible only if the new test partitions the
instances significantly differently from the test it extends (a chi-square
test; this keeps the search from chasing tiny, spurious gains) and scores
better than the worst test in the beam.  The search stops when an iteration
leaves the beam unchanged.  The beam has width two and starts from the seed
and its complement, so that the search can pursue either sense of the seed
literal (the thesis' motivation for a beam over hill climbing).  Adding the
negation of a present literal is allowed and simplifies the test, a literal
implied by one already present is not, and a feature already used by a
compound test on the path is not available.  A final literal-pruning pass
drops literals (in the order they were added) when doing so does not reduce
the gain.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .splits import Literal, LiteralBank, MofNTest, information_gain, partition_chi2_pvalues

__all__ = ["best_binary_split", "search_m_of_n"]

Criterion = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class _Candidate:
    test: MofNTest
    counts: np.ndarray  # satisfied-literal counts per instance
    left: np.ndarray  # class counts on the satisfied side
    score: float
    history: tuple[Literal, ...]  # literals in the order they were added


def _ordering(candidate: _Candidate) -> tuple[float, int, int]:
    """Sort key: higher score first, then simpler tests."""
    return (-candidate.score, candidate.test.n, candidate.test.m)


def _evaluate(test: MofNTest, bank: LiteralBank, y_onehot: np.ndarray, criterion: Criterion, history) -> _Candidate:
    counts = np.zeros(y_onehot.shape[0], dtype=np.int16)
    for lit in test.literals:
        counts += bank.row(lit)
    left = (counts >= test.m).astype(float) @ y_onehot
    score = float(criterion(left, y_onehot.sum(axis=0) - left))
    return _Candidate(test, counts, left, score, tuple(history))


def best_binary_split(
    bank: LiteralBank, y_onehot: np.ndarray, criterion: Criterion = information_gain
) -> tuple[MofNTest, float] | None:
    """Return the single-literal test with the highest score, or ``None``.

    ``None`` means no candidate literal yields any information gain, in which
    case the node should stay a leaf.
    """
    if bank.is_empty:
        return None
    scores = bank.score_binary_splits(y_onehot, criterion)
    best = int(np.argmax(scores))
    if scores[best] <= 0:
        return None
    return MofNTest(1, (bank.literals[best],)), float(scores[best])


def search_m_of_n(
    seed: MofNTest,
    seed_score: float,
    bank: LiteralBank,
    y_onehot: np.ndarray,
    *,
    criterion: Criterion = information_gain,
    beam_width: int = 2,
    max_literals: int | None = None,
    significance: float | None = 0.05,
    min_improvement: float = 0.0,
    excluded_features: Iterable[int] = (),
    prune_literals: bool = True,
) -> tuple[MofNTest, float]:
    """Grow ``seed`` into the best m-of-n test found by beam search.

    Args:
        seed: Best binary split at the node (a single-literal test).
        seed_score: Its score.
        bank: Truth table of the candidate literals on the node's instances.
        y_onehot: One-hot oracle labels of those instances, shape ``(n, n_classes)``.
        criterion: Scoring function, :func:`~trepan.splits.information_gain`
            (thesis) or :func:`~trepan.splits.gain_ratio` (paper).
        beam_width: Number of tests kept between iterations (1 = hill climbing).
        max_literals: Upper bound on ``n``; ``None`` leaves it unbounded.
        significance: Level of the chi-square admissibility test; ``None``
            disables the test.
        min_improvement: Extra margin a candidate must have over the worst
            test in the beam to enter it (0 reproduces the thesis).
        excluded_features: Features already used by compound splits on the path.
        prune_literals: Run the literal-pruning pass on the selected test.

    Returns:
        The selected test and its score.  The seed itself is returned when no
        admissible extension improves on it.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    excluded = set(excluded_features)
    seed_literal = seed.literals[0]
    if seed_literal.feature in excluded:
        # Extending the seed would put its feature into a second compound test
        # on this path, which TREPAN forbids: keep the binary split.
        return seed, seed_score

    totals = y_onehot.sum(axis=0)
    beam = [_evaluate(seed, bank, y_onehot, criterion, seed.literals)]
    if beam_width >= 2:
        complement = MofNTest(1, (seed_literal.negate(),))
        beam.append(_evaluate(complement, bank, y_onehot, criterion, complement.literals))
    seen = {candidate.test for candidate in beam}

    while True:
        worst = min(c.score for c in beam) if len(beam) >= beam_width else -np.inf
        new_candidates: list[_Candidate] = []
        for candidate in beam:
            if max_literals is not None and candidate.test.n >= max_literals:
                continue
            eligible = np.flatnonzero(bank.eligible_mask(candidate.test, excluded))
            if eligible.size == 0:
                continue
            # Row k holds the literal counts after adding eligible literal k.  When
            # the literal is the negation of a present one the simplified test has
            # counts one lower and threshold one lower, so the same mask applies.
            counts_matrix = candidate.counts[None, :] + bank.matrix[eligible]
            for increment in (False, True):
                m_new = candidate.test.m + int(increment)
                left_matrix = (counts_matrix >= m_new).astype(float) @ y_onehot
                scores = criterion(left_matrix, totals - left_matrix)
                admissible = scores > worst + min_improvement
                if significance is not None:
                    p_left = partition_chi2_pvalues(candidate.left, left_matrix)
                    p_right = partition_chi2_pvalues(totals - candidate.left, totals - left_matrix)
                    admissible &= (p_left < significance) | (p_right < significance)
                for k in np.flatnonzero(admissible):
                    literal = bank.literals[eligible[k]]
                    test = candidate.test.with_literal(literal, increment)
                    if test is None or test in seen:
                        continue
                    seen.add(test)
                    if literal.negate() in candidate.test.literals:
                        counts, history = (
                            counts_matrix[k] - 1,
                            tuple(h for h in candidate.history if h != literal.negate()),
                        )
                    else:
                        counts, history = counts_matrix[k], candidate.history + (literal,)
                    new_candidates.append(_Candidate(test, counts, left_matrix[k], float(scores[k]), history))
        if not new_candidates:
            break
        new_beam = sorted(beam + new_candidates, key=_ordering)[:beam_width]
        if [c.test for c in new_beam] == [c.test for c in beam]:
            break
        beam = new_beam

    best = min(beam, key=_ordering)
    if prune_literals and best.test.n > 1:
        best = _prune_literals(best, bank, y_onehot, criterion)
    return best.test, best.score


def _prune_literals(candidate: _Candidate, bank: LiteralBank, y_onehot: np.ndarray, criterion: Criterion) -> _Candidate:
    """Drop literals whose removal does not reduce the score (thesis, Section 3.2.4).

    Literals are visited in the order they were added; for each one, dropping
    it with ``m`` held constant and dropping it with ``m`` decremented are
    tried, and the better modification is kept when it loses no score.
    """
    current = candidate
    for literal in candidate.history:
        if literal not in current.test.literals or current.test.n == 1:
            continue
        options = []
        for decrement in (False, True):
            test = current.test.without_literal(literal, decrement)
            if test is not None:
                options.append(_evaluate(test, bank, y_onehot, criterion, [h for h in current.history if h != literal]))
        if not options:
            continue
        best_option = min(options, key=_ordering)
        if best_option.score >= current.score - 1e-12:
            current = best_option
    return current
