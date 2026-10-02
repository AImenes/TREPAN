import numpy as np
import pytest

from trepan.search import best_binary_split, search_m_of_n
from trepan.splits import LiteralBank, candidate_literals, entropy, gain_ratio


def majority_concept(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 3)).astype(float)
    y = (X.sum(axis=1) >= 2).astype(int)  # 2-of-{x0, x1, x2}
    return X, y


def one_hot(y):
    classes, idx = np.unique(y, return_inverse=True)
    return np.eye(len(classes))[idx]


def prepared(X, y, **kwargs):
    y_onehot = one_hot(y)
    bank = LiteralBank(candidate_literals(X, y, **kwargs), X)
    seed, seed_score = best_binary_split(bank, y_onehot)
    return bank, y_onehot, seed, seed_score


@pytest.mark.parametrize("beam_width", [1, 2, 5])
def test_search_recovers_two_of_three(beam_width):
    X, y = majority_concept()
    bank, y_onehot, seed, seed_score = prepared(X, y, categorical=[0, 1, 2])
    assert seed.n == 1 and 0 < seed_score < entropy(y_onehot.sum(axis=0))
    test, score = search_m_of_n(seed, seed_score, bank, y_onehot, beam_width=beam_width)
    assert score == pytest.approx(entropy(y_onehot.sum(axis=0)))  # all the class information
    assert test.n == 3 and test.m == 2 and test.features == {0, 1, 2}
    assert np.mean(test.evaluate(X) == (y == 1)) in (0.0, 1.0)  # same partition, sides possibly swapped


def test_gain_ratio_criterion_also_recovers_the_concept():
    X, y = majority_concept()
    y_onehot = one_hot(y)
    bank = LiteralBank(candidate_literals(X, y, categorical=[0, 1, 2]), X)
    seed, seed_score = best_binary_split(bank, y_onehot, gain_ratio)
    test, score = search_m_of_n(seed, seed_score, bank, y_onehot, criterion=gain_ratio)
    assert score == pytest.approx(1.0) and test.n == 3


def test_max_literals_is_respected():
    X, y = majority_concept()
    bank, y_onehot, seed, seed_score = prepared(X, y, categorical=[0, 1, 2])
    test, _ = search_m_of_n(seed, seed_score, bank, y_onehot, max_literals=2)
    assert test.n <= 2


def test_excluded_seed_feature_keeps_binary_split():
    X, y = majority_concept()
    bank, y_onehot, seed, seed_score = prepared(X, y, categorical=[0, 1, 2])
    test, score = search_m_of_n(seed, seed_score, bank, y_onehot, excluded_features={seed.literals[0].feature})
    assert test == seed and score == seed_score


def test_excluded_features_never_enter_compound_split():
    X, y = majority_concept()
    bank, y_onehot, seed, seed_score = prepared(X, y, categorical=[0, 1, 2])
    other = ({0, 1, 2} - {seed.literals[0].feature}).pop()
    test, _ = search_m_of_n(seed, seed_score, bank, y_onehot, excluded_features={other})
    assert other not in test.features


def test_min_improvement_blocks_extensions():
    X, y = majority_concept()
    bank, y_onehot, seed, seed_score = prepared(X, y, categorical=[0, 1, 2])
    test, _ = search_m_of_n(seed, seed_score, bank, y_onehot, min_improvement=10.0)
    assert test == seed


def test_best_binary_split_none_when_uninformative():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_onehot = one_hot(np.array([0, 0, 0, 0]))
    bank = LiteralBank(candidate_literals(X), X)
    assert best_binary_split(bank, y_onehot) is None  # a single class: no gain anywhere
    assert best_binary_split(LiteralBank([], X), y_onehot) is None


def test_disjunction_of_thresholds_is_found():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 10, size=(2000, 2))
    y = ((X[:, 0] <= 3) | (X[:, 1] <= 3)).astype(int)  # 1-of-{x0 <= 3, x1 <= 3}
    bank, y_onehot, seed, seed_score = prepared(X, y, max_thresholds=50)
    test, score = search_m_of_n(seed, seed_score, bank, y_onehot)
    assert test.m == 1 and test.n == 2 and test.features == {0, 1}
    assert np.mean(test.evaluate(X) == (y == 1)) > 0.97
    # Never two literals with the same direction on one feature (one would imply the other).
    for lit in test.literals:
        same = [o for o in test.literals if o.feature == lit.feature and o.op == lit.op]
        assert len(same) == 1


def test_single_threshold_concept_stays_binary():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 10, size=(2000, 3))
    y = (X[:, 0] <= 4).astype(int)
    bank, y_onehot, seed, seed_score = prepared(X, y, max_thresholds=50)
    test, score = search_m_of_n(seed, seed_score, bank, y_onehot)
    assert test.n == 1 and score == pytest.approx(seed_score)


def test_chi_square_test_blocks_spurious_literals():
    # x0 decides the class; x1 is pure noise.  Without the admissibility test the search
    # may still pick up noise literals that marginally raise the gain on this sample.
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.integers(0, 2, 400), rng.integers(0, 2, 400)]).astype(float)
    y = X[:, 0].astype(int)
    y[rng.random(400) < 0.08] ^= 1  # a little label noise
    bank, y_onehot, seed, seed_score = prepared(X, y, categorical=[0, 1])
    test, _ = search_m_of_n(seed, seed_score, bank, y_onehot, significance=0.05, prune_literals=False)
    assert test.n == 1
