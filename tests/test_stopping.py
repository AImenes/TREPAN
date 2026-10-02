import math

import pytest
from scipy import stats

from trepan.stopping import (
    cannot_become_pure,
    is_pure_enough,
    proportion_lower_bound,
    proportion_upper_bound,
    required_sample_size,
)


def test_required_sample_size_matches_thesis_formula():
    z = stats.norm.ppf(0.95)
    assert required_sample_size(0.05, 0.05) == math.ceil(z * z * 0.95 / 0.05) == 52
    assert required_sample_size(0.05, 0.01) > required_sample_size(0.05, 0.05)
    assert required_sample_size(0.01, 0.05) > required_sample_size(0.05, 0.05)
    with pytest.raises(ValueError):
        required_sample_size(0.0, 0.05)


def test_wilson_bound_with_unanimous_sample():
    # With p_hat = 1 the one-sided Wilson lower bound is 1 / (1 + z^2 / n).
    z = stats.norm.ppf(0.95)
    assert proportion_lower_bound(100, 100, 0.05) == pytest.approx(1 / (1 + z * z / 100))
    assert proportion_upper_bound(100, 100, 0.05) == 1.0
    assert proportion_lower_bound(0, 100, 0.05) == 0.0
    # m_L unanimous instances are exactly enough for 1 - epsilon.
    m_l = required_sample_size(0.05, 0.05)
    assert is_pure_enough(m_l, m_l, 0.05, 0.05) and not is_pure_enough(m_l - 2, m_l - 2, 0.05, 0.05)


def test_exact_bound_with_unanimous_sample():
    assert proportion_lower_bound(100, 100, 0.05, "exact") == pytest.approx(0.05 ** (1 / 100))


@pytest.mark.parametrize("method", ["wilson", "exact", "normal"])
def test_bounds_bracket_the_estimate_and_tighten_with_n(method):
    lo_small = proportion_lower_bound(80, 100, 0.05, method)
    lo_large = proportion_lower_bound(800, 1000, 0.05, method)
    hi_small = proportion_upper_bound(80, 100, 0.05, method)
    hi_large = proportion_upper_bound(800, 1000, 0.05, method)
    assert lo_small < 0.8 < hi_small
    assert lo_small < lo_large < 0.8 < hi_large < hi_small


def test_purity_decisions():
    assert is_pure_enough(1000, 1000, 0.05, 0.05)
    assert not is_pure_enough(940, 1000, 0.05, 0.05)
    assert is_pure_enough(990, 1000, 0.05, 0.05)
    assert cannot_become_pure(500, 1000, 0.05, 0.05)
    assert cannot_become_pure(10, 20, 0.05, 0.05)
    assert not cannot_become_pure(19, 20, 0.05, 0.05)  # too few instances to be sure either way
    assert not is_pure_enough(19, 20, 0.05, 0.05)


@pytest.mark.parametrize("kwargs", [dict(successes=-1, n=10), dict(successes=11, n=10), dict(successes=5, n=0)])
def test_invalid_counts(kwargs):
    with pytest.raises(ValueError):
        proportion_lower_bound(delta=0.05, **kwargs)


def test_invalid_delta_and_method():
    with pytest.raises(ValueError):
        proportion_lower_bound(5, 10, 0.0)
    with pytest.raises(ValueError):
        proportion_lower_bound(5, 10, 0.05, method="bayes")
