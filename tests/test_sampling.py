import logging

import numpy as np
import pytest

from trepan.sampling import (
    FeatureBounds,
    FeatureDistributions,
    KernelDensityDistribution,
    NominalDistribution,
    derive_bounds,
    draw_instances,
)
from trepan.splits import Constraint, Literal, MofNTest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestFeatureBounds:
    def test_literals_tighten_bounds(self):
        b = FeatureBounds().with_literal(Literal(0, "<=", 5.0)).with_literal(Literal(0, ">", 1.0))
        assert (b.lower, b.upper) == (1.0, 5.0) and not b.is_trivial
        np.testing.assert_array_equal(b.contains(np.array([1.0, 1.5, 5.0, 5.5])), [False, True, True, False])
        n = FeatureBounds().with_literal(Literal(1, "!=", 2.0)).with_literal(Literal(1, "==", 1.0))
        assert n.allowed == {1.0} and n.excluded == {2.0}
        np.testing.assert_array_equal(n.permits_values(np.array([0.0, 1.0, 2.0])), [False, True, False])
        assert FeatureBounds().is_trivial
        assert hash(FeatureBounds()) == hash(FeatureBounds())

    def test_derive_bounds(self):
        a, b, c = Literal(0, "<=", 5.0), Literal(1, "==", 2.0), Literal(0, ">", 1.0)
        constraints = [
            Constraint(MofNTest(1, (a,)), True),  # x0 <= 5
            Constraint(MofNTest(2, (b, c)), True),  # 2-of-2 => both hold
            Constraint(MofNTest(1, (Literal(2, "==", 0.0), Literal(1, "==", 3.0))), False),  # 1-of-2 fails => both fail
            Constraint(MofNTest(2, (Literal(2, "<=", 1.0), Literal(0, "<=", 9.0), Literal(1, "==", 7.0))), True),
        ]
        bounds = derive_bounds(constraints, 3)
        assert (bounds[0].lower, bounds[0].upper) == (1.0, 5.0)
        assert bounds[1].allowed == {2.0} and bounds[1].excluded == {3.0}
        assert bounds[2].excluded == {0.0} and bounds[2].upper == np.inf  # the compound constraint is not decomposed


class TestNominalDistribution:
    def test_frequencies_and_sampling(self, rng):
        dist = NominalDistribution.fit(np.array([0, 0, 0, 1, 2, 2], dtype=float))
        np.testing.assert_array_equal(dist.values, [0, 1, 2])
        np.testing.assert_allclose(dist.probabilities, [0.5, 1 / 6, 1 / 3])
        samples = dist.sample(rng, 5000)
        assert set(np.unique(samples)) <= {0.0, 1.0, 2.0}
        assert abs(np.mean(samples == 0) - 0.5) < 0.03

    def test_bounded_sampling_and_probabilities(self, rng):
        dist = NominalDistribution.fit(np.array([0, 0, 1, 2, 2, 2], dtype=float))
        bounds = FeatureBounds(allowed=frozenset({1.0, 2.0}))
        assert set(dist.sample(rng, 100, bounds)) <= {1.0, 2.0}
        assert dist.prob(Literal(0, "==", 2.0), bounds) == pytest.approx(0.75)
        assert dist.prob(Literal(0, "!=", 2.0), bounds) == pytest.approx(0.25)
        assert dist.prob(Literal(0, "==", 0.0), bounds) == 0.0
        assert dist.prob(Literal(0, "==", 0.0)) == pytest.approx(1 / 3)
        assert len(dist.sample(rng, 10, FeatureBounds(allowed=frozenset({7.0})))) == 0

    def test_difference_test(self):
        dist = NominalDistribution.fit(np.repeat([0, 1], 200).astype(float))
        assert not dist.differs_from(np.repeat([0, 1], 50).astype(float), 0.05)
        assert dist.differs_from(np.zeros(100), 0.05)
        assert not dist.differs_from(np.zeros(1), 0.05)


class TestKernelDensityDistribution:
    def test_bandwidth_rules(self):
        column = np.random.default_rng(1).normal(size=400)
        craven = KernelDensityDistribution.select_bandwidth(column, "craven", scale=10.0)
        assert craven == pytest.approx(10.0 / 20.0)
        assert KernelDensityDistribution.select_bandwidth(column, "craven") == pytest.approx(
            (column.max() - column.min()) / 20
        )
        silverman = KernelDensityDistribution.select_bandwidth(column, "silverman")
        scott = KernelDensityDistribution.select_bandwidth(column, "scott")
        assert 0 < silverman < scott < 1
        assert KernelDensityDistribution.select_bandwidth(column, 0.3) == 0.3
        assert KernelDensityDistribution.select_bandwidth(np.array([1.0]), "silverman") == 0.0
        with pytest.raises(ValueError):
            KernelDensityDistribution.select_bandwidth(column, "magic")

    def test_support_and_integer_rounding(self, rng):
        column = np.array([10.0, 12.0, 15.0, 20.0, 30.0])
        dist = KernelDensityDistribution.fit(column, bandwidth=5.0, support=(10.0, 30.0))
        samples = dist.sample(rng, 2000)
        assert dist.integer_valued
        assert samples.min() >= 10.0 and samples.max() <= 30.0
        assert np.all(np.mod(samples, 1) == 0)
        free = KernelDensityDistribution.fit(column + 0.5, bandwidth=5.0)
        assert not free.integer_valued and free.sample(rng, 2000).min() < 10.5

    def test_bounded_sampling_and_probabilities(self, rng):
        dist = KernelDensityDistribution.fit(np.linspace(0, 10, 500), bandwidth=0.2, support=(0.0, 10.0))
        bounds = FeatureBounds(lower=2.0, upper=3.0)
        samples = dist.sample(rng, 500, bounds)
        assert len(samples) == 500 and np.all((samples > 2.0) & (samples <= 3.0))
        assert dist.prob(Literal(0, "<=", 5.0)) == pytest.approx(0.5, abs=0.02)
        assert dist.prob(Literal(0, "<=", 2.5), bounds) == pytest.approx(0.5, abs=0.05)
        assert dist.prob(Literal(0, ">", 2.5), bounds) == pytest.approx(0.5, abs=0.05)
        assert dist.prob(Literal(0, "<=", 1.0), bounds) == 0.0
        assert dist.prob(Literal(0, "<=", 4.0), bounds) == 1.0
        assert dist.prob(Literal(0, "<=", 5.0), FeatureBounds(lower=50.0)) == 0.0  # empty interval
        assert len(dist.sample(rng, 10, FeatureBounds(lower=50.0))) == 0
        assert dist.cdf(-1) == pytest.approx(0.0, abs=1e-6) and dist.cdf(11) == pytest.approx(1.0, abs=1e-6)

    def test_integer_probabilities(self):
        dist = KernelDensityDistribution.fit(np.arange(0, 11, dtype=float), bandwidth=0.0)
        assert dist.integer_valued
        assert dist.prob(Literal(0, "<=", 4.0)) == pytest.approx(5 / 11)
        assert dist.prob(Literal(0, "<=", 4.5)) == pytest.approx(5 / 11)

    def test_difference_test(self):
        base = np.random.default_rng(2).normal(size=500)
        dist = KernelDensityDistribution.fit(base)
        assert not dist.differs_from(base[:100], 0.05)
        assert dist.differs_from(base[:100] + 3.0, 0.05)


class TestFeatureDistributions:
    def test_fit_and_sample_shapes(self, rng):
        X = np.column_stack([np.random.default_rng(3).normal(size=300), np.repeat([0, 1, 2], 100)])
        dists = FeatureDistributions.fit(X, categorical=[1])
        assert isinstance(dists.distributions[0], KernelDensityDistribution)
        assert isinstance(dists.distributions[1], NominalDistribution)
        samples = dists.sample(rng, 50)
        assert samples.shape == (50, 2)
        assert set(np.unique(samples[:, 1])) <= {0.0, 1.0, 2.0}
        assert dists.n_fit == 300

    def test_local_model_decision_is_node_level(self):
        g = np.random.default_rng(4)
        X_parent = np.column_stack([g.normal(size=400), g.normal(size=400), g.integers(0, 3, 400)])
        parent = FeatureDistributions.fit(X_parent, categorical=[2])
        same = FeatureDistributions.fit_local(X_parent[:150].copy(), parent, alpha=0.10)
        assert same is parent  # nothing differs: inherit the ancestor model

        shifted = X_parent[:150].copy()
        shifted[:, 1] += 5.0
        local = FeatureDistributions.fit_local(shifted, parent, alpha=0.10, depth=1)
        assert local is not parent and local.n_fit == 150 and local.depth == 1
        assert np.all(local.distributions[1].data > 2.0)  # every feature is refitted from the node's examples

        # A difference in a constrained feature does not count.
        assert FeatureDistributions.fit_local(shifted, parent, constrained_features={1}, alpha=0.10) is parent
        # Too few examples always inherit.
        assert FeatureDistributions.fit_local(shifted[:3], parent, alpha=0.10, min_examples=5) is parent


class TestDrawInstances:
    def test_decomposable_constraints(self, rng):
        g = np.random.default_rng(5)
        X = np.column_stack([g.uniform(0, 10, 500), g.uniform(0, 10, 500), g.integers(0, 3, 500)])
        dists = FeatureDistributions.fit(X, categorical=[2])
        constraints = [
            Constraint(MofNTest(1, (Literal(0, ">", 2.0),)), True),
            Constraint(MofNTest(1, (Literal(2, "==", 1.0),)), False),
            Constraint(MofNTest(2, (Literal(1, "<=", 6.0), Literal(0, "<=", 8.0))), True),
        ]
        samples = draw_instances(dists, constraints, 300, rng)
        assert samples.shape == (300, 3)
        assert np.all((samples[:, 0] > 2.0) & (samples[:, 0] <= 8.0) & (samples[:, 1] <= 6.0) & (samples[:, 2] != 1.0))

    def test_compound_constraints(self, rng):
        g = np.random.default_rng(6)
        X = np.column_stack([g.uniform(0, 10, 500), g.uniform(0, 10, 500), g.integers(0, 2, 500)])
        dists = FeatureDistributions.fit(X, categorical=[2])
        disjunction = MofNTest(1, (Literal(0, "<=", 1.0), Literal(2, "==", 1.0)))
        majority = MofNTest(2, (Literal(0, "<=", 5.0), Literal(1, "<=", 5.0), Literal(2, "==", 0.0)))
        for constraints in (
            [Constraint(disjunction, True)],
            [Constraint(majority, False)],
            [Constraint(majority, True)],
        ):
            samples = draw_instances(dists, constraints, 400, rng)
            assert samples.shape == (400, 3)
            assert np.all(constraints[0].evaluate(samples))
        # Both outcomes of a disjunction remain reachable and reasonably distributed.
        satisfied = draw_instances(dists, [Constraint(disjunction, True)], 1000, rng)
        assert 0.1 < np.mean(satisfied[:, 2] == 1.0) < 1.0 and np.any(satisfied[:, 0] <= 1.0)

    def test_no_constraints(self, rng):
        dists = FeatureDistributions.fit(np.random.default_rng(7).normal(size=(100, 2)))
        assert draw_instances(dists, [], 10, rng).shape == (10, 2)
        assert draw_instances(dists, [], 0, rng).shape == (0, 2)

    def test_infeasible_region_returns_empty_with_warning(self, rng, caplog):
        dists = FeatureDistributions.fit(np.linspace(0, 1, 100).reshape(-1, 1), supports=[(0.0, 1.0)])
        constraint = Constraint(MofNTest(1, (Literal(0, ">", 5.0),)), True)
        with caplog.at_level(logging.WARNING, logger="trepan.sampling"):
            samples = draw_instances(dists, [constraint], 10, rng)
        assert samples.shape == (0, 1)
        assert "DrawSample" in caplog.text
