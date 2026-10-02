import numpy as np
import pytest

from trepan.splits import (
    Constraint,
    Literal,
    LiteralBank,
    MofNTest,
    candidate_literals,
    entropy,
    gain_ratio,
    implies,
    information_gain,
    partition_chi2_pvalues,
    satisfies_all,
)


@pytest.fixture
def X():
    return np.array(
        [
            [0.0, 10.0, 1.0],
            [1.0, 20.0, 2.0],
            [2.0, 30.0, 1.0],
            [3.0, 40.0, 3.0],
        ]
    )


class TestLiteral:
    def test_continuous_evaluation_and_negation(self, X):
        lit = Literal(0, "<=", 1.5)
        np.testing.assert_array_equal(lit.evaluate(X), [True, True, False, False])
        np.testing.assert_array_equal(lit.negate().evaluate(X), ~lit.evaluate(X))
        assert lit.negate().op == ">"
        assert lit.negate().negate() == lit

    def test_nominal_evaluation(self, X):
        lit = Literal(2, "==", 1)
        np.testing.assert_array_equal(lit.evaluate(X), [True, False, True, False])
        assert lit.is_nominal and not Literal(0, "<=", 1).is_nominal
        np.testing.assert_array_equal(lit.negate().evaluate(X), [False, True, False, True])

    def test_invalid_operator(self):
        with pytest.raises(ValueError):
            Literal(0, "<", 1.0)

    def test_describe(self):
        assert Literal(1, "<=", 2.5).describe(["a", "b"]) == "b <= 2.5"
        assert Literal(0, "==", 3.0).describe() == "x0 == 3"

    def test_implication(self):
        assert implies(Literal(0, "<=", 1), Literal(0, "<=", 2))
        assert not implies(Literal(0, "<=", 2), Literal(0, "<=", 1))
        assert implies(Literal(0, ">", 2), Literal(0, ">", 1))
        assert implies(Literal(0, "==", 1), Literal(0, "!=", 2))
        assert not implies(Literal(0, "==", 1), Literal(0, "!=", 1))  # that is the complement, not an implication
        assert not implies(Literal(0, "<=", 1), Literal(1, "<=", 2))
        assert not implies(Literal(0, "<=", 1), Literal(0, ">", 0))


class TestMofNTest:
    def test_semantics_of_two_of_three(self, X):
        test = MofNTest(2, (Literal(0, "<=", 1.5), Literal(1, ">", 15), Literal(2, "==", 1)))
        # row 0: T F T -> 2 ; row 1: T T F -> 2 ; row 2: F T T -> 2 ; row 3: F T F -> 1
        np.testing.assert_array_equal(test.literal_counts(X), [2, 2, 2, 1])
        np.testing.assert_array_equal(test.evaluate(X), [True, True, True, False])
        assert test.n == 3 and test.is_compound and test.features == {0, 1, 2}

    def test_single_literal_is_not_compound(self):
        assert not MofNTest(1, (Literal(0, "<=", 1),)).is_compound

    def test_literal_order_is_canonical(self):
        a, b = Literal(0, "<=", 1), Literal(1, "<=", 2)
        assert MofNTest(1, (a, b)) == MofNTest(1, (b, a))
        assert hash(MofNTest(1, (a, b))) == hash(MofNTest(1, (b, a)))

    def test_operators(self):
        base = MofNTest(1, (Literal(0, "<=", 1),))
        hold = base.with_literal(Literal(1, "<=", 2), increment_m=False)
        inc = base.with_literal(Literal(1, "<=", 2), increment_m=True)
        assert (hold.m, hold.n) == (1, 2)
        assert (inc.m, inc.n) == (2, 2)
        assert base.with_literal(Literal(0, "<=", 1), increment_m=False) is None  # already present

    def test_complement_simplification(self):
        a, b, c = Literal(0, "<=", 1), Literal(1, "<=", 2), Literal(2, "==", 3)
        test = MofNTest(2, (a, b, c))
        simplified = test.with_literal(c.negate(), increment_m=False)
        assert simplified == MofNTest(1, (a, b))  # 2-of-{a,b,c,not c} == 1-of-{a,b}
        assert MofNTest(2, (a, b, c)).with_literal(c.negate(), increment_m=True) == MofNTest(2, (a, b))
        assert MofNTest(1, (a,)).with_literal(a.negate(), increment_m=False) is None  # always true
        assert MofNTest(1, (a,)).with_literal(a.negate(), increment_m=True) is None

    def test_without_literal(self):
        a, b = Literal(0, "<=", 1), Literal(1, "<=", 2)
        test = MofNTest(2, (a, b))
        assert test.without_literal(b, decrement_m=True) == MofNTest(1, (a,))
        assert test.without_literal(b, decrement_m=False) is None  # 2-of-1 is impossible
        assert MofNTest(1, (a,)).without_literal(a, decrement_m=False) is None

    @pytest.mark.parametrize("m", [0, 3])
    def test_invalid_threshold(self, m):
        with pytest.raises(ValueError):
            MofNTest(m, (Literal(0, "<=", 1), Literal(1, "<=", 2)))

    def test_duplicate_or_contradictory_literals(self):
        lit = Literal(0, "<=", 1)
        with pytest.raises(ValueError):
            MofNTest(1, (lit, lit))
        with pytest.raises(ValueError):
            MofNTest(1, (lit, lit.negate()))

    def test_describe(self):
        test = MofNTest(2, (Literal(0, "<=", 1), Literal(1, "==", 3)))
        assert test.describe(["a", "b"]) == "2 of {a <= 1, b == 3}"
        assert MofNTest(1, (Literal(0, "<=", 1),)).describe() == "x0 <= 1"


class TestConstraint:
    def test_outcome(self, X):
        test = MofNTest(1, (Literal(0, "<=", 1.5),))
        yes, no = Constraint(test, True), Constraint(test, False)
        np.testing.assert_array_equal(yes.evaluate(X), [True, True, False, False])
        np.testing.assert_array_equal(no.evaluate(X), [False, False, True, True])
        assert no.describe() == "not x0 <= 1.5"
        mask = satisfies_all([yes, Constraint(MofNTest(1, (Literal(2, "==", 1),)), True)], X)
        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_forced_literals(self):
        a, b = Literal(0, "<=", 1), Literal(1, "==", 2)
        assert Constraint(MofNTest(1, (a,)), True).forced_literals == (a,)
        assert Constraint(MofNTest(1, (a,)), False).forced_literals == (a.negate(),)
        assert set(Constraint(MofNTest(2, (a, b)), True).forced_literals) == {a, b}
        assert set(Constraint(MofNTest(1, (a, b)), False).forced_literals) == {a.negate(), b.negate()}
        assert Constraint(MofNTest(1, (a, b)), True).forced_literals is None
        assert Constraint(MofNTest(2, (a, b)), False).forced_literals is None


class TestInformationMeasures:
    def test_entropy(self):
        assert entropy(np.array([5, 5])) == pytest.approx(1.0)
        assert entropy(np.array([10, 0])) == pytest.approx(0.0)
        assert entropy(np.array([0, 0])) == pytest.approx(0.0)
        rows = entropy(np.array([[5, 5], [1, 3]]))
        assert rows.shape == (2,)
        assert rows[1] == pytest.approx(0.8112781244591328)

    def test_information_gain(self):
        assert information_gain(np.array([5, 0]), np.array([0, 5])) == pytest.approx(1.0)
        assert information_gain(np.array([5, 5]), np.array([0, 0])) == pytest.approx(0.0)
        assert information_gain(np.array([3, 3]), np.array([3, 3])) == pytest.approx(0.0)
        left, right = np.array([6, 2]), np.array([0, 4])
        expected = entropy(left + right) - (8 / 12 * entropy(left) + 4 / 12 * entropy(right))
        assert information_gain(left, right) == pytest.approx(expected)
        np.testing.assert_allclose(information_gain(np.array([[5, 0], [3, 3]]), np.array([[0, 5], [2, 2]])), [1.0, 0.0])

    def test_gain_ratio(self):
        assert gain_ratio(np.array([5, 0]), np.array([0, 5])) == pytest.approx(1.0)
        assert gain_ratio(np.array([5, 5]), np.array([0, 0])) == pytest.approx(0.0)
        left, right = np.array([6, 2]), np.array([0, 4])
        assert gain_ratio(left, right) == pytest.approx(information_gain(left, right) / entropy(np.array([8, 4])))

    def test_partition_chi2(self):
        reference = np.array([50, 50])
        candidates = np.array([[50, 50], [52, 48], [90, 10], [0, 0]])
        p = partition_chi2_pvalues(reference, candidates)
        assert p[0] == pytest.approx(1.0)
        assert p[1] > 0.5
        assert p[2] < 1e-6
        assert p[3] == 1.0  # empty candidate sample: nothing to compare
        assert partition_chi2_pvalues(np.array([30, 10]), np.array([[300, 100]]))[0] > 0.9  # same proportions


class TestCandidateLiterals:
    def test_continuous_midpoints_and_constant_feature(self):
        X = np.array([[1.0, 7.0], [3.0, 7.0], [2.0, 7.0]])
        assert candidate_literals(X) == [Literal(0, "<=", 1.5), Literal(0, "<=", 2.5)]

    def test_boundary_points_only(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        assert candidate_literals(X, np.array([0, 0, 1, 1])) == [Literal(0, "<=", 2.5)]
        assert len(candidate_literals(X, np.array([0, 1, 0, 1]))) == 3
        # A mixed group next to a pure group still yields a boundary.
        X_ties = np.array([[1.0], [1.0], [2.0], [2.0]])
        assert candidate_literals(X_ties, np.array([0, 1, 0, 0])) == [Literal(0, "<=", 1.5)]
        assert candidate_literals(X_ties, np.array([0, 0, 0, 0])) == []

    def test_nominal_features(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [1.0, 2.0]])
        lits = candidate_literals(X, categorical=[0, 1])
        assert Literal(0, "==", 0) in lits and Literal(0, "==", 1) not in lits  # two-valued: one split
        assert {lit for lit in lits if lit.feature == 1} == {Literal(1, "==", v) for v in (0, 1, 2)}

    def test_threshold_cap(self):
        X = np.linspace(0, 1, 200).reshape(-1, 1)
        lits = candidate_literals(X, max_thresholds=10)
        assert 1 <= len(lits) <= 10
        assert all(lit.value < 1.0 for lit in lits)


class TestLiteralBank:
    def test_rows_and_scores(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        bank = LiteralBank(candidate_literals(X), X)
        assert len(bank) == 2 * bank.n_positive == 6
        lit = Literal(0, "<=", 1.5)
        np.testing.assert_array_equal(bank.row(lit), lit.evaluate(X))
        np.testing.assert_array_equal(bank.row(lit.negate()), ~lit.evaluate(X))
        scores = bank.score_binary_splits(np.eye(2)[y])
        assert scores[np.argmax(scores)] == pytest.approx(1.0)
        assert bank.literals[int(np.argmax(scores))] == lit
        ratios = bank.score_binary_splits(np.eye(2)[y], gain_ratio)
        assert ratios[np.argmax(ratios)] == pytest.approx(1.0)

    def test_eligible_mask(self):
        X = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 2.0, 3.0], [3.0, 0.0, 4.0]])
        bank = LiteralBank(candidate_literals(X, categorical=[1]), X)
        test = MofNTest(1, (Literal(0, "<=", 1.5),))
        eligible = {lit for lit, ok in zip(bank.literals, bank.eligible_mask(test, excluded_features=[2])) if ok}
        assert Literal(0, "<=", 1.5) not in eligible  # already present
        assert Literal(0, "<=", 2.5) not in eligible and Literal(0, "<=", 0.5) not in eligible  # implied / implying
        assert Literal(0, ">", 1.5) in eligible  # the complement is allowed (it simplifies the test)
        assert Literal(0, ">", 2.5) in eligible  # different threshold, opposite direction: not implied
        assert all(lit.feature != 2 for lit in eligible)  # excluded feature
        assert Literal(1, "==", 0) in eligible  # nominal literals remain available

        nominal_test = MofNTest(1, (Literal(1, "==", 0),))
        eligible = {lit for lit, ok in zip(bank.literals, bank.eligible_mask(nominal_test)) if ok}
        assert Literal(1, "==", 1) in eligible  # another value of the same feature
        assert Literal(1, "!=", 1) not in eligible  # implied by colour == 0
        assert Literal(1, "!=", 0) in eligible  # the complement
