"""End-to-end tests of the Trepan estimator."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from trepan import Oracle, Trepan
from trepan.stopping import required_sample_size


@pytest.fixture(scope="module")
def iris_oracle():
    iris = load_iris()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(iris.data, iris.target)
    return iris, model


@pytest.fixture(scope="module")
def fitted(iris_oracle):
    iris, model = iris_oracle
    tree = Trepan(model, min_sample=300, max_internal_nodes=6, random_state=0).fit(iris.data)
    return iris, model, tree


def test_fidelity_and_shapes(fitted):
    iris, model, tree = fitted
    predictions = tree.predict(iris.data)
    assert predictions.shape == (len(iris.data),)
    assert tree.fidelity(iris.data) >= 0.9
    assert tree.score(iris.data, iris.target) >= 0.85
    assert set(np.unique(predictions)) <= set(tree.classes_)
    assert 1 <= tree.n_internal_nodes_ <= 6
    assert tree.n_leaves_ == tree.n_internal_nodes_ + 1  # binary tree
    assert tree.depth_ >= 1
    assert tree.n_feature_references_ >= tree.n_internal_nodes_


def test_apply_returns_leaf_ids(fitted):
    iris, _, tree = fitted
    leaf_ids = set(tree.apply(iris.data))
    assert leaf_ids <= {leaf.node_id for leaf in tree.root_.leaves()}


def test_oracle_bookkeeping(fitted):
    iris, _, tree = fitted
    assert isinstance(tree.oracle_, Oracle)
    assert tree.n_generated_instances_ > 0
    assert tree.n_oracle_queries_ >= len(iris.data) + tree.n_generated_instances_


def test_every_node_rests_on_at_least_min_sample_instances(fitted):
    _, _, tree = fitted
    for node in tree.nodes_:
        assert node.n_instances >= 300
        assert node.n_examples + node.n_queries == node.n_instances


def test_strict_rule_leaves_are_unanimous_or_size_limited(fitted):
    _, _, tree = fitted
    m_l = required_sample_size(0.05, 0.05)
    for leaf in tree.root_.leaves():
        if leaf.fidelity == 1.0:
            assert leaf.n_instances >= m_l


def test_reach_estimates_are_consistent(fitted):
    _, _, tree = fitted
    assert tree.root_.reach == 1.0
    for node in tree.root_.internal_nodes():
        assert sum(child.reach for child in node.children) == pytest.approx(node.reach)
    for node in tree.nodes_:
        assert 0.0 <= node.fidelity <= 1.0
        assert node.priority == pytest.approx(node.reach * (1 - node.fidelity))


def test_feature_not_reused_in_two_compound_splits_on_a_path(fitted):
    _, _, tree = fitted
    for leaf in tree.root_.leaves():
        used = [f for c in leaf.constraints if c.test.is_compound for f in c.test.features]
        assert len(used) == len(set(used))


def test_exports(fitted):
    iris, _, tree = fitted
    text = tree.export_text(class_names=iris.target_names)
    assert text.startswith("if ") and "class: " in text and "setosa" in text
    dot = tree.export_dot(class_names=iris.target_names, title="iris")
    assert dot.startswith("digraph TREPAN") and dot.rstrip().endswith("}")
    assert '[label="yes"]' in dot and '[label="no"]' in dot and "iris" in dot
    assert "digraph" in tree.to_graphviz(class_names=iris.target_names).source


def test_reproducible_with_random_state(iris_oracle):
    iris, model = iris_oracle
    a = Trepan(model, min_sample=200, max_internal_nodes=4, random_state=123).fit(iris.data)
    b = Trepan(model, min_sample=200, max_internal_nodes=4, random_state=123).fit(iris.data)
    assert a.export_text() == b.export_text()
    assert a.n_generated_instances_ == b.n_generated_instances_


def test_size_limit_zero_gives_single_leaf(iris_oracle):
    iris, model = iris_oracle
    tree = Trepan(model, min_sample=100, max_internal_nodes=0, random_state=0).fit(iris.data)
    assert tree.n_internal_nodes_ == 0 and tree.n_leaves_ == 1
    assert len(set(tree.predict(iris.data))) == 1
    assert tree.export_text().startswith("class: ")


def test_dataframe_input_with_named_categorical_feature():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"colour": rng.integers(0, 4, n), "size": rng.uniform(0, 10, n)})

    def oracle(X):  # the concept depends only on the nominal feature
        return (np.asarray(X)[:, 0] == 2).astype(int)

    tree = Trepan(oracle, categorical_features=["colour"], min_sample=200, max_internal_nodes=3, random_state=0)
    tree.fit(df, y=None)
    assert tree.feature_names_ == ["colour", "size"]
    assert tree.fidelity(df) == 1.0
    assert tree.root_.split.literals[0].feature == 0 and tree.root_.split.literals[0].is_nominal
    assert "colour" in tree.export_text()
    assert tree.n_internal_nodes_ == 1  # one equality test suffices


def test_m_of_n_concept_is_recovered_end_to_end():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, size=(400, 3)).astype(float)

    def oracle(X):
        return (np.asarray(X).sum(axis=1) >= 2).astype(int)  # 2-of-3 majority

    tree = Trepan(oracle, categorical_features=[0, 1, 2], min_sample=500, max_internal_nodes=5, random_state=0)
    tree.fit(X)
    assert tree.root_.split.is_compound and tree.root_.split.m == 2 and tree.root_.split.n == 3
    assert tree.n_internal_nodes_ == 1
    assert tree.fidelity(X) == 1.0


def test_interval_rule_and_other_options(iris_oracle):
    iris, model = iris_oracle
    tree = Trepan(
        model,
        min_sample=200,
        max_internal_nodes=4,
        stopping_rule="interval",
        interval_method="exact",
        criterion="gain_ratio",
        beam_width=1,
        max_literals=2,
        split_significance=None,
        max_thresholds_per_feature=16,
        kde_bandwidth="scott",
        truncate_to_training_range=False,
        prune_literals=False,
        prune=False,
        random_state=0,
    ).fit(iris.data)
    assert tree.fidelity(iris.data) > 0.8
    assert all(node.split.n <= 2 for node in tree.root_.internal_nodes())


def test_validation_selection(iris_oracle):
    iris, model = iris_oracle
    tree = Trepan(model, min_sample=200, max_internal_nodes=8, validation_fraction=0.2, random_state=0).fit(iris.data)
    curve = tree.validation_fidelity_curve_
    assert curve is not None and curve[0][0] == 0 and len(curve) == len(tree.expansion_order_) + 1
    best = max(f for _, f in curve)
    assert tree.n_internal_nodes_ <= len(tree.expansion_order_)
    assert all(f <= best for _, f in curve)
    explicit = Trepan(model, min_sample=200, max_internal_nodes=8, random_state=0).fit(
        iris.data[:120], X_validation=iris.data[120:]
    )
    assert explicit.validation_fidelity_curve_ is not None
    with pytest.raises(ValueError):
        Trepan(model, min_sample=100).fit(iris.data, X_validation=iris.data[:, :2])


def test_pruning_never_changes_predictions(iris_oracle):
    iris, model = iris_oracle
    kept = Trepan(model, min_sample=200, max_internal_nodes=8, prune=False, random_state=7).fit(iris.data)
    pruned = Trepan(model, min_sample=200, max_internal_nodes=8, prune=True, random_state=7).fit(iris.data)
    np.testing.assert_array_equal(kept.predict(iris.data), pruned.predict(iris.data))
    assert pruned.n_internal_nodes_ <= kept.n_internal_nodes_
    assert pruned.n_pruned_nodes_ == kept.n_internal_nodes_ - pruned.n_internal_nodes_


def test_errors():
    iris = load_iris()
    model = DecisionTreeClassifier(max_depth=2).fit(iris.data, iris.target)
    with pytest.raises(RuntimeError):
        Trepan(model).predict(iris.data)
    for bad in (
        dict(epsilon=1.5),
        dict(beam_width=0),
        dict(criterion="gini"),
        dict(stopping_rule="never"),
        dict(validation_fraction=1.0),
        dict(categorical_features=["nope"]),
    ):
        with pytest.raises(ValueError):
            Trepan(model, **bad).fit(iris.data)
    with pytest.raises(TypeError):
        Trepan(model).fit(np.array([["a", "b"], ["c", "d"]]))
    with pytest.raises(TypeError):
        Oracle(object())
    tree = Trepan(model, min_sample=50, max_internal_nodes=1, random_state=0).fit(iris.data)
    with pytest.raises(ValueError):
        tree.predict(iris.data[:, :2])


def test_oracle_passes_dataframes_when_fitted_on_one():
    seen = []

    def model(X):
        seen.append(type(X).__name__)
        return np.zeros(len(X), dtype=int)

    df = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0) % 3})
    Trepan(model, min_sample=20, max_internal_nodes=1, random_state=0).fit(df)
    assert set(seen) == {"DataFrame"}
    Trepan(model, min_sample=20, max_internal_nodes=1, random_state=0).fit(df.to_numpy())
    assert "ndarray" in seen


def test_oracle_batches_and_counts():
    calls = []

    def model(X):
        calls.append(len(X))
        return np.zeros(len(X), dtype=int)

    oracle = Oracle(model, batch_size=4)
    labels = oracle.predict(np.zeros((10, 2)))
    assert labels.shape == (10,) and calls == [4, 4, 2]
    assert oracle.n_queries == 10 and oracle.n_calls == 1
    assert oracle.predict(np.zeros((0, 2))).shape == (0,)
    with pytest.raises(ValueError):
        Oracle(lambda X: np.zeros(3)).predict(np.zeros((2, 2)))
