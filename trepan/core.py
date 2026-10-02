"""The TREPAN algorithm (Craven & Shavlik, 1995; Craven, 1996).

TREPAN extracts a decision tree that approximates the function computed by a
trained classifier -- the *oracle* -- by treating extraction as an inductive
learning problem in which the oracle can be queried.  The outline of the
algorithm, after Figure 9 of the thesis:

    Trepan(Oracle, training set S, features F, min_sample, stopping criteria)
        for each example x in S: class label for x := Oracle(x)
        initialise the root R as a leaf; build an instance model M for R
        query_instances_R := DrawSample({}, min_sample - |S|, M)
        use S and query_instances_R to determine the class label of R
        Queue := {<R, S, query_instances_R, {}>}
        while Queue not empty and global stopping criteria not satisfied:
            remove <N, S_N, query_instances_N, constraints_N> from the head of Queue
            T := ConstructTest(F, S_N + query_instances_N)
            make N an internal node with test T
            for each outcome t of T:
                make C a new child of N; constraints_C := constraints_N + {T = t}
                S_C := members of S_N with outcome t on T
                build an instance model M for C
                query_instances_C := DrawSample(constraints_C, min_sample - |S_C|, M)
                use S_C and query_instances_C to determine the class label of C
                if local stopping criteria not satisfied: put C into Queue
        return the tree rooted at R

The queue is ordered by ``f(N) = reach(N) * (1 - fidelity(N))`` (best-first
expansion), every decision at a node rests on at least ``min_sample``
instances (training examples plus query instances), a node becomes a leaf when
it covers instances of a single class with high probability, and the tree
stops growing at a user-specified number of internal nodes.  Optionally a
validation set selects the best tree from the nested sequence produced by the
best-first expansion, and subtrees whose leaves all predict one class are
collapsed before the tree is returned.
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .oracle import Oracle
from .sampling import FeatureDistributions, draw_instances
from .search import best_binary_split, search_m_of_n
from .splits import Constraint, LiteralBank, MofNTest, candidate_literals, gain_ratio, information_gain
from .stopping import cannot_become_pure, is_pure_enough, required_sample_size
from .tree import Node, export_dot, export_text, prune_redundant_subtrees

__all__ = ["Trepan"]

logger = logging.getLogger(__name__)

_CRITERIA = {"gain": information_gain, "gain_ratio": gain_ratio}


class Trepan:
    """Extract an m-of-n decision tree from a trained classifier.

    The estimator follows the scikit-learn conventions: hyper-parameters are
    given to the constructor, :meth:`fit` builds the tree and attributes
    learned from data carry a trailing underscore.

    Args:
        oracle: The trained model to explain -- anything accepted by
            :class:`~trepan.oracle.Oracle` (an object with ``predict(X)`` or a
            callable), or an :class:`~trepan.oracle.Oracle` instance.
        categorical_features: Indices (or column names, when ``X`` is a pandas
            DataFrame) of the nominal features.  They must be numerically
            encoded; all other features are treated as continuous.
        max_internal_nodes: Global stopping criterion -- the maximum number of
            internal nodes.  The paper used 15 (a complete binary tree of depth
            four), the thesis 31 (depth five).
        min_sample: ``min_sample`` / ``S_min`` -- the minimum number of
            instances (training examples plus query instances) considered
            before labelling a node or choosing its split.  The paper used
            1000, the thesis 10,000.
        epsilon: Purity tolerance of the local stopping criterion.
        delta: Confidence parameter of the local stopping criterion: a node is a
            leaf when ``prob(p_c < 1 - epsilon) < delta``.  The paper used 0.05
            for both, the thesis ``delta = 0.01``.
        stopping_rule: ``"strict"`` (thesis): a leaf must have at least
            ``m_L = z_delta^2 (1 - epsilon) / epsilon`` instances that *all*
            share one class, and more are queried while they keep agreeing.
            ``"interval"`` (paper, read literally): the one-sided lower
            confidence bound on ``p_c`` must reach ``1 - epsilon``, which also
            admits a few disagreeing instances in large samples.
        interval_method: Interval used by the ``"interval"`` rule: ``"wilson"``,
            ``"exact"`` (Clopper-Pearson) or ``"normal"`` (Wald).
        max_queries_per_node: Cap on the query instances drawn for one node;
            defaults to ``2 * min_sample``.
        criterion: ``"gain"`` (information gain, thesis) or ``"gain_ratio"`` (paper).
        beam_width: Beam width of the m-of-n search (thesis: 2; 1 = hill climbing).
        max_literals: Maximum number of literals ``n`` in a split, or ``None``.
        split_significance: Level of the chi-square test an operator application
            must pass to be admissible (thesis: 0.05); ``None`` disables it.
        min_split_improvement: Extra margin a candidate test must have over the
            worst test in the beam (0 reproduces the thesis).
        prune_literals: Run the thesis' literal-pruning pass on each split.
        max_thresholds_per_feature: Cap on candidate thresholds per continuous
            feature (quantiles are used when exceeded).  ``None`` keeps every
            boundary midpoint, as the thesis does.
        distribution_test_alpha: Overall level of the Bonferroni-corrected
            chi-square / Kolmogorov-Smirnov tests that decide whether a node
            fits a local instance model (thesis: 0.10).
        min_local_examples: Nodes with fewer examples always inherit the
            ancestor's instance model.
        kde_bandwidth: Kernel width rule for continuous features: ``"craven"``
            (range / sqrt(m), thesis), ``"silverman"``, ``"scott"`` or a number.
        truncate_to_training_range: Keep sampled continuous values within the
            range observed in the training data.
        validation_fraction: Fraction of the training examples held out to
            pick, from the nested sequence of trees, the one with the highest
            fidelity to the oracle (thesis: 0.10).  ``None`` disables this.
        prune: Collapse subtrees whose leaves all predict the same class
            (thesis, Section 3.2.6).
        random_state: Seed or NumPy ``Generator`` for reproducible sampling.

    Attributes:
        root_: Root :class:`~trepan.tree.Node` of the extracted tree.
        oracle_: The wrapped oracle; ``oracle_.n_queries`` counts every label request.
        feature_names_: Feature names (DataFrame columns or ``x0, x1, ...``).
        classes_: Class labels seen from the oracle.
        n_internal_nodes_, n_leaves_, depth_: Size statistics of the tree.
        n_generated_instances_: Query instances drawn during extraction.
        expansion_order_: Node ids in the order they were expanded (before any pruning).
        validation_fidelity_curve_: ``(n_internal_nodes, fidelity)`` pairs over
            the nested sequence of trees when a validation set is used.
        n_pruned_nodes_: Internal nodes removed by pruning and validation selection.

    Example:
        >>> from sklearn.neural_network import MLPClassifier
        >>> net = MLPClassifier(max_iter=2000).fit(X_train, y_train)   # doctest: +SKIP
        >>> tree = Trepan(net, min_sample=1000, random_state=0).fit(X_train)  # doctest: +SKIP
        >>> print(tree.export_text())                                   # doctest: +SKIP
    """

    def __init__(
        self,
        oracle: Any,
        *,
        categorical_features: Sequence[int | str] | None = None,
        max_internal_nodes: int = 15,
        min_sample: int = 1000,
        epsilon: float = 0.05,
        delta: float = 0.05,
        stopping_rule: str = "strict",
        interval_method: str = "wilson",
        max_queries_per_node: int | None = None,
        criterion: str = "gain",
        beam_width: int = 2,
        max_literals: int | None = None,
        split_significance: float | None = 0.05,
        min_split_improvement: float = 0.0,
        prune_literals: bool = True,
        max_thresholds_per_feature: int | None = None,
        distribution_test_alpha: float = 0.10,
        min_local_examples: int = 5,
        kde_bandwidth: str | float = "craven",
        truncate_to_training_range: bool = True,
        validation_fraction: float | None = None,
        prune: bool = True,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.oracle = oracle
        self.categorical_features = categorical_features
        self.max_internal_nodes = max_internal_nodes
        self.min_sample = min_sample
        self.epsilon = epsilon
        self.delta = delta
        self.stopping_rule = stopping_rule
        self.interval_method = interval_method
        self.max_queries_per_node = max_queries_per_node
        self.criterion = criterion
        self.beam_width = beam_width
        self.max_literals = max_literals
        self.split_significance = split_significance
        self.min_split_improvement = min_split_improvement
        self.prune_literals = prune_literals
        self.max_thresholds_per_feature = max_thresholds_per_feature
        self.distribution_test_alpha = distribution_test_alpha
        self.min_local_examples = min_local_examples
        self.kde_bandwidth = kde_bandwidth
        self.truncate_to_training_range = truncate_to_training_range
        self.validation_fraction = validation_fraction
        self.prune = prune
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any = None, X_validation: Any = None) -> Trepan:
        """Extract the tree from the oracle.

        Args:
            X: Training examples of the oracle, shape ``(n_examples, n_features)``
                (NumPy array or pandas DataFrame).  Only the inputs are needed:
                the tree approximates the oracle, so the labels come from it.
            y: Ignored.  Accepted for compatibility with scikit-learn tooling.
            X_validation: Optional held-out examples used to select the best
                tree from the nested sequence grown by the best-first expansion.
                When omitted and ``validation_fraction`` is set, that fraction
                of ``X`` is held out instead.

        Returns:
            The fitted estimator.
        """
        self._validate_parameters()
        X_arr, feature_names = self._coerce_X(X)
        self.feature_names_ = feature_names
        self.n_features_in_ = X_arr.shape[1]
        self._categorical = self._resolve_categorical(feature_names)
        self._criterion = _CRITERIA[self.criterion]
        self._rng = np.random.default_rng(self.random_state)
        self.oracle_ = self.oracle if isinstance(self.oracle, Oracle) else Oracle(self.oracle)
        if getattr(X, "columns", None) is not None and self.oracle_.feature_names is None:
            self.oracle_.feature_names = list(feature_names)  # the model was probably trained on named columns
        self._max_queries = self.max_queries_per_node if self.max_queries_per_node is not None else 2 * self.min_sample
        self._required_pure = required_sample_size(self.epsilon, self.delta)
        self.n_generated_instances_ = 0
        self.n_pruned_nodes_ = 0
        self._next_node_id = 0

        X_val = None
        if X_validation is not None:
            X_val = self._coerce_X(X_validation)[0]
            if X_val.shape[1] != self.n_features_in_:
                raise ValueError("X_validation must have the same number of features as X")
        elif self.validation_fraction:
            n_val = int(round(self.validation_fraction * len(X_arr)))
            if 0 < n_val < len(X_arr):
                permutation = self._rng.permutation(len(X_arr))
                X_val, X_arr = X_arr[permutation[:n_val]], X_arr[permutation[n_val:]]

        # Instance-model constants: feature ranges in the training data.
        continuous = [j for j in range(self.n_features_in_) if j not in self._categorical]
        self._scales = [
            float(X_arr[:, j].max() - X_arr[:, j].min()) if j in continuous else 0.0 for j in range(self.n_features_in_)
        ]
        self._supports = (
            [
                (float(X_arr[:, j].min()), float(X_arr[:, j].max())) if j in continuous else None
                for j in range(self.n_features_in_)
            ]
            if self.truncate_to_training_range
            else None
        )

        # The oracle, not the data set, labels the training examples.
        y_oracle = self.oracle_.predict(X_arr)
        self._label_dtype = y_oracle.dtype

        root = self._new_node(depth=0, constraints=(), X=X_arr, y=y_oracle, parent=None)
        root.distributions = FeatureDistributions.fit(
            X_arr, self._categorical, bandwidth=self.kde_bandwidth, scales=self._scales, supports=self._supports
        )
        root.reach = 1.0
        self._fill_pool(root)
        root_is_leaf = self._decide_leaf(root)
        self._refresh_estimates(root)

        # Best-first expansion: the queue is ordered by f(N) = reach * (1 - fidelity).
        queue: list[tuple[float, int, Node]] = []
        if not root_is_leaf:
            heapq.heappush(queue, (-root.priority, root.node_id, root))
        self.expansion_order_: list[int] = []

        while queue and len(self.expansion_order_) < self.max_internal_nodes:
            _, _, node = heapq.heappop(queue)
            selection = self._select_split(node)
            if selection is None:
                logger.info("Node %d stays a leaf: no split improves on the oracle labels", node.node_id)
                continue
            split, score = selection
            node.split = split
            node.split_gain = score
            node.expansion_index = len(self.expansion_order_)
            self.expansion_order_.append(node.node_id)
            logger.info(
                "Expanded node %d (depth %d, %d instances, f=%.4f) with '%s' (%s %.3f)",
                node.node_id,
                node.depth,
                node.n_instances,
                node.priority,
                split.describe(self.feature_names_),
                self.criterion,
                score,
            )

            X_pool, _ = node.pool()
            for outcome in (True, False):
                constraint = Constraint(split, outcome)
                train_mask = constraint.evaluate(node.X)
                child = self._new_node(
                    depth=node.depth + 1,
                    constraints=node.constraints + (constraint,),
                    X=node.X[train_mask],
                    y=node.y[train_mask],
                    parent=node,
                )
                child.distributions = FeatureDistributions.fit_local(
                    child.X,
                    node.distributions,
                    child.constrained_features(),
                    alpha=self.distribution_test_alpha,
                    min_examples=self.min_local_examples,
                    bandwidth=self.kde_bandwidth,
                    scales=self._scales,
                    supports=self._supports,
                    depth=child.depth,
                )
                # reach(child) = reach(parent) * fraction of the parent's instances with this outcome
                child.reach = node.reach * float(constraint.evaluate(X_pool).mean())
                node.children.append(child)

                self._fill_pool(child)
                is_leaf = self._decide_leaf(child)
                self._refresh_estimates(child)
                if not is_leaf:
                    heapq.heappush(queue, (-child.priority, child.node_id, child))

        # Nodes still queued when the size limit is hit simply remain leaves.
        self.root_ = root
        self.validation_fidelity_curve_: list[tuple[int, float]] | None = None
        if X_val is not None and len(X_val):
            self._select_by_validation(X_val)
        if self.prune:
            self.n_pruned_nodes_ += prune_redundant_subtrees(root)
        self._finalise()
        return self

    def apply(self, X: Any) -> np.ndarray:
        """Return the id of the leaf each instance of ``X`` ends up in."""
        X_arr = self._check_fitted_X(X)
        return self._route(self.root_, X_arr)

    def predict(self, X: Any) -> np.ndarray:
        """Predict the class of every instance of ``X`` with the extracted tree."""
        leaf_ids = self.apply(X)
        predictions = np.empty(len(leaf_ids), dtype=self._label_dtype)
        for leaf in self.root_.leaves():
            predictions[leaf_ids == leaf.node_id] = leaf.label
        return predictions

    def fidelity(self, X: Any) -> float:
        """Fraction of instances on which the tree agrees with the oracle."""
        X_arr = self._check_fitted_X(X)
        return float(np.mean(self.predict(X_arr) == self.oracle_.predict(X_arr)))

    def score(self, X: Any, y: Any) -> float:
        """Accuracy of the tree against reference labels ``y``."""
        return float(np.mean(self.predict(X) == np.asarray(y).reshape(-1)))

    def export_text(self, class_names: Sequence[str] | None = None, show_counts: bool = True) -> str:
        """Render the tree as indented pseudo-code."""
        self._check_fitted()
        return export_text(self.root_, self.feature_names_, class_names, show_counts)

    def export_dot(
        self, class_names: Sequence[str] | None = None, show_counts: bool = True, title: str | None = None
    ) -> str:
        """Render the tree in Graphviz DOT format."""
        self._check_fitted()
        return export_dot(self.root_, self.feature_names_, class_names, show_counts, title)

    def to_graphviz(self, class_names: Sequence[str] | None = None, **kwargs: Any) -> Any:
        """Return a ``graphviz.Source`` for the tree (requires the ``graphviz`` package)."""
        try:
            import graphviz
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("to_graphviz() requires the optional 'graphviz' package") from exc
        return graphviz.Source(self.export_dot(class_names, **kwargs))

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    @property
    def nodes_(self) -> list[Node]:
        """All nodes in pre-order."""
        self._check_fitted()
        return list(self.root_.iter_nodes())

    @property
    def n_internal_nodes_(self) -> int:
        """Number of internal (splitting) nodes."""
        return len(self.root_.internal_nodes())

    @property
    def n_leaves_(self) -> int:
        """Number of leaves."""
        return len(self.root_.leaves())

    @property
    def depth_(self) -> int:
        """Length of the longest root-to-leaf path."""
        return max(n.depth for n in self.root_.iter_nodes())

    @property
    def n_feature_references_(self) -> int:
        """Syntactic complexity measure of the thesis: literals summed over all splits."""
        return sum(n.split.n for n in self.root_.internal_nodes())

    @property
    def n_oracle_queries_(self) -> int:
        """Every label request made to the oracle, including the training set."""
        return self.oracle_.n_queries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_parameters(self) -> None:
        if self.max_internal_nodes < 0:
            raise ValueError("max_internal_nodes must be >= 0")
        if self.min_sample < 1:
            raise ValueError("min_sample must be >= 1")
        if not 0 < self.epsilon < 1 or not 0 < self.delta < 1:
            raise ValueError("epsilon and delta must lie in (0, 1)")
        if self.stopping_rule not in ("strict", "interval"):
            raise ValueError("stopping_rule must be 'strict' or 'interval'")
        if self.interval_method not in ("wilson", "exact", "normal"):
            raise ValueError("interval_method must be 'wilson', 'exact' or 'normal'")
        if self.max_queries_per_node is not None and self.max_queries_per_node < 0:
            raise ValueError("max_queries_per_node must be >= 0 or None")
        if self.criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {tuple(_CRITERIA)}")
        if self.beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_literals is not None and self.max_literals < 1:
            raise ValueError("max_literals must be >= 1 or None")
        if self.split_significance is not None and not 0 < self.split_significance < 1:
            raise ValueError("split_significance must lie in (0, 1) or be None")
        if self.min_split_improvement < 0:
            raise ValueError("min_split_improvement must be >= 0")
        if self.max_thresholds_per_feature is not None and self.max_thresholds_per_feature < 1:
            raise ValueError("max_thresholds_per_feature must be >= 1 or None")
        if not 0 < self.distribution_test_alpha < 1:
            raise ValueError("distribution_test_alpha must lie in (0, 1)")
        if self.min_local_examples < 2:
            raise ValueError("min_local_examples must be >= 2")
        if self.validation_fraction is not None and not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie in (0, 1) or be None")

    @staticmethod
    def _coerce_X(X: Any) -> tuple[np.ndarray, list[str]]:
        """Convert ``X`` to a float matrix and recover feature names."""
        columns = getattr(X, "columns", None)
        values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        try:
            X_arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "X must be numeric; encode categorical features as numbers (e.g. with OrdinalEncoder) first"
            ) from exc
        if X_arr.ndim != 2:
            raise ValueError(f"X must be two-dimensional, got shape {X_arr.shape}")
        if len(X_arr) == 0:
            raise ValueError("X must contain at least one example")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN or infinite values")
        names = [str(c) for c in columns] if columns is not None else [f"x{j}" for j in range(X_arr.shape[1])]
        return X_arr, names

    def _resolve_categorical(self, feature_names: Sequence[str]) -> set[int]:
        if not self.categorical_features:
            return set()
        resolved = set()
        for item in self.categorical_features:
            if isinstance(item, str):
                if item not in feature_names:
                    raise ValueError(f"Unknown categorical feature name {item!r}")
                resolved.add(feature_names.index(item))
            else:
                j = int(item)
                if not 0 <= j < len(feature_names):
                    raise IndexError(f"Categorical feature index {j} out of range")
                resolved.add(j)
        return resolved

    def _check_fitted(self) -> None:
        if not hasattr(self, "root_"):
            raise RuntimeError("This Trepan instance is not fitted yet; call fit(X) first")

    def _check_fitted_X(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X_arr, _ = self._coerce_X(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features; the tree was fitted with {self.n_features_in_}")
        return X_arr

    def _new_node(self, **kwargs: Any) -> Node:
        node = Node(node_id=self._next_node_id, **kwargs)
        self._next_node_id += 1
        return node

    @staticmethod
    def _route(root: Node, X: np.ndarray, active: set[int] | None = None) -> np.ndarray:
        """Leaf id of every row of ``X``; nodes outside ``active`` are treated as leaves."""
        leaf_ids = np.empty(len(X), dtype=int)
        stack = [(root, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if idx.size == 0:
                continue
            if node.is_leaf or (active is not None and node.node_id not in active):
                leaf_ids[idx] = node.node_id
                continue
            satisfied = node.split.evaluate(X[idx])
            stack.append((node.children[0], idx[satisfied]))
            stack.append((node.children[1], idx[~satisfied]))
        return leaf_ids

    # -- querying -------------------------------------------------------------

    def _query(self, node: Node, n: int) -> int:
        """DrawSample: draw ``n`` instances satisfying the node's constraints and label them."""
        if n <= 0:
            return 0
        X_new = draw_instances(node.distributions, node.constraints, n, self._rng)
        if len(X_new) == 0:
            return 0
        y_new = self.oracle_.predict(X_new)
        node.add_queries(X_new, y_new)
        self.n_generated_instances_ += len(X_new)
        return len(X_new)

    def _fill_pool(self, node: Node) -> None:
        """Make sure at least ``min_sample`` instances are available at the node."""
        needed = min(self.min_sample - node.n_instances, self._max_queries - node.n_queries)
        if needed > 0:
            self._query(node, needed)

    def _decide_leaf(self, node: Node) -> bool:
        """Apply the local stopping criterion, querying more instances as needed.

        Returns ``True`` when the node should remain a leaf.
        """
        while True:
            _, y_pool = node.pool()
            n = len(y_pool)
            if n == 0:
                return True  # nothing reaches here under the model: nothing to learn
            k = int(np.max(np.unique(y_pool, return_counts=True)[1]))
            if self.stopping_rule == "strict":
                if k < n:
                    return False  # a disagreeing instance: the node must be expanded
                if n >= self._required_pure:
                    return True
                wanted = self._required_pure - n
            else:
                if is_pure_enough(k, n, self.epsilon, self.delta, self.interval_method):
                    return True
                if cannot_become_pure(k, n, self.epsilon, self.delta, self.interval_method):
                    return False
                wanted = max(1, self.min_sample // 4)
            room = self._max_queries - node.n_queries
            if room <= 0:
                return False  # undecided within the query budget: let the split search decide
            if self._query(node, min(wanted, room)) == 0:
                return True  # the region cannot be sampled any further

    def _refresh_estimates(self, node: Node) -> None:
        """Set the node's label, class counts and fidelity from its instance pool."""
        _, y_pool = node.pool()
        values, counts = np.unique(y_pool, return_counts=True)
        node.class_counts = {v.item() if hasattr(v, "item") else v: int(c) for v, c in zip(values, counts)}
        best = int(np.argmax(counts))
        node.label = values[best].item() if hasattr(values[best], "item") else values[best]
        node.fidelity = float(counts[best] / counts.sum())

    # -- splitting --------------------------------------------------------------

    def _select_split(self, node: Node) -> tuple[MofNTest, float] | None:
        """ConstructTest: choose the m-of-n split for ``node`` or ``None`` to keep it a leaf."""
        X_pool, y_pool = node.pool()
        classes, y_idx = np.unique(y_pool, return_inverse=True)
        if len(classes) < 2:
            return None
        y_onehot = np.eye(len(classes))[y_idx]
        literals = candidate_literals(X_pool, y_pool, self._categorical, self.max_thresholds_per_feature)
        if not literals:
            return None
        bank = LiteralBank(literals, X_pool)
        seed = best_binary_split(bank, y_onehot, self._criterion)
        if seed is None:
            return None
        return search_m_of_n(
            seed[0],
            seed[1],
            bank,
            y_onehot,
            criterion=self._criterion,
            beam_width=self.beam_width,
            max_literals=self.max_literals,
            significance=self.split_significance,
            min_improvement=self.min_split_improvement,
            excluded_features=node.features_in_compound_splits(),
            prune_literals=self.prune_literals,
        )

    # -- model selection and finishing -----------------------------------------

    def _select_by_validation(self, X_val: np.ndarray) -> None:
        """Keep the prefix of the expansion sequence with the highest validation fidelity."""
        y_val = self.oracle_.predict(X_val)
        nodes = {node.node_id: node for node in self.root_.iter_nodes()}
        labels = {node_id: node.label for node_id, node in nodes.items()}
        curve: list[tuple[int, float]] = []
        for k in range(len(self.expansion_order_) + 1):
            active = set(self.expansion_order_[:k])
            leaf_ids = self._route(self.root_, X_val, active)
            predictions = np.array([labels[i] for i in leaf_ids])
            curve.append((k, float(np.mean(predictions == y_val))))
        best_k = max(range(len(curve)), key=lambda k: (curve[k][1], -k))
        self.validation_fidelity_curve_ = curve
        for node_id in self.expansion_order_[best_k:]:
            node = nodes[node_id]
            if not node.is_leaf:
                self.n_pruned_nodes_ += len(node.internal_nodes())
                node.collapse()
        logger.info(
            "Validation selection kept %d of %d expansions (fidelity %.3f)",
            best_k,
            len(self.expansion_order_),
            curve[best_k][1],
        )

    def _finalise(self) -> None:
        labels = set()
        for node in self.root_.iter_nodes():
            labels.update(node.class_counts)
        self.classes_ = np.array(sorted(labels, key=lambda v: (str(type(v)), v)))
        logger.info(
            "Extraction finished: %d internal nodes, %d leaves, depth %d, %d query instances, %d oracle queries",
            self.n_internal_nodes_,
            self.n_leaves_,
            self.depth_,
            self.n_generated_instances_,
            self.n_oracle_queries_,
        )
