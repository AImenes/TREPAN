"""Tree nodes and read-only tree utilities (rendering, traversal, pruning)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from .splits import Constraint, MofNTest

__all__ = ["Node", "export_text", "export_dot", "prune_redundant_subtrees"]


@dataclass(eq=False)
class Node:
    """A node of a TREPAN tree.

    Every node stores the training examples that reach it together with their
    oracle labels, the synthetic instances drawn for it (the *query
    instances*) and the constraints on the path from the root.  Internal nodes
    additionally carry an m-of-n split and exactly two children:
    ``children[0]`` receives the instances that satisfy the split,
    ``children[1]`` the others.

    Attributes:
        node_id: Creation order, unique within a tree.
        depth: Distance from the root.
        constraints: Path constraints, root first.
        X: Training examples reaching the node, shape ``(n_examples, n_features)``.
        y: Oracle labels of ``X``.
        parent: Parent node, ``None`` for the root.
        children: Child nodes (empty for a leaf).
        split: The m-of-n test of an internal node.
        split_gain: Information gain (or gain ratio) of that test on the node's instances.
        label: Predicted class (majority of the oracle labels at the node).
        class_counts: Oracle label counts over training examples and query instances.
        reach: Estimated fraction of instances that reach the node.
        fidelity: Estimated agreement between ``label`` and the oracle at the node.
        query_X, query_y: Synthetic instances drawn for the node and their labels.
        distributions: Instance model used to draw query instances at this node.
        expansion_index: Position in the best-first expansion order (internal nodes).
    """

    node_id: int
    depth: int
    constraints: tuple[Constraint, ...]
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    parent: Node | None = field(default=None, repr=False)
    children: list[Node] = field(default_factory=list, repr=False)
    split: MofNTest | None = None
    split_gain: float | None = None
    label: object = None
    class_counts: dict = field(default_factory=dict)
    reach: float = 1.0
    fidelity: float = 0.0
    query_X: np.ndarray | None = field(default=None, repr=False)
    query_y: np.ndarray | None = field(default=None, repr=False)
    distributions: object = field(default=None, repr=False)
    expansion_index: int | None = None

    # -- basic properties -------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        """Whether the node has no children."""
        return not self.children

    @property
    def n_examples(self) -> int:
        """Number of training examples reaching the node."""
        return len(self.X)

    @property
    def n_queries(self) -> int:
        """Number of query instances drawn for the node."""
        return 0 if self.query_X is None else len(self.query_X)

    @property
    def n_instances(self) -> int:
        """Training examples plus query instances."""
        return self.n_examples + self.n_queries

    @property
    def priority(self) -> float:
        """Best-first expansion score ``f(n) = reach(n) * (1 - fidelity(n))``."""
        return self.reach * (1.0 - self.fidelity)

    def pool(self) -> tuple[np.ndarray, np.ndarray]:
        """All instances at the node (training examples first, then queries) and their labels."""
        if self.query_X is None or len(self.query_X) == 0:
            return self.X, self.y
        if len(self.X) == 0:
            return self.query_X, self.query_y
        return np.vstack([self.X, self.query_X]), np.concatenate([self.y, self.query_y])

    def add_queries(self, X: np.ndarray, y: np.ndarray) -> None:
        """Append freshly drawn, oracle-labelled instances to the node."""
        if len(X) == 0:
            return
        if self.query_X is None or len(self.query_X) == 0:
            self.query_X, self.query_y = X, y
        else:
            self.query_X = np.vstack([self.query_X, X])
            self.query_y = np.concatenate([self.query_y, y])

    def constrained_features(self) -> set[int]:
        """Features referenced by any test on the path from the root."""
        return {f for c in self.constraints for f in c.test.features}

    def features_in_compound_splits(self) -> set[int]:
        """Features used by compound (n >= 2) splits on the path from the root.

        The same feature may not appear in two m-of-n tests on one
        root-to-leaf path; this is the set of features already taken.
        """
        return {f for c in self.constraints if c.test.is_compound for f in c.test.features}

    # -- traversal ----------------------------------------------------------

    def iter_nodes(self) -> Iterator[Node]:
        """Pre-order traversal of the subtree rooted at this node."""
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def leaves(self) -> list[Node]:
        """Leaves of the subtree in pre-order."""
        return [n for n in self.iter_nodes() if n.is_leaf]

    def internal_nodes(self) -> list[Node]:
        """Internal nodes of the subtree in pre-order."""
        return [n for n in self.iter_nodes() if not n.is_leaf]

    def route(self, x: np.ndarray) -> Node:
        """Follow the splits from this node down to the leaf covering instance ``x``."""
        node = self
        row = np.asarray(x, dtype=float).reshape(1, -1)
        while not node.is_leaf:
            node = node.children[0] if bool(node.split.evaluate(row)[0]) else node.children[1]
        return node

    def collapse(self) -> None:
        """Turn this node into a leaf, discarding its subtree."""
        self.children = []
        self.split = None
        self.split_gain = None
        self.expansion_index = None


def prune_redundant_subtrees(root: Node) -> int:
    """Collapse subtrees whose leaves all predict the same class (thesis, 3.2.6).

    The pass is a post-order traversal so that nested redundant subtrees are
    simplified as far as possible.  Predictions are unchanged.

    Returns:
        Number of internal nodes removed.
    """
    removed = 0

    def visit(node: Node) -> object:
        nonlocal removed
        if node.is_leaf:
            return node.label
        labels = {visit(child) for child in node.children}
        if len(labels) == 1:
            removed += len(node.internal_nodes())
            node.collapse()
            node.label = labels.pop()
            return node.label
        return object()  # sentinel: mixed predictions below this node

    visit(root)
    return removed


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _class_name(label: object, class_names: Sequence[str] | None) -> str:
    if class_names is None:
        return str(label)
    try:
        return str(class_names[int(label)])
    except (TypeError, ValueError, IndexError):
        return str(label)


def _counts_text(node: Node, class_names: Sequence[str] | None) -> str:
    return ", ".join(
        f"{_class_name(k, class_names)}: {v}" for k, v in sorted(node.class_counts.items(), key=lambda kv: str(kv[0]))
    )


def export_text(
    root: Node,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    show_counts: bool = True,
) -> str:
    """Render a tree as indented pseudo-code, one line per node.

    Args:
        root: Root node of the tree.
        feature_names: Names used for features (defaults to ``x0, x1, ...``).
        class_names: Names used for class labels.
        show_counts: Append the oracle label counts at each leaf.
    """
    lines: list[str] = []

    def walk(node: Node, indent: str) -> None:
        if node.is_leaf:
            counts = f"  [{_counts_text(node, class_names)}]" if show_counts and node.class_counts else ""
            lines.append(f"{indent}class: {_class_name(node.label, class_names)}{counts}")
            return
        lines.append(f"{indent}if {node.split.describe(feature_names)}:")
        walk(node.children[0], indent + "    ")
        lines.append(f"{indent}else:")
        walk(node.children[1], indent + "    ")

    walk(root, "")
    return "\n".join(lines)


def export_dot(
    root: Node,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    show_counts: bool = True,
    title: str | None = None,
) -> str:
    """Render a tree in Graphviz DOT format.

    The returned string can be written to a ``.dot`` file, rendered with the
    ``dot`` command-line tool, or wrapped in ``graphviz.Source`` from the
    optional ``graphviz`` Python package.
    """

    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    lines = ["digraph TREPAN {", '    graph [rankdir=TB, fontname="Helvetica"];']
    if title:
        lines.append(f'    label="{escape(title)}"; labelloc=t;')
    lines.append('    node [fontname="Helvetica", style=filled, fontcolor=white];')
    lines.append('    edge [fontname="Helvetica"];')

    for node in root.iter_nodes():
        if node.is_leaf:
            text = f"class: {_class_name(node.label, class_names)}"
            if show_counts and node.class_counts:
                text += f"\\n[{_counts_text(node, class_names)}]"
            text += f"\\nfidelity {node.fidelity:.2f}"
            lines.append(f'    n{node.node_id} [label="{escape(text)}", shape=ellipse, fillcolor="#6b8e23"];')
        else:
            text = node.split.describe(feature_names)
            lines.append(f'    n{node.node_id} [label="{escape(text)}", shape=box, fillcolor="#191970"];')
            lines.append(f'    n{node.node_id} -> n{node.children[0].node_id} [label="yes"];')
            lines.append(f'    n{node.node_id} -> n{node.children[1].node_id} [label="no"];')
    lines.append("}")
    return "\n".join(lines)
