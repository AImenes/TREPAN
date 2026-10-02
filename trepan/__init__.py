"""TREPAN: extracting tree-structured representations of trained networks.

A Python implementation of the TREPAN algorithm (Craven & Shavlik, 1995;
Craven, 1996) for distilling a trained classifier -- typically a neural
network -- into a comprehensible decision tree with m-of-n splits.

Typical use::

    from trepan import Trepan

    tree = Trepan(trained_model, min_sample=1000, max_internal_nodes=15, random_state=0)
    tree.fit(X_train)
    print(tree.export_text(class_names=["setosa", "versicolor", "virginica"]))
    print("fidelity:", tree.fidelity(X_test))
"""

import logging

from .core import Trepan
from .oracle import Oracle
from .splits import Constraint, Literal, MofNTest
from .tree import Node, export_dot, export_text, prune_redundant_subtrees

__version__ = "1.0.0"

__all__ = [
    "Trepan",
    "Oracle",
    "Literal",
    "MofNTest",
    "Constraint",
    "Node",
    "export_text",
    "export_dot",
    "prune_redundant_subtrees",
    "__version__",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
