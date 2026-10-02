#!/usr/bin/env python
"""Extract a TREPAN tree from a neural network trained on the Iris data set.

The network is a scikit-learn multi-layer perceptron (two hidden layers of 16
and 12 units, as in the original notebooks) wrapped in a standardisation
pipeline so that TREPAN can hand it raw feature values.

Run from the repository root::

    python examples/run_iris.py --min-sample 1000 --max-internal-nodes 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from examples.common import add_common_arguments, configure_logging, report
from trepan import Trepan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    args = parser.parse_args()
    configure_logging(args.verbose)

    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    network = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(16, 12), max_iter=5000, random_state=args.seed),
    ).fit(X_train, y_train)
    print(f"Network trained: training accuracy {network.score(X_train, y_train):.3f}")

    tree = Trepan(
        network,
        max_internal_nodes=args.max_internal_nodes,
        min_sample=args.min_sample,
        epsilon=args.epsilon,
        delta=args.delta,
        beam_width=args.beam_width,
        max_literals=args.max_literals,
        criterion=args.criterion,
        stopping_rule=args.stopping_rule,
        validation_fraction=args.validation_fraction,
        random_state=args.seed,
    ).fit(X_train)

    report(tree, X_test, y_test, network, list(iris.target_names), args.output_dir, "iris", not args.no_render)


if __name__ == "__main__":
    main()
