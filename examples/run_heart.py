#!/usr/bin/env python
"""Extract a TREPAN tree from a neural network trained on the heart-disease data.

``data/heart.csv`` is the Cleveland heart-disease data (one of the four domains
used in the original paper).  The nominal attributes -- sex, chest-pain type,
fasting blood sugar, resting ECG, exercise-induced angina, slope, number of
vessels and thalassemia -- are already integer-coded in the file, so they are
passed to TREPAN as categorical features.

Run from the repository root::

    python examples/run_heart.py --min-sample 1000 --max-internal-nodes 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from examples.common import add_common_arguments, configure_logging, report
from trepan import Trepan

CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
CLASS_NAMES = ["no disease", "disease"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "heart.csv")
    args = parser.parse_args()
    configure_logging(args.verbose)

    frame = pd.read_csv(args.data).drop_duplicates()
    X, y = frame.drop(columns="target"), frame["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    network = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=5000, random_state=args.seed),
    ).fit(X_train, y_train)
    print(f"Network trained: training accuracy {network.score(X_train, y_train):.3f}")

    tree = Trepan(
        network,
        categorical_features=CATEGORICAL,
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

    report(tree, X_test, y_test, network, CLASS_NAMES, args.output_dir, "heart", not args.no_render)


if __name__ == "__main__":
    main()
