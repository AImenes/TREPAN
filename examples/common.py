"""Helpers shared by the example scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from trepan import Trepan


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Command-line options shared by every example."""
    parser.add_argument(
        "--min-sample", type=int, default=1000, help="min_sample: instances considered per split (default 1000)"
    )
    parser.add_argument("--max-internal-nodes", type=int, default=15, help="global size limit (default 15)")
    parser.add_argument("--epsilon", type=float, default=0.05, help="purity tolerance of the local stopping rule")
    parser.add_argument("--delta", type=float, default=0.05, help="confidence parameter of the local stopping rule")
    parser.add_argument("--beam-width", type=int, default=2, help="beam width of the m-of-n search")
    parser.add_argument("--max-literals", type=int, default=None, help="maximum literals per split (default unlimited)")
    parser.add_argument("--criterion", choices=["gain", "gain_ratio"], default="gain", help="split criterion")
    parser.add_argument("--stopping-rule", choices=["strict", "interval"], default="strict", help="local stopping rule")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help="hold out this fraction of the training data to pick the best nested tree (thesis: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for the network and the sampler")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/output"), help="where to write the tree files"
    )
    parser.add_argument("--no-render", action="store_true", help="skip rendering the PNG with Graphviz")
    parser.add_argument("-v", "--verbose", action="store_true", help="log the expansion of every node")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )


def report(tree: Trepan, X_test, y_test, oracle, class_names, output_dir: Path, stem: str, render: bool) -> None:
    """Print the extracted tree and its quality metrics, and write DOT / PNG files."""
    print("\nExtracted tree")
    print("--------------")
    print(tree.export_text(class_names=class_names))
    print()
    print(
        f"internal nodes: {tree.n_internal_nodes_}   feature references: {tree.n_feature_references_}   "
        f"depth: {tree.depth_}   pruned: {tree.n_pruned_nodes_}"
    )
    if tree.validation_fidelity_curve_:
        curve = ", ".join(f"{k}:{f:.2f}" for k, f in tree.validation_fidelity_curve_)
        print(f"validation fidelity by number of internal nodes: {curve}")
    print(f"oracle queries: {tree.n_oracle_queries_}   generated instances: {tree.n_generated_instances_}")
    print(f"test-set accuracy of the network : {oracle.score(X_test, y_test):.3f}")
    print(f"test-set accuracy of the tree    : {tree.score(X_test, y_test):.3f}")
    print(f"test-set fidelity to the network : {tree.fidelity(X_test):.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dot_path = output_dir / f"{stem}.dot"
    dot_path.write_text(tree.export_dot(class_names=class_names, title=f"TREPAN tree for {stem}"))
    print(f"\nDOT file written to {dot_path}")
    if render:
        try:
            import graphviz

            png_path = graphviz.Source(dot_path.read_text()).render(output_dir / stem, format="png", cleanup=True)
            print(f"PNG written to {png_path}")
        except ImportError:
            print("Install the 'graphviz' Python package to render the PNG, or run: dot -Tpng", dot_path)
        except Exception as exc:  # graphviz.ExecutableNotFound and friends
            print(f"Could not render the PNG ({exc}); run: dot -Tpng {dot_path} -o {output_dir / stem}.png")
