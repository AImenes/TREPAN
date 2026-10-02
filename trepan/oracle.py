"""The oracle: the trained model whose behaviour the tree must reproduce.

In TREPAN the target concept is *the function computed by the network*, not
the labels of the training data.  The oracle therefore (i) labels the training
examples, (ii) labels the synthetic instances drawn when selecting splits and
(iii) labels the instances drawn when deciding whether a node is a leaf.  Any
object with a scikit-learn style ``predict`` method, or any callable mapping a
2-D array to a 1-D array of labels, can serve as the oracle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

__all__ = ["Oracle"]

Predictor = Callable[[np.ndarray], np.ndarray]


class Oracle:
    """Wraps a trained model and counts how often it is queried.

    Args:
        model: An object exposing ``predict(X) -> labels`` (e.g. any
            scikit-learn classifier or a PyTorch module with such a method), or
            a plain callable with the same contract.
        feature_names: When given, queries are handed to the model as pandas
            DataFrames with these columns.  :class:`~trepan.Trepan` sets this
            automatically when it is fitted on a DataFrame, so models trained
            on named columns (e.g. scikit-learn pipelines) see the same input
            type they were trained on.
        batch_size: If given, large query batches are split into chunks of
            this size before being handed to the model.

    Example:
        >>> from sklearn.neural_network import MLPClassifier
        >>> net = MLPClassifier().fit(X_train, y_train)      # doctest: +SKIP
        >>> oracle = Oracle(net)                              # doctest: +SKIP
        >>> oracle.predict(X_train[:3])                       # doctest: +SKIP
    """

    def __init__(self, model: Any, feature_names: Sequence[str] | None = None, batch_size: int | None = None) -> None:
        if callable(getattr(model, "predict", None)):
            self._predict: Predictor = model.predict
        elif callable(model):
            self._predict = model
        else:
            raise TypeError("model must have a predict(X) method or be callable")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.feature_names = None if feature_names is None else list(feature_names)
        self.batch_size = batch_size
        self.n_queries = 0
        self.n_calls = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the model's class label for every row of ``X``."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be two-dimensional, got shape {X.shape}")
        if len(X) == 0:
            return np.empty(0, dtype=int)
        if self.batch_size is None or len(X) <= self.batch_size:
            chunks = [X]
        else:
            chunks = [X[i : i + self.batch_size] for i in range(0, len(X), self.batch_size)]
        outputs = [np.asarray(self._predict(self._wrap(chunk))).reshape(-1) for chunk in chunks]
        labels = np.concatenate(outputs)
        if len(labels) != len(X):
            raise ValueError(
                f"The oracle returned {len(labels)} labels for {len(X)} instances; "
                "predict(X) must return exactly one label per row"
            )
        self.n_queries += len(X)
        self.n_calls += 1
        return labels

    def _wrap(self, X: np.ndarray) -> Any:
        """Present ``X`` to the model as a DataFrame when feature names are known."""
        if self.feature_names is None:
            return X
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - pandas is optional
            return X
        return pd.DataFrame(X, columns=self.feature_names)

    __call__ = predict

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Oracle(model={type(self.model).__name__}, n_queries={self.n_queries})"
