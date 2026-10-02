"""PyTorch networks with a scikit-learn style ``predict`` method.

These are tidied versions of the feed-forward networks from the original
project.  Anything exposing ``predict(X) -> labels`` can serve as TREPAN's
oracle, so a trained instance can be handed to :class:`trepan.Trepan` directly::

    net = FeedForwardClassifier(input_dim=4, hidden_dims=(16, 12), output_dim=3)
    net.fit(X_train, y_train, epochs=200)
    tree = Trepan(net, min_sample=1000).fit(X_train)

PyTorch is an optional dependency (``pip install torch``); this module is not
imported by the package itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FeedForwardClassifier(nn.Module):
    """A fully connected classifier with configurable hidden layers.

    Args:
        input_dim: Number of input features.
        hidden_dims: Sizes of the hidden layers, e.g. ``(16, 12)`` for the Iris
            network or ``(64, 128, 256)`` for the heart-disease network of the
            original project.
        output_dim: Number of classes.
        dropout: Dropout probability applied after the last hidden layer.
        activation: Activation module class, e.g. ``nn.ReLU`` or ``nn.LeakyReLU``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers += [nn.Linear(previous, width), activation()]
            previous = width
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102 - PyTorch API
        return self.network(x)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        log_every: int | None = 10,
    ) -> FeedForwardClassifier:
        """Train with Adam and cross-entropy loss."""
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y_t = torch.as_tensor(np.asarray(y, dtype=np.int64))
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(epochs):
            total = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch_X)
            if log_every and epoch % log_every == 0:
                print(f"epoch {epoch:4d}  loss {total / len(loader.dataset):.4f}")
        self.eval()
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class index for every row of ``X``."""
        self.eval()
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        return self(X_t).argmax(dim=1).cpu().numpy()
