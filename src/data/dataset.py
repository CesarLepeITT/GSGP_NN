"""Dataset loading and preprocessing for regression tasks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class RegressionDataset:
    """Load a regression dataset from a whitespace-delimited text file.

    The last column is treated as the target variable.  Features and
    targets are independently standardised with ``StandardScaler``.

    Args:
        path: Path to the ``.txt`` data file.
        test_size: Fraction of data reserved for testing.
        seed: Random seed for the train/test split.
    """

    def __init__(
        self, path: str, test_size: float = 0.3, seed: int = 0
    ) -> None:
        self.path = path
        self.seed = seed
        self.test_size = test_size

        # Scalers (fitted during ``_load``)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        # Raw and scaled arrays (set by ``_load``)
        self.X_train: np.ndarray
        self.X_test: np.ndarray
        self.y_train: np.ndarray
        self.y_test: np.ndarray

        # Statistics for inverse-transform on tensors
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

        self._load(path, test_size, seed)

    # ------------------------------------------------------------------

    def _load(self, path: str, test_size: float, seed: int) -> None:
        """Load, scale, and split the dataset."""
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Dataset not found: {resolved.resolve()}"
            )

        data = np.loadtxt(str(resolved))
        X = data[:, :-1]
        y = data[:, -1]

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Store inverse-transform constants
        self.y_mean = float(self.scaler_y.mean_[0])
        self.y_std = float(self.scaler_y.scale_[0])

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=seed
        )

        print(
            f"[Data] Loaded {resolved.name}: "
            f"{X.shape[0]} samples, {X.shape[1]} features | "
            f"Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}"
        )

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def get_tensors(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(X_train, y_train, X_test, y_test)`` as float32 tensors.

        Args:
            device: Target device for the tensors.
        """
        return (
            torch.tensor(self.X_train, dtype=torch.float32, device=device),
            torch.tensor(self.y_train, dtype=torch.float32, device=device),
            torch.tensor(self.X_test, dtype=torch.float32, device=device),
            torch.tensor(self.y_test, dtype=torch.float32, device=device),
        )

    @property
    def num_features(self) -> int:
        """Number of input features."""
        return self.X_train.shape[1]

    @property
    def num_train_patterns(self) -> int:
        """Number of training patterns (M)."""
        return self.X_train.shape[0]
