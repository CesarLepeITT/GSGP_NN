"""Semantic evaluation of GP trees.

Evaluates each GP individual on the full dataset to produce a *semantic
vector* (one scalar per data-point).  The output is returned as a PyTorch
tensor ready for use on the target device (CPU or CUDA).
"""

from __future__ import annotations

import numpy as np
import torch

from src.evolution.nodes import GPNode


class SemanticEvaluator:
    """Evaluate GP trees and return their semantics as PyTorch tensors.

    Args:
        variable_names: Ordered list of feature names (``x1, x2, …``).
        device: Target ``torch.device`` for the returned tensors.
    """

    def __init__(self, variable_names: list[str], device: torch.device) -> None:
        self.variable_names = variable_names
        self.device = device

    # ------------------------------------------------------------------
    # Single tree evaluation
    # ------------------------------------------------------------------

    def evaluate_individual(
        self, tree: GPNode, X: np.ndarray, koza_division: bool = True
    ) -> torch.Tensor:
        """Evaluate *tree* on every row of *X*.

        Args:
            tree: Root node of a GP expression tree.
            X: Feature matrix of shape ``(M, n_features)``.
            koza_division: Use Koza's protected division formula.

        Returns:
            Normalised semantic tensor of shape ``(M,)`` on ``self.device``.
        """
        M = X.shape[0]
        semantics = np.zeros(M, dtype=np.float64)

        for i in range(M):
            variables = {
                name: X[i, j]
                for j, name in enumerate(self.variable_names)
                if j < X.shape[1]
            }
            try:
                semantics[i] = tree.evaluate(variables, koza_division=koza_division)
            except Exception:
                semantics[i] = 0.0

        semantics = self._normalize_semantics(semantics)
        return torch.tensor(semantics, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Population evaluation
    # ------------------------------------------------------------------

    def create_semantic_matrix(
        self, population: list[GPNode], X: np.ndarray, koza_division: bool = True
    ) -> torch.Tensor:
        """Evaluate an entire population on *X*.

        Args:
            population: List of GP tree root nodes.
            X: Feature matrix of shape ``(M, n_features)``.
            koza_division: Use Koza's protected division formula.

        Returns:
            Tensor of shape ``(pop_size, M)`` on ``self.device``.
        """
        rows = [self.evaluate_individual(tree, X, koza_division=koza_division) for tree in population]
        return torch.stack(rows, dim=0)  # (pop_size, M)

    # ------------------------------------------------------------------
    # Normalisation (matches legacy behaviour exactly)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_semantics(semantics: np.ndarray) -> np.ndarray:
        """Clip and standardise a raw semantic vector.

        If the standard deviation is near zero the vector is replaced
        with a deterministic ramp to avoid constant features.
        """
        semantics = np.clip(semantics, -1e6, 1e6)
        mean = np.mean(semantics)
        std = np.std(semantics)

        if std < 1e-10:
            return np.full_like(
                semantics, mean + 1e-8 * np.arange(len(semantics))
            )

        normalised = (semantics - mean) / std
        return np.clip(normalised, -5.0, 5.0)
