"""Training loop for GSGP-NN.

Orchestrates the forward pass, loss computation, automatic
differentiation, and parameter updates on GPU.  Replaces the manual
gradient + Adam logic from the legacy code with standard PyTorch
``loss.backward()`` and ``torch.optim.Adam``.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from src.models.gsgp_nn import GSGPNN
from src.training.callbacks import ModelCheckpoint, TrainingLogger
from src.training.early_stopping import EarlyStopping


class Trainer:
    """Handles the GSGP-NN training loop.

    Args:
        model: The ``GSGPNN`` module (should already be on the target device).
        learning_rate: Learning rate for Adam.
        weight_decay: L2 regularisation coefficient.
        grad_clip_norm: Maximum gradient norm for clipping.
        early_stopping: Optional ``EarlyStopping`` callback.
        logger: Optional ``TrainingLogger`` callback.
        checkpoint: Optional ``ModelCheckpoint`` callback.
    """

    def __init__(
        self,
        model: GSGPNN,
        learning_rate: float = 0.005,
        weight_decay: float = 0.0,
        grad_clip_norm: float = 5.0,
        early_stopping: EarlyStopping | None = None,
        logger: TrainingLogger | None = None,
        checkpoint: ModelCheckpoint | None = None,
    ) -> None:
        self.model = model
        self.grad_clip_norm = grad_clip_norm

        # PyTorch's battle-tested Adam — replaces AdamOptimizerCorregido
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.early_stopping = early_stopping
        self.logger = logger or TrainingLogger()
        self.checkpoint = checkpoint

        # Training history
        self.loss_history: list[float] = []
        self.total_training_time: float = 0.0
        self.epoch_times: list[float] = []

    def fit(
        self,
        semantic_data: dict[str, Any],
        y_train: torch.Tensor,
        epochs: int = 200,
    ) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            semantic_data: Dictionary produced by ``prepare_semantic_data``,
                containing ``initial_semantics``, ``parent_semantics``, and
                ``route_semantics``.
            y_train: Target tensor of shape ``(M,)`` on the same device as
                the model.
            epochs: Maximum number of training epochs.

        Returns:
            Dictionary with training history and timing information.
        """
        initial_sem = semantic_data["initial_semantics"]
        parent_sem = semantic_data["parent_semantics"]
        route_sem = semantic_data["route_semantics"]

        self.model.train()
        self.loss_history.clear()
        self.epoch_times.clear()

        training_start = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # ── Forward ─────────────────────────────────────────────
            y_pred = self.model(initial_sem, parent_sem, route_sem)

            # ── Loss ────────────────────────────────────────────────
            loss = F.mse_loss(y_pred, y_train)

            # ── Backward (replaces ~150 lines of manual gradients) ──
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (replaces manual np.clip per parameter)
            if self.grad_clip_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            # ── Update (replaces manual Adam implementation) ────────
            self.optimizer.step()

            # ── Bookkeeping ─────────────────────────────────────────
            loss_val = loss.item()
            epoch_time = time.time() - epoch_start

            self.loss_history.append(loss_val)
            self.epoch_times.append(epoch_time)

            # Callbacks
            self.logger.on_epoch_end(epoch, loss_val)

            if self.checkpoint is not None:
                self.checkpoint.on_epoch_end(loss_val, self.model)

            # Early stopping
            if self.early_stopping is not None and self.early_stopping(loss_val):
                print(f"\n[EarlyStopping] Stopped at epoch {epoch}")
                break

        self.total_training_time = time.time() - training_start

        print(f"\n[Training] Completed in {self.total_training_time:.2f}s")
        if self.epoch_times:
            avg = self.total_training_time / len(self.epoch_times)
            print(f"[Training] Average time per epoch: {avg:.4f}s")

        return {
            "loss_history": self.loss_history,
            "total_time": self.total_training_time,
            "epoch_times": self.epoch_times,
            "epochs_trained": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else float("inf"),
            "converged": len(self.loss_history) < epochs,
        }

    @torch.no_grad()
    def predict(
        self,
        semantic_data: dict[str, Any],
    ) -> torch.Tensor:
        """Run inference using the trained model.

        Args:
            semantic_data: Semantic tensor dictionary (can be from a
                different dataset, e.g. test set).

        Returns:
            Predictions tensor of shape ``(M,)``.
        """
        self.model.eval()
        return self.model(
            semantic_data["initial_semantics"],
            semantic_data["parent_semantics"],
            semantic_data["route_semantics"],
        )
