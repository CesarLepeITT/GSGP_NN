"""Early stopping callback for training.

Monitors a loss metric and signals the training loop to stop when no
improvement is observed for ``patience`` epochs (after an initial warmup).
"""

from __future__ import annotations

import math


class EarlyStopping:
    """Early stopping with warmup period and best-loss tracking.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum decrease in loss to qualify as improvement.
        warmup_epochs: Number of initial epochs where early stopping
            is disabled.
    """

    def __init__(
        self,
        patience: int = 100,
        min_delta: float = 1e-6,
        warmup_epochs: int = 50,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs

        self.best_loss: float = float("inf")
        self.wait: int = 0
        self.epoch: int = 0
        self.stopped_epoch: int = 0

    def __call__(self, current_loss: float) -> bool:
        """Check whether training should stop.

        Args:
            current_loss: Loss value at the current epoch.

        Returns:
            ``True`` if training should stop.
        """
        self.epoch += 1

        # Guard against NaN/Inf
        if not math.isfinite(current_loss):
            current_loss = float("inf")

        # No stopping during warmup
        if self.epoch < self.warmup_epochs:
            self.best_loss = min(self.best_loss, current_loss)
            return False

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            return False

        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = self.epoch
            return True
        return False

    def reset(self) -> None:
        """Reset internal state for a new training run."""
        self.best_loss = float("inf")
        self.wait = 0
        self.epoch = 0
        self.stopped_epoch = 0

    def state_dict(self) -> dict:
        """Serialise state for checkpointing."""
        return {
            "best_loss": self.best_loss,
            "wait": self.wait,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a checkpoint."""
        self.best_loss = state["best_loss"]
        self.wait = state["wait"]
        self.epoch = state["epoch"]
