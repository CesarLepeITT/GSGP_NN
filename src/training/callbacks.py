"""Training callbacks — logging and checkpointing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch


class TrainingLogger:
    """Logs training metrics to the console.

    Args:
        print_every: Print a log line every *n* epochs.
        window_size: Number of past epochs used to compute improvement
            percentage.
    """

    def __init__(self, print_every: int = 1, window_size: int = 25) -> None:
        self.print_every = print_every
        self.window_size = window_size
        self._history: list[float] = []
        self._epoch_start: float = 0.0

    def on_epoch_start(self) -> None:
        """Mark the beginning of an epoch."""
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch: int, loss: float) -> None:
        """Log metrics at the end of an epoch.

        Args:
            epoch: 1-indexed epoch number.
            loss: Training loss value.
        """
        elapsed = time.time() - self._epoch_start
        self._history.append(loss)

        if epoch % self.print_every != 0:
            return

        improvement = ""
        if len(self._history) >= self.window_size:
            prev = self._history[-self.window_size]
            if prev > 1e-10:
                pct = ((prev - loss) / prev) * 100
                arrow = "↓" if pct > 0 else "↑"
                improvement = f"({arrow}{abs(pct):.2f}%)"

        print(
            f"Epoch {epoch:4d}: Loss = {loss:.6f} {improvement}  "
            f"[{elapsed:.3f}s]"
        )


class ModelCheckpoint:
    """Save model weights when a monitored metric improves.

    Args:
        save_dir: Directory to write checkpoint files into.
        filename: Checkpoint filename template.
    """

    def __init__(
        self,
        save_dir: str = "results",
        filename: str = "best_model.pt",
    ) -> None:
        self.save_path = Path(save_dir) / filename
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_loss: float = float("inf")

    def on_epoch_end(
        self, loss: float, model: torch.nn.Module
    ) -> bool:
        """Save the model if *loss* improved.

        Returns:
            ``True`` if the model was saved.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(model.state_dict(), self.save_path)
            return True
        return False
