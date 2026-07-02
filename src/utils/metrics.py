"""Evaluation metrics computed on PyTorch tensors."""

from __future__ import annotations

import torch


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """Compute standard regression metrics on GPU/CPU tensors.

    All computations stay on the tensor's device to avoid unnecessary
    transfers.  The returned values are plain Python floats.

    Args:
        y_true: Ground-truth targets of shape ``(M,)``.
        y_pred: Model predictions of shape ``(M,)``.

    Returns:
        Dictionary with keys ``mae``, ``mse``, ``rmse``, ``r2``.
    """
    with torch.no_grad():
        residuals = y_true - y_pred
        mae = residuals.abs().mean()
        mse = (residuals ** 2).mean()
        rmse = mse.sqrt()

        ss_res = (residuals ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-12)

    return {
        "mae": mae.item(),
        "mse": mse.item(),
        "rmse": rmse.item(),
        "r2": r2.item(),
    }


def compute_metrics_original_scale(
    y_true_scaled: torch.Tensor,
    y_pred_scaled: torch.Tensor,
    y_mean: float,
    y_std: float,
) -> dict[str, float]:
    """Compute metrics after inverse-transforming to original scale.

    Instead of depending on sklearn's scaler, we apply the inverse
    transform manually using the stored mean/std.

    Args:
        y_true_scaled: Scaled ground-truth targets.
        y_pred_scaled: Scaled model predictions.
        y_mean: Mean of the original (unscaled) target.
        y_std: Std of the original (unscaled) target.

    Returns:
        Dictionary with keys ``mae``, ``mse``, ``rmse``, ``r2``
        (all in original scale).
    """
    y_true_orig = y_true_scaled * y_std + y_mean
    y_pred_orig = y_pred_scaled * y_std + y_mean
    return compute_metrics(y_true_orig, y_pred_orig)

if __name__ == "__main__":
    print("Testing Standard Scale Metrics")
    y_true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y_pred = torch.tensor([1.5, 2.5, 2.0], dtype=torch.float32)

    metrics = compute_metrics(y_true, y_pred)
    for key, value in metrics.items():
        print(f"{key.upper()}: {value:.4f}")

    #Expected MSE is (0.25 + 0.25 + 1.0) / 3 = 0.5
    assert abs(metrics["mse"] - 0.5) < 1e-5, "Standard MSE calculation failed."
    print("Standard metrics test passed.\n")

    print("Testing Original Scale Metrics")
    y_mean = 10.0
    y_std = 2.0
    y_true_scaled = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)  
    y_pred_scaled = torch.tensor([-0.25, 0.25, 0.0], dtype=torch.float32) 

    metrics_orig = compute_metrics_original_scale(y_true_scaled, y_pred_scaled, y_mean, y_std)
    for key, value in metrics_orig.items():
        print(f"{key.upper()}: {value:.4f}")

    #Expected MAE on unscaled data is (0.5 + 0.5 + 1.0) / 3 = 0.6666...
    assert abs(metrics_orig["mae"] - (2.0 / 3.0)) < 1e-5, "Original scale MAE calculation failed."
    print("Original scale metrics test passed.\n")

