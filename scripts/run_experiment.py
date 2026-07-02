"""Main experiment runner for GSGP-NN.

Usage examples::

    # Full robustness analysis (all datasets, 30 runs each)
    python scripts/run_experiment.py

    # Single dataset
    python scripts/run_experiment.py --dataset eheating.txt

    # Custom parameters
    python scripts/run_experiment.py --dataset tower.txt --num-runs 5 --device cuda

    # Force CPU
    python scripts/run_experiment.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.dataset import RegressionDataset
from src.data.loader import load_config, prepare_semantic_data
from src.models.gsgp_nn import GSGPNN
from src.training.callbacks import ModelCheckpoint, TrainingLogger
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer
from src.utils.device import get_device, estimate_memory_usage_mb
from src.utils.metrics import compute_metrics, compute_metrics_original_scale
from src.utils.seed import set_seed


# ── Single experiment ──────────────────────────────────────────────────


def run_single_experiment(
    config: dict,
    seed: int,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Run a single training + evaluation experiment.

    Args:
        config: Dictionary with keys ``path``, ``K``, ``N``, ``lr``, ``epochs``.
        seed: Random seed for reproducibility.
        device: Target compute device.
        verbose: Whether to print per-epoch logs.

    Returns:
        Dictionary of all computed metrics and timing info.
    """
    set_seed(seed)

    # Load data
    dataset = RegressionDataset(
        path=config["path"], test_size=0.3, seed=seed
    )
    X_train_t, y_train_t, X_test_t, y_test_t = dataset.get_tensors(device)

    N = config["N"]
    K = config["K"]
    population_size = int(2 * N * K)

    # Memory estimate
    mem_mb = estimate_memory_usage_mb(N, K, dataset.num_train_patterns)
    print(f"[Memory] Estimated GPU usage: {mem_mb:.1f} MB")

    # Prepare GP semantics (CPU evaluation → GPU tensors)
    train_sem = prepare_semantic_data(
        X=dataset.X_train,
        num_nodes=N,
        num_layers=K,
        population_size=population_size,
        device=device,
    )

    # Build model
    model = GSGPNN(num_nodes=N, num_layers=K).to(device)
    print(f"\n{model.summary()}\n")

    # Train
    trainer = Trainer(
        model=model,
        learning_rate=config["lr"],
        grad_clip_norm=5.0,
        early_stopping=EarlyStopping(patience=150, warmup_epochs=30),
        logger=TrainingLogger(print_every=1 if verbose else 50),
        checkpoint=ModelCheckpoint(save_dir="results"),
    )

    history = trainer.fit(train_sem, y_train_t, epochs=config["epochs"])

    # ── Evaluate on training set ────────────────────────────────────
    y_pred_train = trainer.predict(train_sem)
    train_metrics = compute_metrics(y_train_t, y_pred_train)
    train_metrics_orig = compute_metrics_original_scale(
        y_train_t, y_pred_train, dataset.y_mean, dataset.y_std
    )

    # ── Evaluate on test set ────────────────────────────────────────
    # Re-generate semantics for test data using the SAME GP populations
    test_sem = prepare_semantic_data(
        X=dataset.X_test,
        num_nodes=N,
        num_layers=K,
        population_size=population_size,
        device=device,
        initial_population=train_sem["initial_population"],
        auxiliary_population=train_sem["auxiliary_population"],
        evaluator=train_sem["evaluator"],
    )
    y_pred_test = trainer.predict(test_sem)
    test_metrics = compute_metrics(y_test_t, y_pred_test)
    test_metrics_orig = compute_metrics_original_scale(
        y_test_t, y_pred_test, dataset.y_mean, dataset.y_std
    )

    return {
        "seed": seed,
        "mae_train": train_metrics["mae"],
        "mae_test": test_metrics["mae"],
        "mse_train": train_metrics["mse"],
        "mse_test": test_metrics["mse"],
        "rmse_train": train_metrics["rmse"],
        "rmse_test": test_metrics["rmse"],
        "r2_train": train_metrics["r2"],
        "r2_test": test_metrics["r2"],
        "mae_train_orig": train_metrics_orig["mae"],
        "mae_test_orig": test_metrics_orig["mae"],
        "mse_train_orig": train_metrics_orig["mse"],
        "mse_test_orig": test_metrics_orig["mse"],
        "rmse_train_orig": train_metrics_orig["rmse"],
        "rmse_test_orig": test_metrics_orig["rmse"],
        "r2_train_orig": train_metrics_orig["r2"],
        "r2_test_orig": test_metrics_orig["r2"],
        "training_time": history["total_time"],
        "epochs_trained": history["epochs_trained"],
        "final_loss": history["final_loss"],
        "converged": history["converged"],
    }


# ── Robustness analysis ───────────────────────────────────────────────


def run_robustness_analysis(
    config: dict,
    num_runs: int,
    device: torch.device,
    verbose_individual: bool = True,
) -> pd.DataFrame:
    """Run multiple experiments and return a DataFrame of results.

    Args:
        config: Single dataset configuration dictionary.
        num_runs: Number of independent runs.
        device: Target device.
        verbose_individual: Print per-epoch logs for each run.

    Returns:
        ``pd.DataFrame`` with one row per run.
    """
    print("=" * 80)
    print(f"ROBUSTNESS ANALYSIS: {num_runs} independent runs")
    print(
        f"Dataset: {config['path']}, K={config['K']}, N={config['N']}, "
        f"LR={config['lr']}, Epochs={config['epochs']}"
    )
    print("=" * 80)

    results = []
    start_total = time.time()

    for i in range(num_runs):
        seed = i
        print(f"\n{'=' * 50}")
        print(f"RUN {i + 1}/{num_runs}  (seed={seed})")
        print(f"{'=' * 50}")

        try:
            result = run_single_experiment(
                config, seed, device, verbose=verbose_individual
            )
            results.append(result)

            print(f"✓ Run {i + 1} — R² test: {result['r2_test_orig']:.4f}, "
                  f"RMSE test: {result['rmse_test_orig']:.4f}, "
                  f"Time: {result['training_time']:.2f}s")
        except Exception as e:
            print(f"✗ Run {i + 1} failed: {e}")
            continue

    elapsed = time.time() - start_total

    df = pd.DataFrame(results)

    print(f"\n{'=' * 80}")
    print(f"COMPLETED: {len(results)}/{num_runs} successful runs in {elapsed:.1f}s")
    print(f"{'=' * 80}")

    if not df.empty:
        _print_summary(df, config["path"])
        _save_results(df, config["path"], num_runs)

    return df


def _print_summary(df: pd.DataFrame, dataset_name: str) -> None:
    """Print descriptive statistics for key metrics."""
    print(f"\n{'=' * 60}")
    print(f"ROBUSTNESS STATISTICS — {dataset_name}")
    print(f"{'=' * 60}")

    for col, label in [
        ("r2_test_orig", "R² Test"),
        ("rmse_test_orig", "RMSE Test"),
        ("mae_test_orig", "MAE Test"),
        ("training_time", "Time (s)"),
    ]:
        vals = df[col]
        print(f"\n{label}:")
        print(f"  Mean:   {vals.mean():.6f}")
        print(f"  Median: {vals.median():.6f}")
        print(f"  Std:    {vals.std():.6f}")
        print(f"  Min:    {vals.min():.6f}")
        print(f"  Max:    {vals.max():.6f}")
        print(f"  Q1:     {vals.quantile(0.25):.6f}")
        print(f"  Q3:     {vals.quantile(0.75):.6f}")


def _save_results(df: pd.DataFrame, dataset_name: str, num_runs: int) -> None:
    """Save results to a CSV file in the results/ directory."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = dataset_name.replace(".txt", "").replace(".", "_")
    csv_path = results_dir / f"gsgp_nn_{clean_name}_{num_runs}runs_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Results] Saved to: {csv_path}")


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="GSGP-NN Experiment Runner (PyTorch + CUDA)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run only this dataset (e.g. 'eheating.txt'). Default: all.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Number of runs per dataset. Overrides config value.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-epoch logging.",
    )
    args = parser.parse_args()

    # Device
    prefer_cuda = args.device != "cpu"
    device = get_device(prefer_cuda=prefer_cuda)

    # Config
    config = load_config(args.config)
    num_runs = args.num_runs or config.get("num_corridas", 30)
    configurations = config["configuraciones"]

    # Filter to single dataset if requested
    if args.dataset:
        configurations = [c for c in configurations if c["path"] == args.dataset]
        if not configurations:
            print(f"Dataset '{args.dataset}' not found in config.")
            sys.exit(1)

    # Run
    for cfg in configurations:
        print(f"\n{'=' * 100}")
        print(f"DATASET: {cfg['path']}")
        print(f"{'=' * 100}")
        run_robustness_analysis(
            config=cfg,
            num_runs=num_runs,
            device=device,
            verbose_individual=not args.quiet,
        )


if __name__ == "__main__":
    main()
