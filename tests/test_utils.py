"""Unit tests for the utility modules (seed, device, metrics)."""

import random

import numpy as np
import pytest
import torch

from src.utils.device import get_device, estimate_memory_usage_mb
from src.utils.metrics import compute_metrics, compute_metrics_original_scale
from src.utils.seed import set_seed


# ── set_seed tests ──────────────────────────────────────────────────────


class TestSetSeed:
    def test_determinism(self):
        set_seed(42)
        r1 = random.randint(0, 1000)
        n1 = np.random.randint(0, 1000)
        t1 = torch.randint(0, 1000, (1,)).item()

        set_seed(42)
        r2 = random.randint(0, 1000)
        n2 = np.random.randint(0, 1000)
        t2 = torch.randint(0, 1000, (1,)).item()

        assert r1 == r2
        assert n1 == n2
        assert t1 == t2

    def test_different_seeds_differ(self):
        set_seed(1)
        vals1 = [random.randint(0, 100) for _ in range(5)]

        set_seed(2)
        vals2 = [random.randint(0, 100) for _ in range(5)]

        assert vals1 != vals2


# ── get_device tests ────────────────────────────────────────────────────


class TestGetDevice:
    def test_returns_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_prefer_cpu(self):
        device = get_device(prefer_cuda=False)
        assert device.type == "cpu"


# ── estimate_memory_usage_mb tests ──────────────────────────────────────


class TestEstimateMemory:
    def test_returns_positive(self):
        mem = estimate_memory_usage_mb(num_nodes=10, num_layers=3, num_patterns=100)
        assert mem > 0

    def test_increases_with_nodes(self):
        small = estimate_memory_usage_mb(num_nodes=10, num_layers=3, num_patterns=100)
        large = estimate_memory_usage_mb(num_nodes=20, num_layers=3, num_patterns=100)
        assert large > small

    def test_increases_with_layers(self):
        small = estimate_memory_usage_mb(num_nodes=10, num_layers=1, num_patterns=100)
        large = estimate_memory_usage_mb(num_nodes=10, num_layers=5, num_patterns=100)
        assert large > small

    def test_increases_with_patterns(self):
        small = estimate_memory_usage_mb(num_nodes=10, num_layers=3, num_patterns=100)
        large = estimate_memory_usage_mb(num_nodes=10, num_layers=3, num_patterns=500)
        assert large > small


# ── compute_metrics tests ───────────────────────────────────────────────


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        m = compute_metrics(y, y)
        assert m["mae"] == 0.0
        assert m["mse"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == 1.0

    def test_constant_predictions_at_mean(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        mean = y.mean().item()
        pred = torch.full_like(y, mean)
        m = compute_metrics(y, pred)
        assert m["r2"] == pytest.approx(0.0, abs=1e-6)

    def test_mae_and_mse_known_values(self):
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.5, 2.5, 2.0])
        m = compute_metrics(y_true, y_pred)
        assert m["mse"] == pytest.approx(0.5, abs=1e-5)
        assert m["mae"] == pytest.approx(0.6666667, abs=1e-5)

    def test_works_on_cpu(self):
        y = torch.tensor([0.0, 1.0, 2.0], device="cpu")
        pred = torch.tensor([0.1, 0.9, 2.1], device="cpu")
        m = compute_metrics(y, pred)
        assert all(v > 0 for v in m.values())


class TestComputeMetricsOriginalScale:
    def test_inverse_transform(self):
        y_scaled = torch.tensor([-0.5, 0.0, 0.5])
        y_pred_scaled = torch.tensor([-0.25, 0.25, 0.0])
        m = compute_metrics_original_scale(y_scaled, y_pred_scaled, y_mean=10.0, y_std=2.0)
        # y_true_orig = [9, 10, 11], y_pred_orig = [9.5, 10.5, 10]
        # MAE = (0.5 + 0.5 + 1.0) / 3 = 0.666...
        assert m["mae"] == pytest.approx(2.0 / 3.0, abs=1e-5)

    def test_perfect_after_inverse(self):
        y_scaled = torch.tensor([0.0, 1.0, 2.0])
        m = compute_metrics_original_scale(y_scaled, y_scaled, y_mean=5.0, y_std=3.0)
        assert m["r2"] == 1.0
