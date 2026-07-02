"""Unit tests for the RegressionDataset class."""

import numpy as np
import pytest
import torch

from src.data.dataset import RegressionDataset


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_data_file(tmp_path):
    """Create a small synthetic regression dataset as a text file."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + np.random.randn(100) * 0.1
    data = np.column_stack([X, y])
    path = tmp_path / "synthetic.txt"
    np.savetxt(path, data)
    return str(path)


@pytest.fixture
def dataset(synthetic_data_file):
    return RegressionDataset(synthetic_data_file, test_size=0.3, seed=42)


# ── RegressionDataset tests ─────────────────────────────────────────────


class TestRegressionDataset:
    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            RegressionDataset("no_existe.txt")

    def test_train_test_split_sizes(self, dataset):
        total = 100
        assert dataset.X_train.shape[0] == pytest.approx(total * 0.7, abs=1)
        assert dataset.X_test.shape[0] == pytest.approx(total * 0.3, abs=1)

    def test_num_features(self, dataset):
        assert dataset.num_features == 4

    def test_num_train_patterns(self, dataset):
        assert dataset.num_train_patterns == dataset.X_train.shape[0]

    def test_y_mean_stored(self, dataset):
        assert isinstance(dataset.y_mean, float)
        assert dataset.y_mean != 0.0

    def test_y_std_stored(self, dataset):
        assert isinstance(dataset.y_std, float)
        assert dataset.y_std > 0.0

    def test_get_tensors_shapes(self, dataset):
        device = torch.device("cpu")
        X_train_t, y_train_t, X_test_t, y_test_t = dataset.get_tensors(device)
        assert X_train_t.shape == dataset.X_train.shape
        assert y_train_t.shape == (dataset.X_train.shape[0],)
        assert X_test_t.shape == dataset.X_test.shape
        assert y_test_t.shape == (dataset.X_test.shape[0],)

    def test_get_tensors_device(self, dataset):
        device = torch.device("cpu")
        tensors = dataset.get_tensors(device)
        for t in tensors:
            assert t.device.type == "cpu"

    def test_get_tensors_dtype(self, dataset):
        device = torch.device("cpu")
        tensors = dataset.get_tensors(device)
        for t in tensors:
            assert t.dtype == torch.float32

    def test_get_tensors_reproducible(self, synthetic_data_file):
        d1 = RegressionDataset(synthetic_data_file, test_size=0.3, seed=42)
        d2 = RegressionDataset(synthetic_data_file, test_size=0.3, seed=42)
        np.testing.assert_array_equal(d1.X_train, d2.X_train)
        np.testing.assert_array_equal(d1.y_test, d2.y_test)
