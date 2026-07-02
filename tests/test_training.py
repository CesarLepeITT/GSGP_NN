"""Integration tests for the training pipeline."""

import pytest
import torch

from src.models.gsgp_nn import GSGPNN
from src.training.early_stopping import EarlyStopping
from src.training.callbacks import TrainingLogger
from src.training.trainer import Trainer


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def small_setup(device):
    """Create a minimal model + synthetic semantic data for testing."""
    torch.manual_seed(42)
    N, M, K = 5, 20, 2

    model = GSGPNN(num_nodes=N, num_layers=K).to(device)

    semantic_data = {
        "initial_semantics": torch.randn(N, M, device=device),
        "parent_semantics": torch.randn(N, M, device=device),
        "route_semantics": [
            {
                "rt1": torch.randn(N, M, device=device),
                "rt2": torch.randn(N, M, device=device),
            }
            for _ in range(K)
        ],
    }

    y_train = torch.randn(M, device=device)

    return model, semantic_data, y_train


class TestTrainer:
    def test_fit_runs(self, small_setup):
        model, sem_data, y_train = small_setup
        trainer = Trainer(model, learning_rate=0.01)
        history = trainer.fit(sem_data, y_train, epochs=5)

        assert "loss_history" in history
        assert len(history["loss_history"]) == 5
        assert history["total_time"] > 0

    def test_loss_decreases(self, small_setup):
        model, sem_data, y_train = small_setup
        trainer = Trainer(model, learning_rate=0.01)
        history = trainer.fit(sem_data, y_train, epochs=50)

        losses = history["loss_history"]
        # Loss at the end should be less than at the start
        assert losses[-1] < losses[0]

    def test_predict_shape(self, small_setup):
        model, sem_data, y_train = small_setup
        trainer = Trainer(model, learning_rate=0.01)
        trainer.fit(sem_data, y_train, epochs=5)

        preds = trainer.predict(sem_data)
        assert preds.shape == y_train.shape

    def test_early_stopping_triggers(self, device):
        """Train with very tight patience on constant loss."""
        torch.manual_seed(0)
        N, M, K = 3, 10, 1
        model = GSGPNN(num_nodes=N, num_layers=K).to(device)
        sem_data = {
            "initial_semantics": torch.zeros(N, M, device=device),
            "parent_semantics": torch.zeros(N, M, device=device),
            "route_semantics": [
                {"rt1": torch.zeros(N, M, device=device),
                 "rt2": torch.zeros(N, M, device=device)}
            ],
        }
        y_train = torch.zeros(M, device=device)

        es = EarlyStopping(patience=5, warmup_epochs=2)
        trainer = Trainer(model, learning_rate=0.001, early_stopping=es)
        history = trainer.fit(sem_data, y_train, epochs=1000)

        # Should stop well before 1000 epochs
        assert history["epochs_trained"] < 1000


class TestEarlyStopping:
    def test_no_stop_during_warmup(self):
        es = EarlyStopping(patience=3, warmup_epochs=5)
        for _ in range(5):
            assert es(100.0) is False  # warmup period

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3, warmup_epochs=0)
        es(1.0)  # best
        es(2.0)  # worse
        es(2.0)  # worse
        assert es(2.0) is True  # 3rd worse → stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3, warmup_epochs=0)
        es(1.0)
        es(2.0)  # +1
        es(2.0)  # +2
        es(0.5)  # improvement → reset
        es(2.0)  # +1 again
        assert es(2.0) is False  # only +2, need 3

    def test_reset_method(self):
        es = EarlyStopping(patience=3, warmup_epochs=0)
        es(1.0)
        es(2.0)
        es.reset()
        assert es.epoch == 0
        assert es.wait == 0
        assert es.best_loss == float("inf")
