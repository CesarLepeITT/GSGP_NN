"""Integration tests for the training pipeline."""

from pathlib import Path

import pytest
import torch

from src.models.gsgp_nn import GSGPNN
from src.training.early_stopping import EarlyStopping
from src.training.callbacks import TrainingLogger, ModelCheckpoint
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


# ── TrainingLogger tests ────────────────────────────────────────────────


class TestTrainingLogger:
    def test_init_defaults(self):
        logger = TrainingLogger()
        assert logger.print_every == 1
        assert logger.window_size == 25

    def test_on_epoch_end_accumulates_history(self):
        logger = TrainingLogger()
        logger.on_epoch_start()
        logger.on_epoch_end(1, 0.5)
        logger.on_epoch_start()
        logger.on_epoch_end(2, 0.3)
        assert len(logger._history) == 2
        assert logger._history == [0.5, 0.3]

    def test_on_epoch_end_no_improvement_window(self):
        """No improvement string when fewer epochs than window_size."""
        logger = TrainingLogger(print_every=1, window_size=10)
        logger.on_epoch_start()
        logger.on_epoch_end(1, 0.5)
        assert len(logger._history) == 1

    def test_logger_custom_every(self):
        logger = TrainingLogger(print_every=5)
        assert logger.print_every == 5


# ── ModelCheckpoint tests ───────────────────────────────────────────────


class TestModelCheckpoint:
    @pytest.fixture
    def model(self):
        return GSGPNN(num_nodes=3, num_layers=1)

    def test_creates_directory(self, tmp_path, model):
        save_dir = str(tmp_path / "checkpoints")
        cp = ModelCheckpoint(save_dir=save_dir)
        assert Path(save_dir).exists()

    def test_saves_on_improvement(self, tmp_path, model):
        cp = ModelCheckpoint(save_dir=str(tmp_path))
        saved = cp.on_epoch_end(1.0, model)
        assert saved is True
        assert cp.save_path.exists()

    def test_does_not_save_worse(self, tmp_path, model):
        cp = ModelCheckpoint(save_dir=str(tmp_path))
        cp.on_epoch_end(1.0, model)  # best = 1.0
        saved = cp.on_epoch_end(2.0, model)  # worse
        assert saved is False

    def test_saves_again_on_improvement(self, tmp_path, model):
        cp = ModelCheckpoint(save_dir=str(tmp_path))
        cp.on_epoch_end(2.0, model)  # best = 2.0
        saved = cp.on_epoch_end(1.0, model)  # better
        assert saved is True
        assert cp.best_loss == 1.0
