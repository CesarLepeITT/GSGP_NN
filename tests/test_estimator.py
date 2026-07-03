"""Tests for the sklearn-compatible GSGPNNRegressor and SRBench helpers."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from src.evolution.nodes import GPNode
from src.models.gsgp_nn_estimator import (
    GSGPNNRegressor,
    complexity,
    model,
)


# ── GPNode helpers (to_sympy / count_nodes) ─────────────────────────────


class TestGPNodeHelpers:
    def test_to_sympy_terminal_variable(self):
        node = GPNode("x1", is_terminal=True)
        assert node.to_sympy() == "x1"

    def test_to_sympy_terminal_constant(self):
        node = GPNode(0.5, is_terminal=True)
        assert node.to_sympy() == "(0.5)"

    def test_to_sympy_addition(self):
        root = GPNode("+", is_terminal=False, arity=2)
        root.add_child(GPNode("x1", is_terminal=True))
        root.add_child(GPNode(1.0, is_terminal=True))
        expected = "(x1 + (1.0))"
        assert root.to_sympy() == expected

    def test_to_sympy_subtraction(self):
        root = GPNode("-", is_terminal=False, arity=2)
        root.add_child(GPNode("x1", is_terminal=True))
        root.add_child(GPNode("x2", is_terminal=True))
        expected = "(x1 - x2)"
        assert root.to_sympy() == expected

    def test_to_sympy_multiplication(self):
        root = GPNode("*", is_terminal=False, arity=2)
        root.add_child(GPNode(2.0, is_terminal=True))
        root.add_child(GPNode("x1", is_terminal=True))
        expected = "((2.0) * x1)"
        assert root.to_sympy() == expected

    def test_to_sympy_division(self):
        root = GPNode("/", is_terminal=False, arity=2)
        root.add_child(GPNode("x1", is_terminal=True))
        root.add_child(GPNode(2.0, is_terminal=True))
        expected = "(x1 / (2.0))"
        assert root.to_sympy() == expected

    def test_to_sympy_nested(self):
        # (x1 + x2) * (x3 - 1.0)
        inner1 = GPNode("+", is_terminal=False, arity=2)
        inner1.add_child(GPNode("x1", is_terminal=True))
        inner1.add_child(GPNode("x2", is_terminal=True))

        inner2 = GPNode("-", is_terminal=False, arity=2)
        inner2.add_child(GPNode("x3", is_terminal=True))
        inner2.add_child(GPNode(1.0, is_terminal=True))

        root = GPNode("*", is_terminal=False, arity=2)
        root.add_child(inner1)
        root.add_child(inner2)

        expected = "((x1 + x2) * (x3 - (1.0)))"
        assert root.to_sympy() == expected

    def test_count_nodes_terminal(self):
        assert GPNode("x1", is_terminal=True).count_nodes() == 1

    def test_count_nodes_simple(self):
        root = GPNode("+", is_terminal=False, arity=2)
        root.add_child(GPNode("x1", is_terminal=True))
        root.add_child(GPNode(1.0, is_terminal=True))
        assert root.count_nodes() == 3

    def test_count_nodes_nested(self):
        inner = GPNode("+", is_terminal=False, arity=2)
        inner.add_child(GPNode("x1", is_terminal=True))
        inner.add_child(GPNode("x2", is_terminal=True))

        root = GPNode("*", is_terminal=False, arity=2)
        root.add_child(inner)
        root.add_child(GPNode(2.0, is_terminal=True))
        assert root.count_nodes() == 5  # root + inner + x1 + x2 + 2.0


# ── Synthetic data fixture ───────────────────────────────────────────────


@pytest.fixture
def synth_data():
    np.random.seed(0)
    X = np.random.randn(40, 3)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * np.random.randn(40)
    return X, y


# ── GSGPNNRegressor tests ───────────────────────────────────────────────


class TestGSGPNNRegressor:
    def test_fit_returns_self(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        result = est.fit(X, y)
        assert result is est

    def test_fit_populates_attributes(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        assert hasattr(est, "scaler_X_")
        assert hasattr(est, "scaler_y_")
        assert hasattr(est, "gp_generator_")
        assert hasattr(est, "evaluator_")
        assert hasattr(est, "initial_population_")
        assert hasattr(est, "auxiliary_population_")
        assert hasattr(est, "model_")
        assert hasattr(est, "trainer_")
        assert hasattr(est, "history_")
        assert hasattr(est, "best_tree_idx_")
        assert hasattr(est, "device_")

    def test_predict_shape(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        y_pred = est.predict(X)
        assert y_pred.shape == y.shape

    def test_predict_finite(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        y_pred = est.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_score_returns_float(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        r2 = est.score(X, y)
        assert isinstance(r2, float)

    def test_predict_on_new_data(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X[:20], y[:20])
        y_pred = est.predict(X[20:])
        assert y_pred.shape == (20,)
        assert np.all(np.isfinite(y_pred))

    def test_reproducibility_same_seed(self, synth_data):
        X, y = synth_data
        est1 = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=42,
        )
        est2 = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=42,
        )
        est1.fit(X, y)
        est2.fit(X, y)
        y1 = est1.predict(X)
        y2 = est2.predict(X)
        np.testing.assert_allclose(y1, y2, atol=1e-10)

    def test_different_seeds_differ(self, synth_data):
        X, y = synth_data
        est1 = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est2 = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=1,
        )
        est1.fit(X, y)
        est2.fit(X, y)
        y1 = est1.predict(X)
        y2 = est2.predict(X)
        assert not np.allclose(y1, y2, atol=1e-6)

    def test_default_population_size(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=2, epochs=2,
            verbose=False, random_state=0,
        )
        est.fit(X, y)
        expected = 2 * 5 * 2  # 2 * N * K = 20
        assert len(est.initial_population_) == expected

    def test_custom_population_size(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=2, population_size=8,
            epochs=2, verbose=False, random_state=0,
        )
        est.fit(X, y)
        assert len(est.initial_population_) == 8

    def test_history_keys(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        expected_keys = {
            "loss_history", "total_time", "epoch_times",
            "epochs_trained", "final_loss", "converged", "stop_reason",
        }
        assert set(est.history_.keys()) == expected_keys

    def test_stop_reason_completed(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=10, verbose=False, random_state=0,
        )
        est.fit(X, y)
        assert est.history_["stop_reason"] == "completed"
        assert est.history_["epochs_trained"] == 10

    def test_stop_reason_max_time(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=200, max_time=0.5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        assert est.history_["stop_reason"] == "max_time"
        assert est.history_["epochs_trained"] < 200


# ── SRBench-required functions ──────────────────────────────────────────


class TestSRBenchFunctions:
    def test_model_returns_string(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        expr = model(est)
        assert isinstance(expr, str)
        assert len(expr) > 0

    def test_model_contains_variables(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        expr = model(est)
        assert "x1" in expr or "x2" in expr or "x3" in expr

    def test_model_accepts_optional_X(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        expr_none = model(est, X=None)
        expr_data = model(est, X=X)
        assert isinstance(expr_none, str)
        assert expr_none == expr_data

    def test_complexity_returns_int(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        c = complexity(est)
        assert isinstance(c, int)
        assert c > 0

    def test_complexity_reasonable(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        c = complexity(est)
        assert 1 <= c <= 100  # trees shouldn't be huge with max_depth=10

    def test_best_tree_property(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        tree = est.best_tree_
        assert tree is est.initial_population_[est.best_tree_idx_]


# ── Edge-case tests ──────────────────────────────────────────────────────


class TestEstimatorEdgeCases:
    def test_single_feature(self):
        np.random.seed(0)
        X = np.random.randn(30, 1)
        y = 3.0 * X[:, 0] + 0.3 * np.random.randn(30)
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        y_pred = est.predict(X)
        assert y_pred.shape == y.shape

    def test_many_features(self):
        np.random.seed(0)
        X = np.random.randn(50, 10)
        y = X[:, 0] + X[:, 5]
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        y_pred = est.predict(X)
        assert y_pred.shape == y.shape

    def test_small_data(self):
        np.random.seed(0)
        X = np.random.randn(10, 2)
        y = X[:, 0] + X[:, 1]
        est = GSGPNNRegressor(
            num_nodes=4, num_layers=1, population_size=8,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X, y)
        y_pred = est.predict(X)
        assert y_pred.shape == y.shape

    def test_with_pandas_input(self, synth_data):
        pd = pytest.importorskip("pandas")
        X, y = synth_data
        X_df = pd.DataFrame(X, columns=["a", "b", "c"])
        y_ser = pd.Series(y)

        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=5, verbose=False, random_state=0,
        )
        est.fit(X_df, y_ser)
        y_pred = est.predict(X_df)
        assert y_pred.shape == y_ser.shape

    def test_get_params(self):
        est = GSGPNNRegressor(num_nodes=30, epochs=100)
        params = est.get_params()
        assert params["num_nodes"] == 30
        assert params["epochs"] == 100
        assert params["num_layers"] == 2

    def test_set_params(self):
        est = GSGPNNRegressor()
        est.set_params(num_nodes=20, num_layers=3)
        assert est.num_nodes == 20
        assert est.num_layers == 3

    def test_repr_html_works(self):
        """Ensure the estimator string representation is valid."""
        est = GSGPNNRegressor(num_nodes=15, num_layers=2, random_state=42)
        s = repr(est)
        assert "GSGPNNRegressor" in s

    def test_device_cpu_works(self, synth_data):
        X, y = synth_data
        est = GSGPNNRegressor(
            num_nodes=5, num_layers=1, population_size=10,
            epochs=3, verbose=False, random_state=0, device="cpu",
        )
        est.fit(X, y)
        assert str(est.device_) == "cpu"
        y_pred = est.predict(X)
        assert np.all(np.isfinite(y_pred))
