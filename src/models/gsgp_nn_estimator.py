"""scikit-learn compatible GSGP-NN regressor.

Exposes a :class:`GSGPNNRegressor` that follows the ``BaseEstimator`` /
``RegressorMixin`` contract and provides the ``model()`` and
``complexity()`` functions required by the SRBench evaluation framework.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from src.data.loader import prepare_semantic_data
from src.evolution.generator import GPGenerator
from src.evolution.semantics import SemanticEvaluator
from src.models.gsgp_nn import GSGPNN
from src.training.callbacks import ModelCheckpoint, TrainingLogger
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer
from src.utils.device import get_device
from src.utils.seed import set_seed


class GSGPNNRegressor(BaseEstimator, RegressorMixin):
    """Geometric Semantic GP Neural Network — sklearn interface.

    Parameters
    ----------
    num_nodes : int, default=50
        Number of semantic nodes (N) per layer.
    num_layers : int, default=2
        Number of semantic layers (K).
    population_size : int or None, default=None
        Size of each GP population.  If ``None`` defaults to ``2 * N * K``.
    learning_rate : float, default=0.005
        Adam learning rate.
    epochs : int, default=200
        Maximum number of training epochs.
    max_time : float or None, default=None
        Maximum wall-clock time in **seconds** for training.  The loop
        stops after the first epoch that exceeds this limit.
    grad_clip_norm : float, default=5.0
        Maximum gradient norm for clipping.
    weight_decay : float, default=0.0
        L2 weight regularisation coefficient.
    early_stopping_patience : int, default=150
        Patience for early stopping.
    early_stopping_warmup : int, default=30
        Warm-up epochs before early stopping activates.
    early_stopping_min_delta : float, default=1e-6
        Minimum loss decrease to count as improvement.
    device : str, default="auto"
        Compute device: ``"auto"``, ``"cuda"``, or ``"cpu"``.
    verbose : bool, default=True
        Whether to print per-epoch logs.

    GP generator parameters
    -----------------------
    function_set : tuple of str, default=("+", "-", "*", "/")
        Operators available for GP tree construction.
    min_depth : int, default=1
        Minimum tree depth for Ramped Half-and-Half.
    max_depth : int, default=10
        Maximum tree depth for Ramped Half-and-Half.
    function_prob : float, default=0.5
        Probability of choosing a function node (vs terminal).
    terminal_var_ratio : float, default=0.7
        Probability of choosing a variable (vs constant) when a terminal
        is selected.
    const_min : float, default=-1.0
        Lower bound for ephemeral random constants.
    const_max : float, default=1.0
        Upper bound for ephemeral random constants.
    koza_division : bool, default=True
        Use Koza's protected division formula.

    Reproducibility
    ---------------
    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes (fitted)
    -------------------
    scaler_X_ : StandardScaler
        Fitted feature scaler.
    scaler_y_ : StandardScaler
        Fitted target scaler.
    gp_generator_ : GPGenerator
        GP tree generator instance.
    evaluator_ : SemanticEvaluator
        Semantic evaluator instance.
    initial_population_ : list[GPNode]
        GP trees for the initial population.
    auxiliary_population_ : list[GPNode]
        GP trees for the auxiliary population.
    model_ : GSGPNN
        Trained PyTorch model.
    trainer_ : Trainer
        Trainer instance used for fitting.
    history_ : dict
        Training history returned by ``Trainer.fit()``.
    best_tree_idx_ : int
        Index of the initial-population tree with the highest estimated
        influence on the final model output.
    device_ : torch.device
        Device used for computation.
    """

    def __init__(
        self,
        num_nodes: int = 50,
        num_layers: int = 2,
        population_size: int | None = None,
        learning_rate: float = 0.005,
        epochs: int = 200,
        max_time: float | None = None,
        grad_clip_norm: float = 5.0,
        weight_decay: float = 0.0,
        early_stopping_patience: int = 150,
        early_stopping_warmup: int = 30,
        early_stopping_min_delta: float = 1e-6,
        device: str = "auto",
        verbose: bool = True,
        function_set: tuple[str, ...] = ("+", "-", "*", "/"),
        min_depth: int = 1,
        max_depth: int = 10,
        function_prob: float = 0.5,
        terminal_var_ratio: float = 0.7,
        const_min: float = -1.0,
        const_max: float = 1.0,
        koza_division: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_time = max_time
        self.grad_clip_norm = grad_clip_norm
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_warmup = early_stopping_warmup
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = device
        self.verbose = verbose
        self.function_set = function_set
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.function_prob = function_prob
        self.terminal_var_ratio = terminal_var_ratio
        self.const_min = const_min
        self.const_max = const_max
        self.koza_division = koza_division
        self.random_state = random_state

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> GSGPNNRegressor:
        """Fit the GSGP-NN model to the training data.

        Args:
            X: Feature matrix of shape ``(M, n_features)``.
            y: Target vector of shape ``(M,)``.

        Returns:
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if self.random_state is not None:
            set_seed(self.random_state)

        self.device_ = self._resolve_device()

        # Scale
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)
        y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()

        pop_size = self.population_size or (2 * self.num_nodes * self.num_layers)

        # GP generation
        variable_names = [f"x{i + 1}" for i in range(X.shape[1])]
        self.gp_generator_ = GPGenerator(
            function_set=list(self.function_set),
            terminal_set=variable_names,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            function_prob=self.function_prob,
            terminal_var_ratio=self.terminal_var_ratio,
            const_min=self.const_min,
            const_max=self.const_max,
        )

        train_semantic = prepare_semantic_data(
            X=X_scaled,
            num_nodes=self.num_nodes,
            num_layers=self.num_layers,
            population_size=pop_size,
            device=self.device_,
            koza_division=self.koza_division,
        )

        self.initial_population_ = train_semantic["initial_population"]
        self.auxiliary_population_ = train_semantic["auxiliary_population"]
        self.evaluator_ = train_semantic["evaluator"]

        # Model
        model = GSGPNN(
            num_nodes=self.num_nodes,
            num_layers=self.num_layers,
        ).to(self.device_)

        y_tensor = torch.tensor(y_scaled, dtype=torch.float64, device=self.device_)

        # Trainer
        print_every = 1 if self.verbose else 50
        trainer = Trainer(
            model=model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
            early_stopping=EarlyStopping(
                patience=self.early_stopping_patience,
                warmup_epochs=self.early_stopping_warmup,
                min_delta=self.early_stopping_min_delta,
            ),
            logger=TrainingLogger(print_every=print_every),
        )

        # Train
        self.history_ = trainer.fit(
            train_semantic, y_tensor, epochs=self.epochs, max_time=self.max_time
        )

        self.model_ = model
        self.trainer_ = trainer

        # Compute best tree
        self.best_tree_idx_ = self._compute_best_tree()

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data.

        Args:
            X: Feature matrix of shape ``(M, n_features)``.

        Returns:
            Predictions of shape ``(M,)``.
        """
        X = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_X_.transform(X)

        test_sem = prepare_semantic_data(
            X=X_scaled,
            num_nodes=self.num_nodes,
            num_layers=self.num_layers,
            population_size=0,
            device=self.device_,
            initial_population=self.initial_population_,
            auxiliary_population=self.auxiliary_population_,
            evaluator=self.evaluator_,
            koza_division=self.koza_division,
        )

        y_pred_scaled = self.trainer_.predict(test_sem)
        y_pred = y_pred_scaled.cpu().numpy()
        y_pred = self.scaler_y_.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_pred

    # ------------------------------------------------------------------
    # Best-tree extraction helpers
    # ------------------------------------------------------------------
    
    def _compute_best_tree(self) -> int:
        """Return the index of the initial-population tree with the
        highest linearised influence on the model output.

        We propagate the output-layer weights backward through each layer
        using the trained ``alpha`` matrices, ignoring the non-linear
        mutation term.
        """
        with torch.no_grad():
            importance = self.model_.output_layer.weight.data.squeeze(0)

            for layer in reversed(self.model_.layers):
                alpha_prev = layer.alpha.data[1:]  # (N, N)
                d = layer.alpha.data.abs().sum(dim=0).clamp(min=1e-12)
                importance = alpha_prev @ (importance / d)

            best_idx = importance.abs().argmax().item()
        return best_idx

    @property
    def best_tree_(self):
        """The GP tree from the initial population with the highest
        estimated influence on the model output."""
        return self.initial_population_[self.best_tree_idx_]

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return get_device(prefer_cuda=True)
        if self.device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            print("[Device] CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")


# ======================================================================
# SRBench-required functions
# ======================================================================

# FIX: No implementado
def model(est: GSGPNNRegressor, X: np.ndarray | None = None) -> str:
    """Return a sympy-compatible expression string of the discovered model.

    For GSGP-NN this is the best GP tree from the initial population,
    selected by its linearised influence on the final output.

    Args:
        est: Fitted :class:`GSGPNNRegressor`.
        X: Optional feature matrix (ignored for tree-based extraction).

    Returns:
        Sympy expression string.
    """

    return est.best_tree_.to_sympy()

# Fix no implementado
def complexity(est: GSGPNNRegressor) -> int:
    """Return the node count of the extracted symbolic expression.

    Args:
        est: Fitted :class:`GSGPNNRegressor`.

    Returns:
        Number of nodes in the best GP tree.
    """
    return est.best_tree_.count_nodes()
