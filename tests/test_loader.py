"""Unit tests for the data loader module."""

import numpy as np
import pytest
import torch
import yaml

from src.data.loader import load_config, prepare_semantic_data
from src.evolution.generator import GPGenerator
from src.evolution.semantics import SemanticEvaluator


# ── load_config tests ───────────────────────────────────────────────────


class TestLoadConfig:
    def test_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("no_existe.yaml")

    def test_loads_valid_yaml(self, tmp_path):
        cfg = {"key": "value", "nested": {"a": 1}}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)
        result = load_config(str(path))
        assert result == cfg


# ── prepare_semantic_data tests ─────────────────────────────────────────


class TestPrepareSemanticData:
    @pytest.fixture
    def X(self):
        np.random.seed(0)
        return np.random.randn(20, 3)

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def generator(self):
        return GPGenerator(
            function_set=["+", "-", "*", "/"],
            terminal_set=["x1", "x2", "x3"],
            max_depth=3,
        )

    @pytest.fixture
    def populations(self, generator):
        np.random.seed(0)
        import random
        random.seed(0)
        return generator.generate_population(10), generator.generate_population(10)

    def test_generate_mode_keys(self, X, device):
        result = prepare_semantic_data(
            X=X, num_nodes=4, num_layers=2, population_size=10, device=device
        )
        expected_keys = {
            "initial_semantics", "parent_semantics", "route_semantics",
            "gp_generator", "evaluator", "initial_population", "auxiliary_population",
        }
        assert set(result.keys()) == expected_keys

    def test_generate_mode_shapes(self, X, device):
        N, K, M = 4, 2, 20
        result = prepare_semantic_data(
            X=X, num_nodes=N, num_layers=K, population_size=10, device=device
        )
        assert result["initial_semantics"].shape == (N, M)
        assert result["parent_semantics"].shape == (N, M)
        assert len(result["route_semantics"]) == K
        assert result["route_semantics"][0]["rt1"].shape == (N, M)
        assert result["route_semantics"][0]["rt2"].shape == (N, M)

    def test_generate_mode_generator_not_none(self, X, device):
        result = prepare_semantic_data(
            X=X, num_nodes=4, num_layers=2, population_size=10, device=device
        )
        assert result["gp_generator"] is not None
        assert isinstance(result["gp_generator"], GPGenerator)
        assert isinstance(result["evaluator"], SemanticEvaluator)

    def test_reuse_mode_generator_is_none(self, X, device, populations):
        init_pop, aux_pop = populations
        evaluator = SemanticEvaluator(["x1", "x2", "x3"], device=device)
        result = prepare_semantic_data(
            X=X, num_nodes=4, num_layers=2, population_size=999,
            device=device,
            initial_population=init_pop,
            auxiliary_population=aux_pop,
            evaluator=evaluator,
        )
        assert result["gp_generator"] is None
        assert result["evaluator"] is evaluator

    def test_reuse_mode_shapes(self, X, device, populations):
        init_pop, aux_pop = populations
        N, K, M = 4, 2, 20
        result = prepare_semantic_data(
            X=X, num_nodes=N, num_layers=K, population_size=999,
            device=device,
            initial_population=init_pop,
            auxiliary_population=aux_pop,
            evaluator=SemanticEvaluator(["x1", "x2", "x3"], device=device),
        )
        assert result["initial_semantics"].shape == (N, M)
        assert result["parent_semantics"].shape == (N, M)
        assert len(result["route_semantics"]) == K

    def test_reuse_mode_populations_match(self, X, device, populations):
        init_pop, aux_pop = populations
        result = prepare_semantic_data(
            X=X, num_nodes=4, num_layers=2, population_size=999,
            device=device,
            initial_population=init_pop,
            auxiliary_population=aux_pop,
            evaluator=SemanticEvaluator(["x1", "x2", "x3"], device=device),
        )
        assert result["initial_population"] is init_pop
        assert result["auxiliary_population"] is aux_pop

    def test_reuse_mode_creates_evaluator_if_missing(self, X, device, populations):
        init_pop, aux_pop = populations
        result = prepare_semantic_data(
            X=X, num_nodes=4, num_layers=2, population_size=999,
            device=device,
            initial_population=init_pop,
            auxiliary_population=aux_pop,
        )
        assert result["evaluator"] is not None
        assert isinstance(result["evaluator"], SemanticEvaluator)
