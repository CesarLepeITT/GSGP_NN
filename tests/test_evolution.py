"""Unit tests for the GP evolution components."""

import pytest
import numpy as np
import torch

from src.evolution.nodes import GPNode
from src.evolution.generator import GPGenerator
from src.evolution.semantics import SemanticEvaluator


# ── GPNode tests ────────────────────────────────────────────────────────


class TestGPNode:
    def test_terminal_constant(self):
        node = GPNode(0.5, is_terminal=True)
        assert node.evaluate({}) == 0.5

    def test_terminal_variable(self):
        node = GPNode("x1", is_terminal=True)
        assert node.evaluate({"x1": 3.0}) == 3.0

    def test_terminal_missing_variable(self):
        node = GPNode("x1", is_terminal=True)
        assert node.evaluate({}) == 0.0  # default

    def test_addition(self):
        root = GPNode("+", is_terminal=False, arity=2)
        root.add_child(GPNode(2.0, is_terminal=True))
        root.add_child(GPNode(3.0, is_terminal=True))
        assert root.evaluate({}) == 5.0

    def test_subtraction(self):
        root = GPNode("-", is_terminal=False, arity=2)
        root.add_child(GPNode(5.0, is_terminal=True))
        root.add_child(GPNode(3.0, is_terminal=True))
        assert root.evaluate({}) == 2.0

    def test_multiplication(self):
        root = GPNode("*", is_terminal=False, arity=2)
        root.add_child(GPNode(4.0, is_terminal=True))
        root.add_child(GPNode(3.0, is_terminal=True))
        assert root.evaluate({}) == 12.0

    def test_protected_division_normal(self):
        root = GPNode("/", is_terminal=False, arity=2)
        root.add_child(GPNode(6.0, is_terminal=True))
        root.add_child(GPNode(3.0, is_terminal=True))
        assert root.evaluate({}) == 2.0

    def test_protected_division_by_zero_koza(self):
        root = GPNode("/", is_terminal=False, arity=2)
        root.add_child(GPNode(6.0, is_terminal=True))
        root.add_child(GPNode(0.0, is_terminal=True))
        # Koza: a / sqrt(1 + b*b) → 6.0 / sqrt(1) = 6.0
        assert root.evaluate({}) == 6.0

    def test_protected_division_by_zero_fallback(self):
        root = GPNode("/", is_terminal=False, arity=2)
        root.add_child(GPNode(6.0, is_terminal=True))
        root.add_child(GPNode(0.0, is_terminal=True))
        # Legacy: returns 1.0 on division-by-zero
        assert root.evaluate({}, koza_division=False) == 1.0

    def test_result_clamped(self):
        # Very large multiplication
        root = GPNode("*", is_terminal=False, arity=2)
        root.add_child(GPNode(1e6, is_terminal=True))
        root.add_child(GPNode(1e6, is_terminal=True))
        result = root.evaluate({})
        assert result <= 1e6

    def test_deep_copy(self):
        root = GPNode("+", is_terminal=False, arity=2)
        root.add_child(GPNode("x1", is_terminal=True))
        root.add_child(GPNode(1.0, is_terminal=True))

        copy = root.deep_copy()
        # Modifying copy shouldn't affect original
        copy.children[1].value = 999.0
        assert root.children[1].value == 1.0

    def test_repr(self):
        root = GPNode("+", is_terminal=False, arity=2)
        root.add_child(GPNode("x1", is_terminal=True))
        root.add_child(GPNode(1.0, is_terminal=True))
        s = repr(root)
        assert "+" in s
        assert "x1" in s


# ── GPGenerator tests ──────────────────────────────────────────────────


class TestGPGenerator:
    @pytest.fixture
    def generator(self):
        return GPGenerator(
            function_set=["+", "-", "*", "/"],
            terminal_set=["x1", "x2", "x3"],
            max_depth=3,
        )

    def test_generate_single_tree(self, generator):
        import random
        random.seed(0)
        tree = generator.generate_tree_grow(3)
        assert isinstance(tree, GPNode)

    def test_tree_evaluates_without_error(self, generator):
        import random
        random.seed(0)
        tree = generator.generate_tree_grow(3)
        result = tree.evaluate({"x1": 1.0, "x2": 2.0, "x3": 3.0})
        assert isinstance(result, float)

    def test_generate_population(self, generator):
        import random
        random.seed(0)
        pop = generator.generate_population(20)
        assert len(pop) == 20
        assert all(isinstance(t, GPNode) for t in pop)

    def test_population_diversity(self, generator):
        """Different seeds should produce different trees."""
        import random

        random.seed(0)
        pop1 = generator.generate_population(5)

        random.seed(42)
        pop2 = generator.generate_population(5)

        # At least one tree should differ
        vals1 = [t.evaluate({"x1": 1.0, "x2": 2.0, "x3": 3.0}) for t in pop1]
        vals2 = [t.evaluate({"x1": 1.0, "x2": 2.0, "x3": 3.0}) for t in pop2]
        assert vals1 != vals2


# ── SemanticEvaluator tests ─────────────────────────────────────────────


class TestSemanticEvaluator:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def evaluator(self, device):
        return SemanticEvaluator(
            variable_names=["x1", "x2"],
            device=device,
        )

    @pytest.fixture
    def simple_tree(self):
        root = GPNode("+", is_terminal=False, arity=2)
        root.add_child(GPNode("x1", is_terminal=True))
        root.add_child(GPNode(2.0, is_terminal=True))
        return root

    def test_evaluate_individual_shape(self, evaluator, simple_tree):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = evaluator.evaluate_individual(simple_tree, X)
        assert result.shape == (3,)

    def test_evaluate_individual_device(self, evaluator, simple_tree):
        X = np.array([[1.0, 2.0]])
        result = evaluator.evaluate_individual(simple_tree, X)
        assert result.device.type == "cpu"

    def test_evaluate_individual_values(self, evaluator, simple_tree):
        X = np.array([[3.0, 0.0], [5.0, 0.0]])
        result = evaluator.evaluate_individual(simple_tree, X)
        # tree: x1 + 2.0 → values: 5.0, 7.0
        # After normalisation: mean=6, std=1 → [-1.0, 1.0]
        expected = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_create_semantic_matrix_shape(self, evaluator, simple_tree):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        matrix = evaluator.create_semantic_matrix([simple_tree, simple_tree], X)
        assert matrix.shape == (2, 2)

    def test_normalize_semantics_zero_std(self):
        constant = np.array([5.0, 5.0, 5.0])
        result = SemanticEvaluator._normalize_semantics(constant)
        assert result.shape == (3,)
        # Should produce a deterministic ramp, not crash
        assert len(np.unique(result)) == 3

    def test_normalize_semantics_clipping(self):
        huge = np.array([1e10, -1e10, 0.0])
        result = SemanticEvaluator._normalize_semantics(huge)
        # After clipping to [-1e6, 1e6] and standardising, clamp to [-5, 5]
        assert result.max() <= 5.0
        assert result.min() >= -5.0
