"""Random GP tree population generator (Ramped Half-and-Half).

Implements Koza's Ramped Half-and-Half method: for each individual a
random depth is chosen uniformly between *min_depth* and *max_depth*;
the tree is then built with either the *Grow* or *Full* method (50 %
chance each), promoting structural and depth diversity.
"""

from __future__ import annotations

import random
from typing import Sequence

from src.evolution.nodes import GPNode

# Binary operators and their arities
_OPERATORS: dict[str, int] = {"+": 2, "-": 2, "*": 2, "/": 2}


class GPGenerator:
    """Generates random GP expression trees using Ramped Half-and-Half.

    Args:
        function_set: Operators to use (subset of ``+, -, *, /``).
        terminal_set: Variable names available as leaves.
        min_depth: Minimum tree depth for generation (inclusive).
        max_depth: Maximum tree depth for generation (inclusive).
        function_prob: Probability of choosing a function node (vs terminal).
        terminal_var_ratio: Probability of choosing a variable (vs constant)
            when a terminal is selected.
        const_min: Lower bound for ephemeral random constants.
        const_max: Upper bound for ephemeral random constants.
    """

    def __init__(
        self,
        function_set: Sequence[str],
        terminal_set: Sequence[str],
        min_depth: int = 1,
        max_depth: int = 3,
        function_prob: float = 0.5,
        terminal_var_ratio: float = 0.7,
        const_min: float = -1.0,
        const_max: float = 1.0,
    ) -> None:
        if not 0 <= function_prob <= 1:
            raise ValueError("function_prob must be in [0, 1]")
        if not 0 <= terminal_var_ratio <= 1:
            raise ValueError("terminal_var_ratio must be in [0, 1]")
        if const_min > const_max:
            raise ValueError("const_min must be <= const_max")

        self.function_set = [op for op in function_set if op in _OPERATORS]
        self.terminal_set = list(terminal_set)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.function_prob = function_prob
        self.terminal_var_ratio = terminal_var_ratio
        self.const_min = const_min
        self.const_max = const_max

    # ------------------------------------------------------------------
    # Node creation helpers
    # ------------------------------------------------------------------

    def _random_constant(self) -> float:
        """Return a random ephemeral constant in [const_min, const_max]."""
        return random.uniform(self.const_min, self.const_max)


    def _create_random_node(self, force_terminal: bool = False) -> GPNode:
        """Create a single random node (terminal or function)."""
        if force_terminal or random.random() > self.function_prob:
            if random.random() < self.terminal_var_ratio and self.terminal_set:
                value = random.choice(self.terminal_set)
            else:
                value = self._random_constant()
            return GPNode(value, is_terminal=True, arity=0)

        op = random.choice(self.function_set)
        return GPNode(op, is_terminal=False, arity=_OPERATORS[op])

    # ------------------------------------------------------------------
    # Tree generation
    # ------------------------------------------------------------------

    def generate_tree_grow(self, max_depth: int) -> GPNode:
        """Generate a single tree using the Grow method.

        Args:
            max_depth: Remaining depth budget.

        Returns:
            Root ``GPNode`` of the generated tree.
        """
        if max_depth <= 0:
            return self._create_random_node(force_terminal=True)

        node = self._create_random_node()
        if node.is_terminal:
            return node

        for _ in range(node.arity):
            child = self.generate_tree_grow(max_depth - 1)
            node.add_child(child)

        return node

    def generate_tree_full(self, depth: int) -> GPNode:
        """Generate a single full tree using the Full method.

        All branches extend to exactly *depth* — every internal node is a
        function and only leaves are terminals.

        Args:
            depth: Remaining depth budget.

        Returns:
            Root ``GPNode`` of the generated tree.
        """
        if depth <= 0:
            return self._create_random_node(force_terminal=True)

        op = random.choice(self.function_set)
        node = GPNode(op, is_terminal=False, arity=_OPERATORS[op])
        for _ in range(node.arity):
            child = self.generate_tree_full(depth - 1)
            node.add_child(child)

        return node

    def generate_population(self, size: int) -> list[GPNode]:
        """Generate a population using Ramped Half-and-Half.

        For each individual a random depth is chosen uniformly in
        ``[min_depth, max_depth]``, and the tree is built with either
        the *Grow* or *Full* method (equal probability).

        Args:
            size: Number of trees to create.

        Returns:
            List of ``GPNode`` root nodes.
        """
        population: list[GPNode] = []
        for _ in range(size):
            depth = random.randint(self.min_depth, self.max_depth)
            if random.random() < 0.5:
                tree = self.generate_tree_grow(depth)
            else:
                tree = self.generate_tree_full(depth)
            population.append(tree)
        return population
