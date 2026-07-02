"""Random GP tree population generator.

Uses the *Grow* method: at each depth level a function or terminal is
chosen randomly, with terminals forced at the maximum depth.
"""

from __future__ import annotations

import random
from typing import Sequence

from src.evolution.nodes import GPNode

# Binary operators and their arities
_OPERATORS: dict[str, int] = {"+": 2, "-": 2, "*": 2, "/": 2}


class GPGenerator:
    """Generates random GP expression trees using the Grow method.

    Args:
        function_set: Operators to use (subset of ``+, -, *, /``).
        terminal_set: Variable names available as leaves.
        max_depth: Maximum tree depth for generation.
    """

    def __init__(
        self,
        function_set: Sequence[str],
        terminal_set: Sequence[str],
        max_depth: int = 3,
    ) -> None:
        self.function_set = [op for op in function_set if op in _OPERATORS]
        self.terminal_set = list(terminal_set)
        self.max_depth = max_depth

    # ------------------------------------------------------------------
    # Node creation helpers
    # ------------------------------------------------------------------

    @staticmethod
    # TODO: Revisar si las constantes tienen que estar restringidas de [-1, 1] o de [MIN_CONST, MAX_CONST] y su distribución
    def _random_constant() -> float:
        """Return a random ephemeral constant in [-1, 1]."""
        return round(random.uniform(-1.0, 1.0), 3)

    # TODO: Revisar la proporción 70 % variable, 30 % constante
    def _create_random_node(self, force_terminal: bool = False) -> GPNode:
        """Create a single random node (terminal or function)."""
        if force_terminal or random.random() < 0.5:
            # Terminal: 70 % variable, 30 % constant
            if random.random() < 0.7 and self.terminal_set:
                value = random.choice(self.terminal_set)
            else:
                value = self._random_constant()
            return GPNode(value, is_terminal=True, arity=0)

        # Function node
        op = random.choice(self.function_set)
        return GPNode(op, is_terminal=False, arity=_OPERATORS[op])

    # ------------------------------------------------------------------
    # Tree generation
    # ------------------------------------------------------------------

    # TODO: Revisar el metodo para generar el arbol (Grow) 
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

    # TODO: Revisar la lógica de generacion de poblacion y profundidad
    def generate_population(self, size: int) -> list[GPNode]:
        """Generate a population of GP trees with varied depths.

        Depths cycle through ``[1, 2, 2, 3, 3, 3]`` to promote
        diversity, matching the legacy ``generar_poblacion_grow`` logic.

        Args:
            size: Number of trees to create.

        Returns:
            List of ``GPNode`` root nodes.
        """
        depth_schedule = [1, 2, 2, 3, 3, 3] # TODO: Esto deberia de ser un hiperparametro min_depth a max_depth
        #  y ademas se deberia de repartir la poblacion en esas profundidades de manera uniforme con el metodo de Rampend Half and half
        population: list[GPNode] = []
        for i in range(size):
            depth = depth_schedule[i % len(depth_schedule)]
            tree = self.generate_tree_grow(depth)
            population.append(tree)
        return population
