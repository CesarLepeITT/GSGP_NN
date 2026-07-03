"""GP expression tree node.

Each ``GPNode`` represents a single node in a symbolic regression tree.
Terminal nodes hold a variable name or numeric constant; function nodes
hold an operator and have child subtrees.
"""

from __future__ import annotations

import math
from typing import Union


class GPNode:
    """A node in a Genetic Programming expression tree.

    Attributes:
        value: The operator (``'+', '-', '*', '/'``) or terminal
            (variable name or float constant).
        is_terminal: Whether this node is a leaf.
        arity: Number of expected children (0 for terminals, 2 for
            binary operators).
        children: List of child ``GPNode`` instances.
    """

    __slots__ = ("value", "is_terminal", "arity", "children")

    def __init__(
        self,
        value: Union[str, float],
        is_terminal: bool = False,
        arity: int = 0,
    ) -> None:
        self.value = value
        self.is_terminal = is_terminal
        self.arity = arity
        self.children: list[GPNode] = []

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def add_child(self, child: GPNode) -> None:
        """Append *child* if there is room according to ``arity``."""
        if len(self.children) < self.arity:
            self.children.append(child)

    def is_complete(self) -> bool:
        """Return ``True`` when all child slots are filled."""
        return len(self.children) == self.arity

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, variables: dict[str, float], koza_division: bool = True) -> float:
        """Recursively evaluate the subtree given variable assignments.

        When *koza_division* is ``True`` (default) protected division uses
        Koza's formula: ``a / sqrt(1 + b*b)`` when ``b == 0``.  Otherwise
        it returns ``1.0`` on division-by-zero.

        Results are clamped to ``[-1e6, 1e6]`` to prevent blow-up.

        Args:
            variables: Mapping of variable names to their numeric values.
            koza_division: Use Koza's protected division formula.

        Returns:
            Scalar evaluation result.
        """
        if self.is_terminal:
            if isinstance(self.value, (int, float)):
                return float(self.value)
            return float(variables.get(self.value, 0.0))

        try:
            child_values: list[float] = []
            for child in self.children:
                val = child.evaluate(variables, koza_division=koza_division)
                if not math.isfinite(val):
                    val = 0.0
                child_values.append(val)

            a, b = child_values[0], child_values[1]

            if self.value == "+":
                result = a + b
            elif self.value == "-":
                result = a - b
            elif self.value == "*":
                result = a * b
            elif self.value == "/":
                if abs(b) >= 1e-10:
                    result = a / b
                elif koza_division:
                    result = a / math.sqrt(1.0 + b * b)
                else:
                    result = 1.0
            else:
                result = 0.0

            if not math.isfinite(result):
                result = 0.0

            return max(-1e6, min(1e6, result))

        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Copying
    # ------------------------------------------------------------------

    def deep_copy(self) -> GPNode:
        """Return an independent deep copy of this subtree."""
        new_node = GPNode(self.value, self.is_terminal, self.arity)
        for child in self.children:
            new_node.add_child(child.deep_copy())
        return new_node

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def to_sympy(self) -> str:
        """Convert subtree to a sympy-compatible expression string.

        Terminal nodes are rendered as their value (``x1``, ``0.5``, …).
        Internal nodes use standard infix notation with parentheses.
        Protected division is expressed as plain division for symbolic
        extraction.
        """
        if self.is_terminal:
            if isinstance(self.value, float):
                return f"({self.value})"
            return str(self.value)

        left, right = self.children[0].to_sympy(), self.children[1].to_sympy()

        if self.value == "+":
            return f"({left} + {right})"
        if self.value == "-":
            return f"({left} - {right})"
        if self.value == "*":
            return f"({left} * {right})"
        if self.value == "/":
            return f"({left} / {right})"
        return "0"

    def count_nodes(self) -> int:
        """Return the total number of nodes in this subtree."""
        return 1 + sum(c.count_nodes() for c in self.children)

    def __repr__(self) -> str:  # pragma: no cover
        if self.is_terminal:
            return str(self.value)
        child_strs = ", ".join(repr(c) for c in self.children)
        return f"({self.value} {child_strs})"
