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

    def evaluate(self, variables: dict[str, float]) -> float:
        """Recursively evaluate the subtree given variable assignments.

        Protected division returns 1.0 when the divisor is near zero.
        Results are clamped to [-1e6, 1e6] to prevent blow-up.

        Args:
            variables: Mapping of variable names to their numeric values.

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
                val = child.evaluate(variables)
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

            # TODO: Implementar la división protegida de la siguiente manera
            # if (tmp!=0) {
            #        pushGenes[tid] = push(tmp2 / tmp,pushGenes,stackInd);
            #        out = tmp2 / tmp;
            # }else {
            #        pushGenes[tid] = push(tmp2 / sqrtf(1+tmp*tmp),pushGenes,stackInd);
            #        out = tmp2 / sqrtf(1+tmp*tmp);

            elif self.value == "/":
                result = a / b if abs(b) >= 1e-10 else 1.0
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

    def __repr__(self) -> str:  # pragma: no cover
        if self.is_terminal:
            return str(self.value)
        child_strs = ", ".join(repr(c) for c in self.children)
        return f"({self.value} {child_strs})"
