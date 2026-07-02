"""GSGP-NN model — Geometric Semantic GP Neural Network.

This module contains the core ``torch.nn.Module`` classes that implement
the GSGP-NN architecture.  All heavy computation is expressed as batched
tensor operations so that it runs efficiently on CUDA GPUs.

Architecture overview
---------------------
::

    initial_semantics (N, M)
            │
            ▼
    ┌─── GSGPNNLayer (k=1) ───┐
    │  inherited + mutation     │  ← parent_semantics, route_semantics[0]
    └──────────┬───────────────┘
               │ (N, M)
               ▼
    ┌─── GSGPNNLayer (k=2) ───┐
    │  inherited + mutation     │  ← parent_semantics, route_semantics[1]
    └──────────┬───────────────┘
               │ (N, M)
               ▼
         nn.Linear(N, 1)  → predictions (M,)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class GSGPNNLayer(nn.Module):
    """A single GSGP-NN semantic layer.

    Each layer computes new semantic vectors for *all* N nodes **in
    parallel** using matrix operations.  This replaces the sequential
    ``for n in range(1, N+1)`` loop from the legacy code.

    The forward computation for each node *n* is::

        inherited[n] = (α[0,n]*parent[n] + Σ_i α[i,n]*prev[i]) / Σ_j |α[j,n]|
        mutation[n]  = β[n] * tanh(ms[n] * (RT1[n] - RT2[n]))
        output[n]    = inherited[n] + mutation[n]

    Learnable parameters:
        - ``alpha`` — shape ``(N+1, N)`` — inheritance weights.
        - ``ms``    — shape ``(N,)``     — mutation scale.
        - ``beta``  — shape ``(N,)``     — mutation coefficient.

    Args:
        num_nodes: Number of semantic nodes (N).
    """

    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        self.N = num_nodes

        # Alpha: row 0 = parent weight, rows 1..N = prev-layer weights
        self.alpha = nn.Parameter(torch.empty(num_nodes + 1, num_nodes))

        # Mutation parameters
        self.ms = nn.Parameter(torch.empty(num_nodes))
        self.beta = nn.Parameter(torch.empty(num_nodes))

        self._init_parameters()

    def _init_parameters(self) -> None:
        """He initialisation for alpha, small normal for ms/beta."""
        nn.init.kaiming_normal_(self.alpha, mode="fan_in", nonlinearity="linear")
        nn.init.normal_(self.ms, mean=0.0, std=0.1)
        nn.init.normal_(self.beta, mean=0.0, std=0.1)

    def forward(
        self,
        prev_semantics: torch.Tensor,
        parent_semantics: torch.Tensor,
        rt1: torch.Tensor,
        rt2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute all N node semantics in one batched operation.

        Args:
            prev_semantics: ``(N, M)`` — output of the previous layer
                (or initial semantics for the first layer).
            parent_semantics: ``(N, M)`` — fixed parent semantic vectors.
            rt1: ``(N, M)`` — random tree 1 semantics (fixed).
            rt2: ``(N, M)`` — random tree 2 semantics (fixed).

        Returns:
            ``(N, M)`` tensor of updated semantics.
        """
        # ── Inherited term ──────────────────────────────────────────
        # Parent contribution:  α[0, :] broadcast over M
        #   α[0] is (N,), parent_semantics is (N, M)
        parent_contrib = self.alpha[0].unsqueeze(1) * parent_semantics  # (N, M)

        # Previous-layer contribution via matrix multiply:
        #   α[1:] is (N_in, N_out), prev is (N_in, M)
        #   result: (N_out, M)
        prev_contrib = torch.mm(self.alpha[1:].t(), prev_semantics)  # (N, M)

        # Numerator
        H = parent_contrib + prev_contrib  # (N, M)

        # Denominator: sum of |α_j| per output node
        d = self.alpha.abs().sum(dim=0).unsqueeze(1)  # (N, 1)
        d = d.clamp(min=1e-12)

        inherited = H / d  # (N, M) with broadcasting

        # ── Mutation term ───────────────────────────────────────────
        rt_diff = rt1 - rt2                                         # (N, M)
        mutation_base = self.ms.unsqueeze(1) * rt_diff               # (N, M)
        mutation = self.beta.unsqueeze(1) * torch.tanh(mutation_base)  # (N, M)

        return inherited + mutation  # (N, M)


class GSGPNN(nn.Module):
    """Geometric Semantic Genetic Programming Neural Network.

    Stacks ``K`` ``GSGPNNLayer`` modules followed by a linear output layer.
    All tensor operations are batched for GPU execution.

    Args:
        num_nodes: Number of semantic nodes per layer (N).
        num_layers: Number of semantic layers (K).
    """

    def __init__(self, num_nodes: int, num_layers: int) -> None:
        super().__init__()
        self.N = num_nodes
        self.K = num_layers

        # Semantic layers
        self.layers = nn.ModuleList(
            [GSGPNNLayer(num_nodes) for _ in range(num_layers)]
        )

        # Linear output: combines the N final semantics into a scalar per sample
        self.output_layer = nn.Linear(num_nodes, 1)

    def forward(
        self,
        initial_semantics: torch.Tensor,
        parent_semantics: torch.Tensor,
        route_semantics: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Full forward pass through K layers + linear output.

        Args:
            initial_semantics: ``(N, M)`` — layer-0 input (from GP evaluation).
            parent_semantics: ``(N, M)`` — fixed parent semantic vectors.
            route_semantics: List of K dicts, each containing ``'rt1'``
                and ``'rt2'`` tensors of shape ``(N, M)``.

        Returns:
            Predictions of shape ``(M,)``.
        """
        current = initial_semantics  # (N, M)

        for k, layer in enumerate(self.layers):
            current = layer(
                prev_semantics=current,
                parent_semantics=parent_semantics,
                rt1=route_semantics[k]["rt1"],
                rt2=route_semantics[k]["rt2"],
            )

        # current is (N, M) → transpose to (M, N) for nn.Linear
        output = self.output_layer(current.t())  # (M, 1)
        return output.squeeze(1)  # (M,)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Return a human-readable model summary."""
        lines = [
            f"GSGPNN(N={self.N}, K={self.K})",
            f"  Trainable parameters: {self.count_parameters():,}",
            f"  Layers:",
        ]
        for i, layer in enumerate(self.layers):
            alpha_shape = tuple(layer.alpha.shape)
            lines.append(
                f"    [{i}] GSGPNNLayer — "
                f"alpha: {alpha_shape}, ms: ({layer.N},), beta: ({layer.N},)"
            )
        out_w = tuple(self.output_layer.weight.shape)
        lines.append(f"    [out] Linear — weight: {out_w}, bias: (1,)")
        return "\n".join(lines)
