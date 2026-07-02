"""Semantic data preparation and config loading.

Bridges the GP evolution (CPU) and neural network (GPU) worlds by
generating GP populations, evaluating their semantics, and organising
the results into the tensor structure consumed by ``GSGPNN.forward()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.evolution.generator import GPGenerator
from src.evolution.nodes import GPNode
from src.evolution.semantics import SemanticEvaluator


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load experiment configuration from a YAML file.

    Args:
        config_path: Path to the YAML config.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# TODO: Paralelizar esto
def prepare_semantic_data(
    X: np.ndarray,
    num_nodes: int,
    num_layers: int,
    population_size: int,
    device: torch.device,
    initial_population: list[GPNode] | None = None,
    auxiliary_population: list[GPNode] | None = None,
    evaluator: SemanticEvaluator | None = None,
) -> dict[str, Any]:
    """Generate GP populations, evaluate semantics, and pack into tensors.

    This is the one-time initialisation step that runs on CPU (GP tree
    evaluation) and then moves the resulting tensors to the target device
    for GPU-accelerated training.

    When ``initial_population`` and ``auxiliary_population`` are provided,
    the function skips tree generation and re-evaluates the given populations
    on *X* (useful for producing test-set semantics from the same GP trees
    used during training).  ``population_size`` is ignored in this case.

    Args:
        X: Feature matrix of shape ``(M, n_features)``.
        num_nodes: Number of semantic nodes ``N``.
        num_layers: Number of semantic layers ``K``.
        population_size: Size of each GP population (typically ``2*N*K``).
        device: Target device for output tensors.
        initial_population: Optional pre-generated initial population.
        auxiliary_population: Optional pre-generated auxiliary population.
        evaluator: Optional pre-created evaluator (ignored when
            ``initial_population`` is ``None``).

    Returns:
        Dictionary with the following keys:

        - ``initial_semantics``: ``(N, M)`` tensor — layer-0 input.
        - ``parent_semantics``: ``(N, M)`` tensor — fixed parent features.
        - ``route_semantics``: list of K dicts, each with ``rt1`` and ``rt2``
          tensors of shape ``(N, M)`` — random tree pairs for mutation.
        - ``gp_generator``: The ``GPGenerator`` instance (``None`` when
          populations were provided).
        - ``evaluator``: The ``SemanticEvaluator`` instance.
        - ``initial_population``: List of GP trees for the initial population.
        - ``auxiliary_population``: List of GP trees for the auxiliary population.
    """
    num_features = X.shape[1]
    variable_names = [f"x{i + 1}" for i in range(num_features)]

    if initial_population is not None and auxiliary_population is not None:
        generator = None
        print("[GP] Using provided populations (skipping generation)...")
        if evaluator is None:
            evaluator = SemanticEvaluator(variable_names, device=device)
    else:
        # --- GP tree generation (CPU) ------------------------------------
        generator = GPGenerator(
            function_set=["+", "-", "*", "/"],
            terminal_set=variable_names,
            max_depth=10,
        )

        print(f"[GP] Generating populations of size {population_size}...")
        initial_population = generator.generate_population(population_size)
        auxiliary_population = generator.generate_population(population_size)
        evaluator = SemanticEvaluator(variable_names, device=device)

    # --- Semantic evaluation (CPU → device) ---------------------------
    print("[GP] Evaluating initial population semantics...")
    initial_matrix = evaluator.create_semantic_matrix(initial_population, X)
    print("[GP] Evaluating auxiliary population semantics...")
    auxiliary_matrix = evaluator.create_semantic_matrix(auxiliary_population, X)

    M = X.shape[0]

    # --- Parent semantics: one per node, cycling through initial pop ---
    parent_rows = []
    for n in range(num_nodes):
        idx = n % initial_matrix.shape[0]
        parent_rows.append(initial_matrix[idx])
    parent_semantics = torch.stack(parent_rows, dim=0)  # (N, M)

    # --- Initial (layer-0) semantics: cycling through combined pops ---
    combined_matrix = torch.cat(
        [initial_matrix, auxiliary_matrix], dim=0
    )  # (2*pop_size, M)
    init_rows = []
    for n in range(num_nodes):
        idx = n % combined_matrix.shape[0]
        init_rows.append(combined_matrix[idx])
    initial_semantics = torch.stack(init_rows, dim=0)  # (N, M)

    # --- Route semantics: RT1, RT2 pairs per layer per node -----------
    route_semantics: list[dict[str, torch.Tensor]] = []
    aux_idx = 0
    for _k in range(num_layers):
        rt1_rows = []
        rt2_rows = []
        for _n in range(num_nodes):
            rt1_idx = aux_idx % auxiliary_matrix.shape[0]
            rt1_rows.append(auxiliary_matrix[rt1_idx])
            aux_idx += 1

            rt2_idx = aux_idx % auxiliary_matrix.shape[0]
            rt2_rows.append(auxiliary_matrix[rt2_idx])
            aux_idx += 1

        route_semantics.append(
            {
                "rt1": torch.stack(rt1_rows, dim=0),  # (N, M)
                "rt2": torch.stack(rt2_rows, dim=0),  # (N, M)
            }
        )

    print(
        f"[GP] Semantic tensors ready — "
        f"initial: {initial_semantics.shape}, "
        f"parent: {parent_semantics.shape}, "
        f"routes: {num_layers} layers × (N={num_nodes}, M={M})"
    )

    return {
        "initial_semantics": initial_semantics,
        "parent_semantics": parent_semantics,
        "route_semantics": route_semantics,
        "gp_generator": generator,
        "evaluator": evaluator,
        "initial_population": initial_population,
        "auxiliary_population": auxiliary_population,
    }
