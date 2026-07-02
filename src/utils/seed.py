"""Reproducibility utilities for deterministic experiments."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Configures Python's ``random``, NumPy, and PyTorch (CPU + CUDA) to use
    the given seed.  Also enables deterministic cuDNN behaviour, which may
    slightly reduce performance but guarantees reproducible results.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms — may be slower but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # PyTorch >= 1.8: optional stricter determinism
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Older PyTorch version

if __name__ == "__main__":
    print("Testing seed")
    set_seed(42)
    print("Seed set")    
    print("random", random.randint(0, 10))
    print("numpy", np.random.randint(0, 10))
    print("torch", torch.randint(0, 10, (1,)).item())
    set_seed(1)
    print("Seed reset")
    print("random", random.randint(0, 10))
    print("numpy", np.random.randint(0, 10))
    print("torch", torch.randint(0, 10, (1,)).item())
