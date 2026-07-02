"""CUDA device management utilities."""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available compute device.

    Args:
        prefer_cuda: If True, use CUDA when available.

    Returns:
        A ``torch.device`` pointing to the selected backend.
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"[Device] Using GPU: {props.name}")
        print(f"[Device] VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"[Device] CUDA Capability: {props.major}.{props.minor}")
    else:
        device = torch.device("cpu")
        if prefer_cuda and not torch.cuda.is_available():
            print("[Device] CUDA requested but not available — falling back to CPU")
        else:
            print("[Device] Using CPU")
    return device


def estimate_memory_usage_mb(
    num_nodes: int, num_layers: int, num_patterns: int
) -> float:
    """Estimate GPU memory usage in MB for a given configuration.

    This is a rough lower-bound estimate covering model parameters and the
    semantic tensors that must be resident during training.

    Args:
        num_nodes: Number of semantic nodes (N).
        num_layers: Number of semantic layers (K).
        num_patterns: Number of training patterns (M).

    Returns:
        Estimated memory in megabytes.
    """
    bytes_per_float = 4  # float32

    # Parameters per layer: alpha (N+1, N) + ms (N,) + beta (N,)
    params_per_layer = (num_nodes + 1) * num_nodes + 2 * num_nodes
    total_params = params_per_layer * num_layers + num_nodes + 1  # + output layer

    # Semantic tensors: initial (N, M), parent (N, M), rt1/rt2 per layer (N, M each)
    semantic_tensors = (2 + 2 * num_layers) * num_nodes * num_patterns

    # Gradient storage ≈ same as params + intermediate activations
    gradient_storage = total_params + num_layers * num_nodes * num_patterns

    total_bytes = (total_params + semantic_tensors + gradient_storage) * bytes_per_float
    return total_bytes / (1024 * 1024)

if __name__ == "__main__":
    print("Testing device")
    print(get_device())
    print("Testing memory usage estimation")
    print(estimate_memory_usage_mb(num_nodes=100, num_layers=5, num_patterns=10000))    