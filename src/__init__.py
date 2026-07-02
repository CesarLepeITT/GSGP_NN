"""GSGP-NN: Geometric Semantic Genetic Programming Neural Network.

A hybrid model combining GP-derived semantic features with
neural network layers, accelerated with PyTorch + CUDA.
"""

from src.models.gsgp_nn import GSGPNN, GSGPNNLayer

__all__ = ["GSGPNN", "GSGPNNLayer"]
