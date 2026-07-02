"""Unit tests for the GSGP-NN model (src/models/gsgp_nn.py)."""

import pytest
import torch

from src.models.gsgp_nn import GSGPNN, GSGPNNLayer


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def layer(device):
    """A small GSGPNNLayer for testing."""
    torch.manual_seed(0)
    return GSGPNNLayer(num_nodes=5).to(device)


@pytest.fixture
def model(device):
    """A small GSGPNN model for testing."""
    torch.manual_seed(0)
    return GSGPNN(num_nodes=5, num_layers=2).to(device)


@pytest.fixture
def dummy_data(device):
    """Generate dummy semantic tensors for N=5, M=10."""
    torch.manual_seed(42)
    N, M = 5, 10
    return {
        "initial_semantics": torch.randn(N, M, device=device),
        "parent_semantics": torch.randn(N, M, device=device),
        "route_semantics": [
            {
                "rt1": torch.randn(N, M, device=device),
                "rt2": torch.randn(N, M, device=device),
            }
            for _ in range(2)  # K=2
        ],
    }


# ── GSGPNNLayer tests ──────────────────────────────────────────────────


class TestGSGPNNLayer:
    def test_output_shape(self, layer, device):
        N, M = 5, 10
        prev = torch.randn(N, M, device=device)
        parent = torch.randn(N, M, device=device)
        rt1 = torch.randn(N, M, device=device)
        rt2 = torch.randn(N, M, device=device)

        output = layer(prev, parent, rt1, rt2)
        assert output.shape == (N, M)

    def test_output_finite(self, layer, device):
        N, M = 5, 10
        prev = torch.randn(N, M, device=device)
        parent = torch.randn(N, M, device=device)
        rt1 = torch.randn(N, M, device=device)
        rt2 = torch.randn(N, M, device=device)

        output = layer(prev, parent, rt1, rt2)
        assert torch.isfinite(output).all()

    def test_gradients_flow(self, layer, device):
        N, M = 5, 10
        prev = torch.randn(N, M, device=device, requires_grad=True)
        parent = torch.randn(N, M, device=device)
        rt1 = torch.randn(N, M, device=device)
        rt2 = torch.randn(N, M, device=device)

        output = layer(prev, parent, rt1, rt2)
        loss = output.sum()
        loss.backward()

        assert prev.grad is not None
        assert layer.alpha.grad is not None
        assert layer.ms.grad is not None
        assert layer.beta.grad is not None

    def test_parameter_count(self, layer):
        # alpha: (N+1, N) = 6*5 = 30, ms: 5, beta: 5 → total: 40
        total = sum(p.numel() for p in layer.parameters())
        assert total == 40


# ── GSGPNN tests ───────────────────────────────────────────────────────


class TestGSGPNN:
    def test_output_shape(self, model, dummy_data):
        output = model(**dummy_data)
        M = dummy_data["initial_semantics"].shape[1]
        assert output.shape == (M,)

    def test_output_finite(self, model, dummy_data):
        output = model(**dummy_data)
        assert torch.isfinite(output).all()

    def test_gradients_flow_through_all_layers(self, model, dummy_data):
        output = model(**dummy_data)
        loss = output.mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"

    def test_count_parameters(self, model):
        count = model.count_parameters()
        assert count > 0
        # 2 layers × 40 params + Linear(5, 1) = 80 + 6 = 86
        assert count == 86

    def test_summary_returns_string(self, model):
        s = model.summary()
        assert isinstance(s, str)
        assert "GSGPNN" in s

    def test_deterministic_output(self, model, dummy_data):
        model.eval()
        with torch.no_grad():
            out1 = model(**dummy_data)
            out2 = model(**dummy_data)
        assert torch.allclose(out1, out2)


# ── Edge-case tests ─────────────────────────────────────────────────────


class TestGSGPNNEdgeCases:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_single_node_layer(self, device):
        """N=1, K=1 should work (smallest possible model)."""
        torch.manual_seed(0)
        model = GSGPNN(num_nodes=1, num_layers=1).to(device)
        N, M = 1, 5
        data = {
            "initial_semantics": torch.randn(N, M, device=device),
            "parent_semantics": torch.randn(N, M, device=device),
            "route_semantics": [
                {"rt1": torch.randn(N, M, device=device),
                 "rt2": torch.randn(N, M, device=device)}
            ],
        }
        output = model(**data)
        assert output.shape == (M,)
        assert torch.isfinite(output).all()

    def test_single_pattern(self, device):
        """M=1 should work (single data point)."""
        torch.manual_seed(0)
        model = GSGPNN(num_nodes=5, num_layers=2).to(device)
        N, M = 5, 1
        data = {
            "initial_semantics": torch.randn(N, M, device=device),
            "parent_semantics": torch.randn(N, M, device=device),
            "route_semantics": [
                {"rt1": torch.randn(N, M, device=device),
                 "rt2": torch.randn(N, M, device=device)}
                for _ in range(2)
            ],
        }
        output = model(**data)
        assert output.shape == (M,)
        assert torch.isfinite(output).all()

    def test_state_dict_roundtrip(self, device):
        """save and load state_dict should preserve output."""
        torch.manual_seed(0)
        model = GSGPNN(num_nodes=3, num_layers=1).to(device)
        N, M = 3, 10
        data = {
            "initial_semantics": torch.randn(N, M, device=device),
            "parent_semantics": torch.randn(N, M, device=device),
            "route_semantics": [
                {"rt1": torch.randn(N, M, device=device),
                 "rt2": torch.randn(N, M, device=device)}
            ],
        }
        model.eval()
        with torch.no_grad():
            out_before = model(**data)

        state = model.state_dict()
        model2 = GSGPNN(num_nodes=3, num_layers=1).to(device)
        model2.load_state_dict(state)
        model2.eval()
        with torch.no_grad():
            out_after = model2(**data)

        assert torch.allclose(out_before, out_after)

    def test_gradient_flow_single_layer(self, device):
        """K=1: gradients should flow to all parameters."""
        torch.manual_seed(0)
        model = GSGPNN(num_nodes=4, num_layers=1).to(device)
        N, M = 4, 10
        data = {
            "initial_semantics": torch.randn(N, M, device=device),
            "parent_semantics": torch.randn(N, M, device=device),
            "route_semantics": [
                {"rt1": torch.randn(N, M, device=device),
                 "rt2": torch.randn(N, M, device=device)}
            ],
        }
        output = model(**data)
        loss = output.mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"
