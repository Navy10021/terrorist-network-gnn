"""
Unit tests for baselines.py
"""

import pytest
import torch
import torch.nn as nn

from src.baselines import (
    StaticGAT,
    StaticGCN,
    StaticGraphSAGE,
)


class TestStaticGCN:
    """Test StaticGCN baseline model"""

    @pytest.fixture
    def model(self):
        return StaticGCN(num_node_features=64, hidden_dim=128, num_layers=3, dropout=0.1)

    def test_model_initialization(self, model):
        """Test model initialization"""
        assert isinstance(model, nn.Module)
        assert model.output_dim == 128
        assert len(model.convs) == 3

    def test_forward_pass(self, model):
        """Test forward pass"""
        num_nodes = 10
        num_edges = 15

        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        output = model(x, edge_index)

        assert output.shape == (num_nodes, 128)
        assert not torch.isnan(output).any()

    def test_different_graph_sizes(self, model):
        """Test model with different graph sizes"""
        for num_nodes in [5, 10, 20]:
            x = torch.randn(num_nodes, 64)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))

            output = model(x, edge_index)
            assert output.shape[0] == num_nodes

    def test_gradient_flow(self, model):
        """Test gradient flow"""
        x = torch.randn(10, 64, requires_grad=True)
        edge_index = torch.randint(0, 10, (2, 15))

        output = model(x, edge_index)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestStaticGAT:
    """Test StaticGAT baseline model"""

    @pytest.fixture
    def model(self):
        return StaticGAT(
            num_node_features=64,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
        )

    def test_model_initialization(self, model):
        """Test model initialization"""
        assert isinstance(model, nn.Module)
        assert model.output_dim == 128
        assert len(model.convs) == 3

    def test_forward_pass(self, model):
        """Test forward pass"""
        num_nodes = 10
        num_edges = 15

        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        output = model(x, edge_index)

        assert output.shape == (num_nodes, 128)
        assert not torch.isnan(output).any()

    def test_attention_mechanism(self, model):
        """Test that attention mechanism works"""
        x = torch.randn(10, 64)
        edge_index = torch.randint(0, 10, (2, 15))

        # Should not raise errors
        output = model(x, edge_index)
        assert output.dim() == 2


class TestStaticGraphSAGE:
    """Test StaticGraphSAGE baseline model"""

    @pytest.fixture
    def model(self):
        return StaticGraphSAGE(num_node_features=64, hidden_dim=128, num_layers=3, dropout=0.1)

    def test_model_initialization(self, model):
        """Test model initialization"""
        assert isinstance(model, nn.Module)
        assert model.output_dim == 128
        assert len(model.convs) == 3

    def test_forward_pass(self, model):
        """Test forward pass"""
        num_nodes = 10
        num_edges = 15

        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        output = model(x, edge_index)

        assert output.shape == (num_nodes, 128)
        assert not torch.isnan(output).any()

    def test_sampling_aggregation(self, model):
        """Test GraphSAGE sampling and aggregation"""
        x = torch.randn(20, 64)
        edge_index = torch.randint(0, 20, (2, 40))

        output = model(x, edge_index)

        assert output.shape == (20, 128)


class TestBaselineComparison:
    """Test comparing baseline models"""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing"""
        num_nodes = 15
        num_edges = 25

        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        return x, edge_index

    def test_all_baselines_run(self, sample_graph):
        """Test that all baseline models can process the same graph"""
        x, edge_index = sample_graph

        models = [
            StaticGCN(64, 128, 2),
            StaticGAT(64, 128, 2, num_heads=4),
            StaticGraphSAGE(64, 128, 2),
        ]

        outputs = []
        for model in models:
            output = model(x, edge_index)
            outputs.append(output)

            assert output.shape[0] == x.shape[0]
            assert output.shape[1] == 128

    def test_baseline_embeddings_differ(self, sample_graph):
        """Test that different baselines produce different embeddings"""
        x, edge_index = sample_graph

        gcn = StaticGCN(64, 128, 2)
        gat = StaticGAT(64, 128, 2, num_heads=4)

        gcn_output = gcn(x, edge_index)
        gat_output = gat(x, edge_index)

        # Outputs should be different
        assert not torch.allclose(gcn_output, gat_output)


class TestTrainingCompatibility:
    """Test that baselines are compatible with training"""

    def test_optimizer_compatibility(self):
        """Test that models work with optimizers"""
        model = StaticGCN(64, 128, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.randn(10, 64)
        edge_index = torch.randint(0, 10, (2, 15))

        # Forward pass
        output = model(x, edge_index)
        loss = output.sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters were updated
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_train_eval_modes(self):
        """Test train and eval modes"""
        model = StaticGCN(64, 128, 2, dropout=0.5)

        x = torch.randn(10, 64)
        edge_index = torch.randint(0, 10, (2, 15))

        # Train mode
        model.train()
        output_train = model(x, edge_index)

        # Eval mode
        model.eval()
        output_eval = model(x, edge_index)

        # Outputs might differ due to dropout
        assert output_train.shape == output_eval.shape


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
