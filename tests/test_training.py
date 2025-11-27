"""
Unit tests for training.py
"""

import pytest
import torch
import torch.nn as nn

from src.advanced_tgnn import AdvancedTemporalGNN
from src.training import (
    ContrastiveLoss,
    EnhancedTemporalGNNTrainer,
    GraphReconstructionLoss,
    NodeReconstructionLoss,
    TemporalAutoencoderLoss,
    TemporalLinkPredictionLoss,
)


class TestTemporalLinkPredictionLoss:
    """Test TemporalLinkPredictionLoss"""

    @pytest.fixture
    def loss_fn(self):
        return TemporalLinkPredictionLoss(negative_sampling_ratio=1.0)

    def test_loss_initialization(self, loss_fn):
        """Test loss function initializes correctly"""
        assert isinstance(loss_fn, nn.Module)
        assert loss_fn.negative_sampling_ratio == 1.0

    def test_loss_forward(self, loss_fn):
        """Test loss forward pass"""
        num_nodes = 10
        hidden_dim = 64
        num_edges = 5

        embeddings = torch.randn(num_nodes, hidden_dim)
        pos_edges = torch.randint(0, num_nodes, (2, num_edges))
        neg_edges = torch.randint(0, num_nodes, (2, num_edges))

        loss = loss_fn(embeddings, pos_edges, neg_edges)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative

    def test_loss_gradient_flow(self, loss_fn):
        """Test that gradients flow through loss"""
        embeddings = torch.randn(10, 64, requires_grad=True)
        pos_edges = torch.randint(0, 10, (2, 5))
        neg_edges = torch.randint(0, 10, (2, 5))

        loss = loss_fn(embeddings, pos_edges, neg_edges)
        loss.backward()

        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()


class TestContrastiveLoss:
    """Test ContrastiveLoss"""

    @pytest.fixture
    def loss_fn(self):
        return ContrastiveLoss(temperature=0.5)

    def test_loss_initialization(self, loss_fn):
        """Test loss function initializes correctly"""
        assert isinstance(loss_fn, nn.Module)
        assert loss_fn.temperature == 0.5

    def test_loss_forward(self, loss_fn):
        """Test loss forward pass with equal sized embeddings"""
        num_nodes = 10
        hidden_dim = 64

        embeddings_t = torch.randn(num_nodes, hidden_dim)
        embeddings_t1 = torch.randn(num_nodes, hidden_dim)

        loss = loss_fn(embeddings_t, embeddings_t1)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_different_sizes(self, loss_fn):
        """Test loss handles different sized embeddings"""
        embeddings_t = torch.randn(10, 64)
        embeddings_t1 = torch.randn(8, 64)  # Different size

        loss = loss_fn(embeddings_t, embeddings_t1)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()


class TestNodeReconstructionLoss:
    """Test NodeReconstructionLoss"""

    @pytest.fixture
    def loss_fn(self):
        return NodeReconstructionLoss()

    def test_loss_without_mask(self, loss_fn):
        """Test loss without mask"""
        num_nodes = 10
        num_features = 64

        reconstructed = torch.randn(num_nodes, num_features)
        original = torch.randn(num_nodes, num_features)

        loss = loss_fn(reconstructed, original)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_with_mask(self, loss_fn):
        """Test loss with mask"""
        num_nodes = 10
        num_features = 64

        reconstructed = torch.randn(num_nodes, num_features)
        original = torch.randn(num_nodes, num_features)
        mask = torch.randint(0, 2, (num_nodes,)).bool()

        loss = loss_fn(reconstructed, original, mask)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()


class TestTemporalAutoencoderLoss:
    """Test TemporalAutoencoderLoss"""

    @pytest.fixture
    def loss_fn(self):
        return TemporalAutoencoderLoss(hidden_dim=64)

    def test_loss_initialization(self, loss_fn):
        """Test loss function has predictor"""
        assert hasattr(loss_fn, "predictor")
        assert isinstance(loss_fn.predictor, nn.Sequential)

    def test_loss_forward(self, loss_fn):
        """Test temporal prediction loss"""
        embeddings_t = torch.randn(10, 64)
        embeddings_t1 = torch.randn(10, 64)

        loss = loss_fn(embeddings_t, embeddings_t1)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_different_timesteps(self, loss_fn):
        """Test loss with different sized timesteps"""
        embeddings_t = torch.randn(10, 64)
        embeddings_t1 = torch.randn(8, 64)

        loss = loss_fn(embeddings_t, embeddings_t1)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()


class TestGraphReconstructionLoss:
    """Test GraphReconstructionLoss"""

    @pytest.fixture
    def loss_fn(self):
        return GraphReconstructionLoss(margin=1.0)

    def test_loss_initialization(self, loss_fn):
        """Test loss function initializes correctly"""
        assert isinstance(loss_fn, nn.Module)
        assert loss_fn.margin == 1.0

    def test_loss_forward(self, loss_fn):
        """Test graph reconstruction loss"""
        num_nodes = 10
        hidden_dim = 64

        embeddings = torch.randn(num_nodes, hidden_dim)
        adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()

        loss = loss_fn(embeddings, adj_matrix)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestEnhancedTemporalGNNTrainer:
    """Test EnhancedTemporalGNNTrainer"""

    @pytest.fixture
    def model(self):
        """Create a model for training tests"""
        return AdvancedTemporalGNN(
            num_node_features=64,
            num_edge_features=32,
            hidden_dim=128,
            num_temporal_layers=2,
            num_graph_layers=2,
            num_attention_heads=4,
            memory_size=50,
            dropout=0.1,
        )

    @pytest.fixture
    def trainer(self, model):
        """Create trainer instance"""
        return EnhancedTemporalGNNTrainer(
            model=model,
            learning_rate=0.001,
            weight_decay=1e-5,
            device="cpu",
        )

    def test_trainer_initialization(self, trainer, model):
        """Test trainer initializes correctly"""
        assert trainer.model == model
        assert trainer.device == torch.device("cpu")
        assert hasattr(trainer, "optimizer")

    def test_trainer_has_loss_functions(self, trainer):
        """Test trainer has all required loss functions"""
        assert hasattr(trainer, "link_prediction_loss")
        assert hasattr(trainer, "contrastive_loss")
        assert hasattr(trainer, "temporal_autoencoder_loss")

    @pytest.mark.slow
    def test_training_step(self, trainer):
        """Test single training step"""
        # Create sample batch
        batch_size = 2
        num_nodes = 10
        num_edges = 15

        node_features = torch.randn(batch_size, num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (batch_size, 2, num_edges))
        edge_attr = torch.randn(batch_size, num_edges, 32)
        timestamps = torch.arange(batch_size).float()

        # Training step
        loss = trainer.train_step(node_features, edge_index, edge_attr, timestamps)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_model_parameters_update(self, trainer, model):
        """Test that model parameters are updated during training"""
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Create sample batch
        node_features = torch.randn(2, 10, 64)
        edge_index = torch.randint(0, 10, (2, 2, 15))
        edge_attr = torch.randn(2, 15, 32)
        timestamps = torch.arange(2).float()

        # Training step
        trainer.train_step(node_features, edge_index, edge_attr, timestamps)

        # Check that at least some parameters changed
        params_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current):
                params_changed = True
                break

        assert params_changed, "Model parameters should be updated during training"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
