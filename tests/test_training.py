"""
Test Suite for Training Framework
==================================

Tests for self-supervised learning, loss functions, and training utilities.

Author: Yoon-Seop Lee
"""

import pytest
import torch
import torch.nn as nn
from src.training import (
    TemporalGNNTrainer,
    TemporalLinkPredictionLoss,
    ContrastiveLoss,
    NodeReconstructionLoss
)
from src.advanced_tgnn import AdvancedTemporalGNN
from src.terrorist_network_dataset import TerroristNetworkGenerator, NetworkConfig


class TestTemporalLinkPredictionLoss:
    """Test suite for TemporalLinkPredictionLoss"""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance"""
        return TemporalLinkPredictionLoss()

    def test_loss_initialization(self, loss_fn):
        """Test loss function initializes correctly"""
        assert loss_fn is not None
        assert isinstance(loss_fn, nn.Module)

    def test_loss_computation(self, loss_fn):
        """Test loss computation with dummy data"""
        batch_size = 10
        embedding_dim = 64

        # Create dummy embeddings
        node_embeddings = torch.randn(batch_size, embedding_dim)
        edge_index = torch.randint(0, batch_size, (2, 20))

        # Compute loss
        loss = loss_fn(node_embeddings, edge_index)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0  # loss should be non-negative
        assert not torch.isnan(loss)

    def test_loss_backward(self, loss_fn):
        """Test loss can be backpropagated"""
        node_embeddings = torch.randn(10, 64, requires_grad=True)
        edge_index = torch.randint(0, 10, (2, 15))

        loss = loss_fn(node_embeddings, edge_index)
        loss.backward()

        # Check gradients exist
        assert node_embeddings.grad is not None
        assert not torch.isnan(node_embeddings.grad).any()


class TestContrastiveLoss:
    """Test suite for ContrastiveLoss"""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance"""
        return ContrastiveLoss(temperature=0.5)

    def test_loss_initialization(self, loss_fn):
        """Test loss function initializes correctly"""
        assert loss_fn is not None
        assert loss_fn.temperature == 0.5

    def test_loss_computation(self, loss_fn):
        """Test contrastive loss computation"""
        batch_size = 16
        embedding_dim = 64

        # Create anchor and positive embeddings
        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)

        # Compute loss
        loss = loss_fn(anchor, positive)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_loss_with_negatives(self, loss_fn):
        """Test contrastive loss with negative samples"""
        batch_size = 8
        embedding_dim = 64

        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negative = torch.randn(batch_size, embedding_dim)

        loss = loss_fn(anchor, positive, negative)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


class TestNodeReconstructionLoss:
    """Test suite for NodeReconstructionLoss"""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance"""
        return NodeReconstructionLoss()

    def test_loss_initialization(self, loss_fn):
        """Test loss function initializes correctly"""
        assert loss_fn is not None
        assert isinstance(loss_fn, nn.Module)

    def test_loss_computation(self, loss_fn):
        """Test reconstruction loss computation"""
        batch_size = 20
        feature_dim = 32

        # Create original and reconstructed features
        original = torch.randn(batch_size, feature_dim)
        reconstructed = torch.randn(batch_size, feature_dim)

        # Compute loss
        loss = loss_fn(reconstructed, original)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_perfect_reconstruction(self, loss_fn):
        """Test loss is zero for perfect reconstruction"""
        features = torch.randn(10, 32)
        loss = loss_fn(features, features)

        assert loss.item() < 1e-6  # approximately zero


class TestTemporalGNNTrainer:
    """Test suite for TemporalGNNTrainer"""

    @pytest.fixture
    def model(self):
        """Create model for training"""
        return AdvancedTemporalGNN(
            num_node_features=32,
            num_edge_features=16,
            hidden_dim=64,
            num_temporal_layers=2,
            num_graph_layers=2
        )

    @pytest.fixture
    def trainer(self, model):
        """Create trainer instance"""
        return TemporalGNNTrainer(
            model=model,
            learning_rate=1e-3,
            weight_decay=1e-5
        )

    @pytest.fixture
    def sample_network(self):
        """Create sample network for training"""
        config = NetworkConfig(initial_nodes=15, max_nodes=20)
        generator = TerroristNetworkGenerator(config, seed=42)
        return generator.generate_temporal_network(
            num_timesteps=5,
            num_node_features=32,
            num_edge_features=16
        )

    def test_trainer_initialization(self, trainer, model):
        """Test trainer initializes correctly"""
        assert trainer is not None
        assert trainer.model == model
        assert trainer.optimizer is not None

    def test_training_step(self, trainer, sample_network):
        """Test single training step"""
        initial_params = [p.clone() for p in trainer.model.parameters()]

        # Perform one training step
        loss = trainer.train_step(sample_network)

        # Check loss is valid
        assert isinstance(loss, float)
        assert loss >= 0
        assert not np.isnan(loss)

        # Check parameters updated
        updated = False
        for p_init, p_new in zip(initial_params, trainer.model.parameters()):
            if not torch.equal(p_init, p_new):
                updated = True
                break
        assert updated, "Model parameters should be updated after training step"

    def test_validation_step(self, trainer, sample_network):
        """Test validation step"""
        val_loss = trainer.validate_step(sample_network)

        assert isinstance(val_loss, float)
        assert val_loss >= 0
        assert not np.isnan(val_loss)

    def test_fit_method(self, trainer, sample_network):
        """Test fit method with small dataset"""
        train_networks = [sample_network]
        val_networks = [sample_network]

        history = trainer.fit(
            train_networks=train_networks,
            val_networks=val_networks,
            num_epochs=2,
            verbose=False
        )

        # Check history contains required keys
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2

        # Check losses are decreasing or stable
        assert all(not np.isnan(loss) for loss in history['train_loss'])
        assert all(not np.isnan(loss) for loss in history['val_loss'])

    def test_save_and_load_model(self, trainer, tmp_path):
        """Test model saving and loading"""
        save_path = tmp_path / "model.pt"

        # Save model
        trainer.save_model(str(save_path))
        assert save_path.exists()

        # Load model
        trainer.load_model(str(save_path))

        # Model should still work
        assert trainer.model is not None

    def test_learning_rate_scheduling(self, trainer):
        """Test learning rate can be adjusted"""
        initial_lr = trainer.optimizer.param_groups[0]['lr']

        # Adjust learning rate
        new_lr = initial_lr * 0.5
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = new_lr

        current_lr = trainer.optimizer.param_groups[0]['lr']
        assert current_lr == new_lr


class TestTrainingUtilities:
    """Test suite for training utilities"""

    def test_gradient_clipping(self):
        """Test gradient clipping functionality"""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        # Create large gradients
        x = torch.randn(5, 10, requires_grad=True)
        y = model(x)
        loss = (y ** 2).sum()
        loss.backward()

        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Check gradients are clipped
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= max_norm + 1e-6  # allow small numerical error


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingIntegration:
    """Integration tests for training pipeline"""

    def test_full_training_pipeline(self):
        """Test complete training pipeline from scratch"""
        # Generate dataset
        config = NetworkConfig(initial_nodes=20, max_nodes=30)
        generator = TerroristNetworkGenerator(config, seed=42)

        train_networks = [
            generator.generate_temporal_network(5, 32, 16)
            for _ in range(2)
        ]
        val_networks = [
            generator.generate_temporal_network(5, 32, 16)
        ]

        # Create model
        model = AdvancedTemporalGNN(
            num_node_features=32,
            num_edge_features=16,
            hidden_dim=64,
            num_temporal_layers=2,
            num_graph_layers=2
        )

        # Create trainer
        trainer = TemporalGNNTrainer(model, learning_rate=1e-3)

        # Train
        history = trainer.fit(
            train_networks=train_networks,
            val_networks=val_networks,
            num_epochs=3,
            verbose=False
        )

        # Verify training completed
        assert len(history['train_loss']) == 3
        assert len(history['val_loss']) == 3
        assert all(loss >= 0 for loss in history['train_loss'])
        assert all(loss >= 0 for loss in history['val_loss'])


# Import numpy for numerical checks
import numpy as np
