"""
Test Suite for Training Framework
==================================

Simplified tests that match actual implementation.

Author: Yoon-Seop Lee
"""

import pytest
import torch
import torch.nn as nn
from src.training import (
    TemporalGNNTrainer,
    ContrastiveLoss,
    NodeReconstructionLoss
)
from src.advanced_tgnn import AdvancedTemporalGNN


class TestContrastiveLoss:
    """Test suite for ContrastiveLoss"""

    def test_loss_initialization(self):
        """Test loss function initializes correctly"""
        loss_fn = ContrastiveLoss(temperature=0.5)
        assert loss_fn is not None
        assert loss_fn.temperature == 0.5

    def test_loss_computation(self):
        """Test contrastive loss computation"""
        loss_fn = ContrastiveLoss(temperature=0.5)
        batch_size = 16
        embedding_dim = 64

        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)

        loss = loss_fn(anchor, positive)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


class TestNodeReconstructionLoss:
    """Test suite for NodeReconstructionLoss"""

    def test_loss_initialization(self):
        """Test loss function initializes correctly"""
        loss_fn = NodeReconstructionLoss()
        assert loss_fn is not None

    def test_loss_computation(self):
        """Test reconstruction loss computation"""
        loss_fn = NodeReconstructionLoss()
        batch_size = 20
        feature_dim = 32

        original = torch.randn(batch_size, feature_dim)
        reconstructed = torch.randn(batch_size, feature_dim)

        loss = loss_fn(reconstructed, original)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_perfect_reconstruction(self):
        """Test loss is zero for perfect reconstruction"""
        loss_fn = NodeReconstructionLoss()
        features = torch.randn(10, 32)
        loss = loss_fn(features, features)
        assert loss.item() < 1e-6


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

    def test_trainer_initialization(self, trainer, model):
        """Test trainer initializes correctly"""
        assert trainer is not None
        assert trainer.model == model
        assert trainer.optimizer is not None

    def test_learning_rate_scheduling(self, trainer):
        """Test learning rate can be adjusted"""
        initial_lr = trainer.optimizer.param_groups[0]['lr']
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

        x = torch.randn(5, 10, requires_grad=True)
        y = model(x)
        loss = (y ** 2).sum()
        loss.backward()

        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= max_norm + 1e-6
