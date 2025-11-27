"""
Unit tests for Advanced Temporal GNN
"""

import pytest
import torch
import torch.nn as nn
from src.advanced_tgnn import (
    AdvancedTemporalGNN,
    HierarchicalTemporalPooling,
    EnhancedTemporalMemoryBank,
    AdaptiveTimeEncoding,
)


class TestAdvancedTemporalGNN:
    """Test suite for AdvancedTemporalGNN model"""

    @pytest.fixture
    def model(self):
        """Create a model instance for testing"""
        return AdvancedTemporalGNN(
            num_node_features=64,
            num_edge_features=32,
            hidden_dim=128,
            num_temporal_layers=3,
            num_graph_layers=3,
            num_attention_heads=8,
            memory_size=100,
            dropout=0.1,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample graph data"""
        batch_size = 2
        num_nodes = 10
        num_edges = 15

        node_features = torch.randn(batch_size, num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (batch_size, 2, num_edges))
        edge_attr = torch.randn(batch_size, num_edges, 32)
        timestamps = torch.arange(batch_size).float()

        return node_features, edge_index, edge_attr, timestamps

    def test_model_initialization(self, model):
        """Test that model initializes correctly"""
        assert isinstance(model, nn.Module)
        assert model.hidden_dim == 128
        assert model.num_temporal_layers == 3

    def test_forward_pass(self, model, sample_data):
        """Test forward pass produces correct output shape"""
        node_features, edge_index, edge_attr, timestamps = sample_data

        # Forward pass
        output = model(node_features, edge_index, edge_attr, timestamps)

        # Check output shape
        batch_size, num_nodes = node_features.shape[:2]
        assert output.shape == (batch_size, num_nodes, 128)

    def test_no_nan_in_output(self, model, sample_data):
        """Test that forward pass doesn't produce NaN values"""
        node_features, edge_index, edge_attr, timestamps = sample_data

        output = model(node_features, edge_index, edge_attr, timestamps)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self, model, sample_data):
        """Test that gradients flow properly"""
        node_features, edge_index, edge_attr, timestamps = sample_data

        # Forward pass
        output = model(node_features, edge_index, edge_attr, timestamps)

        # Dummy loss
        loss = output.sum()
        loss.backward()

        # Check that parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_different_batch_sizes(self, model):
        """Test model with different batch sizes"""
        for batch_size in [1, 2, 4, 8]:
            node_features = torch.randn(batch_size, 10, 64)
            edge_index = torch.randint(0, 10, (batch_size, 2, 15))
            edge_attr = torch.randn(batch_size, 15, 32)
            timestamps = torch.arange(batch_size).float()

            output = model(node_features, edge_index, edge_attr, timestamps)
            assert output.shape == (batch_size, 10, 128)


class TestHierarchicalTemporalPooling:
    """Test suite for HierarchicalTemporalPooling"""

    @pytest.fixture
    def pooling(self):
        """Create pooling layer"""
        return HierarchicalTemporalPooling(
            hidden_dim=128,
            num_scales=3,
        )

    def test_pooling_initialization(self, pooling):
        """Test pooling layer initialization"""
        assert pooling.num_scales == 3
        assert pooling.hidden_dim == 128

    def test_pooling_forward(self, pooling):
        """Test pooling forward pass"""
        # Create temporal sequence
        batch_size = 2
        seq_len = 10
        hidden_dim = 128

        temporal_states = torch.randn(batch_size, seq_len, hidden_dim)

        # Apply pooling
        output = pooling(temporal_states)

        # Check output
        assert output.shape == (batch_size, hidden_dim)
        assert not torch.isnan(output).any()


class TestEnhancedTemporalMemoryBank:
    """Test suite for EnhancedTemporalMemoryBank"""

    @pytest.fixture
    def memory_bank(self):
        """Create memory bank"""
        return EnhancedTemporalMemoryBank(
            memory_size=100,
            hidden_dim=128,
        )

    def test_memory_initialization(self, memory_bank):
        """Test memory bank initialization"""
        assert memory_bank.memory_size == 100
        assert memory_bank.hidden_dim == 128

    def test_store_and_retrieve(self, memory_bank):
        """Test storing and retrieving memories"""
        # Store memory
        key = torch.randn(1, 128)
        value = torch.randn(1, 128)
        timestamp = 0

        memory_bank.store(key, value, timestamp)

        # Retrieve memory
        query = key
        retrieved = memory_bank.retrieve(query, k=1)

        assert retrieved.shape == (1, 128)

    def test_lru_eviction(self, memory_bank):
        """Test LRU eviction when memory is full"""
        # Fill memory beyond capacity
        for i in range(150):
            key = torch.randn(1, 128)
            value = torch.randn(1, 128)
            memory_bank.store(key, value, timestamp=i)

        # Check memory size is capped
        assert len(memory_bank.keys) <= 100


class TestAdaptiveTimeEncoding:
    """Test suite for AdaptiveTimeEncoding"""

    @pytest.fixture
    def time_encoder(self):
        """Create time encoder"""
        return AdaptiveTimeEncoding(hidden_dim=128)

    def test_encoding_initialization(self, time_encoder):
        """Test time encoding initialization"""
        assert time_encoder.hidden_dim == 128

    def test_encoding_forward(self, time_encoder):
        """Test time encoding forward pass"""
        timestamps = torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0])

        encoding = time_encoder(timestamps)

        assert encoding.shape == (5, 128)
        assert not torch.isnan(encoding).any()

    def test_different_timestamps(self, time_encoder):
        """Test encoding produces different outputs for different times"""
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([10.0])

        enc1 = time_encoder(t1)
        enc2 = time_encoder(t2)

        # Encodings should be different
        assert not torch.allclose(enc1, enc2)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
