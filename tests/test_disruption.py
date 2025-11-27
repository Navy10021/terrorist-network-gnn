"""
Unit tests for terrorist_network_disruption.py
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.terrorist_network_disruption import (
    NetworkLayer,
    MultiLayerTemporalNetwork,
    MultiLayerTemporalGNN,
    EnhancedCriticalNodeDetector,
)


class TestNetworkLayer:
    """Test NetworkLayer dataclass"""

    def test_network_layer_creation(self):
        """Test creating a network layer"""
        num_nodes = 10
        num_edges = 15

        layer = NetworkLayer(
            name="physical",
            layer_type="contact",
            node_features=torch.randn(num_nodes, 64),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            edge_features=torch.randn(num_edges, 32),
            edge_weights=torch.rand(num_edges),
            metadata={"description": "Physical contact network"},
        )

        assert layer.name == "physical"
        assert layer.layer_type == "contact"
        assert layer.node_features.size(0) == num_nodes
        assert layer.edge_index.size(1) == num_edges
        assert isinstance(layer.metadata, dict)


class TestMultiLayerTemporalNetwork:
    """Test MultiLayerTemporalNetwork"""

    @pytest.fixture
    def network(self):
        """Create a multi-layer temporal network"""
        return MultiLayerTemporalNetwork(num_nodes=10, num_layers=5)

    @pytest.fixture
    def sample_layers(self):
        """Create sample network layers"""
        num_nodes = 10
        layers = []
        layer_names = ["physical", "digital", "financial", "ideological", "operational"]

        for name in layer_names:
            num_edges = np.random.randint(5, 15)
            layer = NetworkLayer(
                name=name,
                layer_type=name,
                node_features=torch.randn(num_nodes, 64),
                edge_index=torch.randint(0, num_nodes, (2, num_edges)),
                edge_features=torch.randn(num_edges, 32),
                edge_weights=torch.rand(num_edges),
                metadata={},
            )
            layers.append(layer)

        return layers

    def test_network_initialization(self, network):
        """Test network initialization"""
        assert network.num_nodes == 10
        assert network.num_layers == 5
        assert len(network.layer_names) == 5
        assert network.layers_history == []

    def test_add_timestep(self, network, sample_layers):
        """Test adding a timestep"""
        network.add_timestep(sample_layers)

        assert len(network.layers_history) == 1
        assert len(network.layers_history[0]) == 5

    def test_get_timestep(self, network, sample_layers):
        """Test getting layers at a specific timestep"""
        network.add_timestep(sample_layers)

        retrieved_layers = network.get_timestep(0)

        assert len(retrieved_layers) == 5
        assert retrieved_layers[0].name == "physical"

    def test_get_layer_sequence(self, network, sample_layers):
        """Test getting temporal sequence of a layer"""
        # Add 3 timesteps
        for _ in range(3):
            network.add_timestep(sample_layers)

        physical_sequence = network.get_layer_sequence(0)

        assert len(physical_sequence) == 3
        assert all(layer.name == "physical" for layer in physical_sequence)

    def test_get_aggregated_network(self, network, sample_layers):
        """Test aggregating multiple layers"""
        network.add_timestep(sample_layers)

        edge_index, edge_weights = network.get_aggregated_network(t=0)

        assert edge_index.size(0) == 2  # [2, num_edges]
        assert edge_weights.size(0) == edge_index.size(1)

    def test_weighted_aggregation(self, network, sample_layers):
        """Test weighted layer aggregation"""
        network.add_timestep(sample_layers)

        # Give more weight to first layer
        weights = [2.0, 1.0, 1.0, 1.0, 1.0]
        edge_index, edge_weights = network.get_aggregated_network(t=0, weights=weights)

        assert edge_index.dim() == 2
        assert edge_weights.dim() == 1


class TestMultiLayerTemporalGNN:
    """Test MultiLayerTemporalGNN"""

    @pytest.fixture
    def model(self):
        """Create multi-layer GNN model"""
        return MultiLayerTemporalGNN(
            num_node_features=64,
            num_edge_features=32,
            hidden_dim=128,
            num_layers=5,
            layer_fusion="attention",
            num_temporal_layers=2,
            num_graph_layers=2,
            num_attention_heads=4,
            memory_size=50,
            dropout=0.1,
        )

    def test_model_initialization(self, model):
        """Test model initialization"""
        assert isinstance(model, nn.Module)
        assert model.num_layers == 5
        assert len(model.layer_gnns) == 5
        assert model.layer_fusion == "attention"

    def test_model_has_layer_gnns(self, model):
        """Test that model has GNN for each layer"""
        assert len(model.layer_gnns) == 5
        for gnn in model.layer_gnns:
            assert isinstance(gnn, nn.Module)

    @pytest.mark.slow
    def test_forward_pass(self, model):
        """Test forward pass through multi-layer GNN"""
        # Create a small multi-layer network
        num_nodes = 10
        num_timesteps = 3

        network = MultiLayerTemporalNetwork(num_nodes=num_nodes, num_layers=5)

        # Add timesteps
        for t in range(num_timesteps):
            layers = []
            for layer_idx in range(5):
                num_edges = 8
                layer = NetworkLayer(
                    name=f"layer_{layer_idx}",
                    layer_type="test",
                    node_features=torch.randn(num_nodes, 64),
                    edge_index=torch.randint(0, num_nodes, (2, num_edges)),
                    edge_features=torch.randn(num_edges, 32),
                    edge_weights=torch.rand(num_edges),
                    metadata={},
                )
                layers.append(layer)
            network.add_timestep(layers)

        timestamps = torch.arange(num_timesteps).float()

        # Forward pass
        output = model(network, timestamps)

        # Check output shape
        assert output.dim() == 2
        assert output.size(0) == num_nodes
        assert output.size(1) == 128  # hidden_dim


class TestEnhancedCriticalNodeDetector:
    """Test EnhancedCriticalNodeDetector"""

    @pytest.fixture
    def detector(self):
        """Create critical node detector"""
        return EnhancedCriticalNodeDetector()

    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert isinstance(detector.importance_metrics, list)
        assert len(detector.importance_metrics) > 0
        assert "degree_centrality" in detector.importance_metrics
        assert "betweenness_centrality" in detector.importance_metrics

    def test_custom_metrics(self):
        """Test creating detector with custom metrics"""
        custom_metrics = ["degree_centrality", "pagerank"]
        detector = EnhancedCriticalNodeDetector(importance_metrics=custom_metrics)

        assert detector.importance_metrics == custom_metrics
        assert len(detector.importance_metrics) == 2

    def test_detector_has_required_methods(self, detector):
        """Test that detector has required methods"""
        assert hasattr(detector, "compute_multilayer_centrality")
        # Note: We'd add more method tests as we implement them


class TestDisruptionMetrics:
    """Test disruption evaluation metrics"""

    def test_disruption_score_calculation(self):
        """Test calculating disruption scores"""
        # Create a simple network
        num_nodes = 10
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

        # This would test actual disruption score computation
        # For now, just verify the structure
        assert edge_index.size(0) == 2

    def test_network_fragmentation(self):
        """Test measuring network fragmentation"""
        # Test would verify fragmentation metrics
        # after removing critical nodes
        pass


class TestTemporalResilience:
    """Test temporal resilience prediction"""

    def test_resilience_over_time(self):
        """Test predicting network resilience over time"""
        # This would test the resilience prediction
        # functionality once implemented
        pass


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
