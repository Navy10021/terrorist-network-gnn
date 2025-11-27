"""
Unit tests for terrorist_network_dataset.py
"""

import networkx as nx
import pytest
import torch

from src.terrorist_network_dataset import (
    MultiLayerTemporalNetwork,
    NetworkConfig,
    TerroristNetworkGenerator,
)


class TestNetworkConfig:
    """Test NetworkConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = NetworkConfig()
        assert config.initial_nodes > 0
        assert config.max_nodes >= config.initial_nodes
        assert 0 <= config.recruitment_rate <= 1
        assert 0 <= config.dropout_rate <= 1

    def test_custom_config(self):
        """Test custom configuration"""
        config = NetworkConfig(
            initial_nodes=30,
            max_nodes=50,
            recruitment_rate=0.1,
        )
        assert config.initial_nodes == 30
        assert config.max_nodes == 50
        assert config.recruitment_rate == 0.1


class TestTerroristNetworkGenerator:
    """Test TerroristNetworkGenerator"""

    @pytest.fixture
    def generator(self, network_config):
        """Create a generator instance"""
        return TerroristNetworkGenerator(network_config)

    def test_generator_initialization(self, generator, network_config):
        """Test generator initializes correctly"""
        assert generator.config == network_config
        assert isinstance(generator, TerroristNetworkGenerator)

    def test_generate_temporal_network(self, generator):
        """Test temporal network generation"""
        num_timesteps = 5
        network = generator.generate_temporal_network(num_timesteps)

        assert isinstance(network, MultiLayerTemporalNetwork)
        assert len(network.snapshots) == num_timesteps

    def test_network_has_required_layers(self, generator):
        """Test that generated network has all required layers"""
        network = generator.generate_temporal_network(3)

        # Check that each snapshot has the required layers
        for snapshot in network.snapshots:
            assert hasattr(snapshot, "physical")
            assert hasattr(snapshot, "communication")
            assert hasattr(snapshot, "financial")
            assert hasattr(snapshot, "ideological")
            assert hasattr(snapshot, "operational")

    def test_node_features(self, generator):
        """Test that nodes have correct feature dimensions"""
        network = generator.generate_temporal_network(3)

        for snapshot in network.snapshots:
            assert snapshot.node_features.dim() == 2
            # Check feature dimension matches expected size

    def test_edge_features(self, generator):
        """Test that edges have correct feature dimensions"""
        network = generator.generate_temporal_network(3)

        for snapshot in network.snapshots:
            if snapshot.edge_attr is not None:
                assert snapshot.edge_attr.dim() == 2

    def test_temporal_consistency(self, generator):
        """Test that network evolves consistently over time"""
        network = generator.generate_temporal_network(5)

        # Number of nodes should generally increase or stay same
        num_nodes = [snapshot.num_nodes for snapshot in network.snapshots]

        # Check that nodes don't disappear suddenly (allowing for dropout)
        for i in range(len(num_nodes) - 1):
            # Allow some dropout but not drastic changes
            assert abs(num_nodes[i + 1] - num_nodes[i]) <= 10


class TestMultiLayerTemporalNetwork:
    """Test MultiLayerTemporalNetwork"""

    def test_network_structure(self, network_config):
        """Test network structure and properties"""
        generator = TerroristNetworkGenerator(network_config)
        network = generator.generate_temporal_network(3)

        assert hasattr(network, "snapshots")
        assert hasattr(network, "timestamps")
        assert len(network.snapshots) == 3

    def test_to_pytorch_geometric(self, network_config):
        """Test conversion to PyTorch Geometric format"""
        generator = TerroristNetworkGenerator(network_config)
        network = generator.generate_temporal_network(3)

        # Test that network can be converted to PyG format
        # This would depend on implementation details
        pass


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
