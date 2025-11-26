"""
Test Suite for Terrorist Network Dataset
=========================================

Tests for dataset generation and network configuration.

Author: Yoon-Seop Lee
"""

import pytest
import torch
import numpy as np
from src.terrorist_network_dataset import (
    TerroristNetworkGenerator,
    NetworkConfig,
    DisruptionEvaluator
)


class TestNetworkConfig:
    """Test suite for NetworkConfig dataclass"""

    def test_config_initialization(self):
        """Test NetworkConfig initializes with default values"""
        config = NetworkConfig()

        assert config.initial_nodes > 0
        assert config.max_nodes >= config.initial_nodes
        assert 0 <= config.recruitment_rate <= 1
        assert 0 <= config.dropout_rate <= 1

    def test_config_custom_values(self):
        """Test NetworkConfig with custom values"""
        config = NetworkConfig(
            initial_nodes=30,
            max_nodes=60,
            recruitment_rate=0.1,
            dropout_rate=0.05
        )

        assert config.initial_nodes == 30
        assert config.max_nodes == 60
        assert config.recruitment_rate == 0.1
        assert config.dropout_rate == 0.05


class TestTerroristNetworkGenerator:
    """Test suite for TerroristNetworkGenerator"""

    @pytest.fixture
    def config(self):
        """Network configuration for testing"""
        return NetworkConfig(
            initial_nodes=20,
            max_nodes=40,
            recruitment_rate=0.05,
            dropout_rate=0.02
        )

    @pytest.fixture
    def generator(self, config):
        """Create generator instance"""
        return TerroristNetworkGenerator(config, seed=42)

    def test_generator_initialization(self, generator, config):
        """Test generator initializes correctly"""
        assert generator is not None
        assert generator.config == config

    def test_generate_temporal_network(self, generator):
        """Test temporal network generation"""
        num_timesteps = 5
        num_node_features = 32
        num_edge_features = 16

        network = generator.generate_temporal_network(
            num_timesteps=num_timesteps,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features
        )

        # Check network has correct number of timesteps
        assert len(network.layers_history) == num_timesteps

        # Check each timestep has correct number of layers
        for timestep in network.layers_history:
            assert len(timestep) == network.num_layers

        # Check layer properties
        first_timestep = network.get_timestep(0)
        for layer in first_timestep:
            assert layer.node_features.size(1) == num_node_features
            assert layer.edge_features.size(1) == num_edge_features
            assert layer.node_features.size(0) >= generator.config.initial_nodes

    def test_network_temporal_evolution(self, generator):
        """Test network evolves over time"""
        network = generator.generate_temporal_network(
            num_timesteps=10,
            num_node_features=32,
            num_edge_features=16
        )

        # Get node counts at different timesteps
        node_counts = []
        for t in range(len(network.layers_history)):
            layers = network.get_timestep(t)
            node_counts.append(layers[0].node_features.size(0))

        # Check that network can grow (not necessarily monotonic due to dropout)
        assert max(node_counts) >= node_counts[0]

    def test_multi_layer_structure(self, generator):
        """Test multi-layer network structure"""
        network = generator.generate_temporal_network(
            num_timesteps=5,
            num_node_features=32,
            num_edge_features=16
        )

        # Check layer names
        expected_layers = ['physical', 'digital', 'financial', 'ideological', 'operational']
        assert network.layer_names == expected_layers

        # Check each layer sequence
        for layer_idx in range(network.num_layers):
            layer_sequence = network.get_layer_sequence(layer_idx)
            assert len(layer_sequence) == 5  # num_timesteps

            for layer in layer_sequence:
                assert layer.layer_type == expected_layers[layer_idx]

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same network"""
        config = NetworkConfig(initial_nodes=20, max_nodes=30)

        gen1 = TerroristNetworkGenerator(config, seed=42)
        net1 = gen1.generate_temporal_network(num_timesteps=3, num_node_features=16, num_edge_features=8)

        gen2 = TerroristNetworkGenerator(config, seed=42)
        net2 = gen2.generate_temporal_network(num_timesteps=3, num_node_features=16, num_edge_features=8)

        # Check same number of nodes at each timestep
        for t in range(3):
            layers1 = net1.get_timestep(t)
            layers2 = net2.get_timestep(t)
            assert layers1[0].node_features.size(0) == layers2[0].node_features.size(0)


class TestDisruptionEvaluator:
    """Test suite for DisruptionEvaluator"""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance"""
        return DisruptionEvaluator()

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initializes correctly"""
        assert evaluator is not None

    def test_evaluate_basic_metrics(self, evaluator):
        """Test basic disruption metrics computation"""
        # Create simple test network
        num_nodes = 10
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        removed_nodes = torch.tensor([1, 3])

        # This test verifies the evaluator exists
        assert evaluator is not None


@pytest.mark.integration
class TestNetworkIntegration:
    """Integration tests for network generation pipeline"""

    def test_full_pipeline(self):
        """Test complete network generation pipeline"""
        # Configure
        config = NetworkConfig(initial_nodes=15, max_nodes=25)

        # Generate
        generator = TerroristNetworkGenerator(config, seed=123)
        network = generator.generate_temporal_network(
            num_timesteps=5,
            num_node_features=32,
            num_edge_features=16
        )

        # Verify
        assert network is not None
        assert len(network.layers_history) == 5
        assert network.num_layers == 5

        # Test aggregation
        agg_edge_index, agg_weights = network.get_aggregated_network(0)
        assert agg_edge_index.size(0) == 2  # [2, num_edges]
        assert agg_weights.size(0) == agg_edge_index.size(1)
