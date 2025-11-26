"""
Test Suite for Network Disruption Algorithms
=============================================

Simplified tests that match actual implementation.

Author: Yoon-Seop Lee
"""

import pytest
import torch
from src.terrorist_network_disruption import (
    MultiLayerTemporalNetwork,
    NetworkLayer,
    CriticalNodeDetector
)


class TestMultiLayerTemporalNetwork:
    """Test suite for MultiLayerTemporalNetwork"""

    def test_network_initialization(self):
        """Test network initializes correctly"""
        network = MultiLayerTemporalNetwork(num_nodes=20, num_layers=5)
        assert network.num_nodes == 20
        assert network.num_layers == 5

    def test_add_timestep(self):
        """Test adding timestep to network"""
        network = MultiLayerTemporalNetwork(num_nodes=20, num_layers=5)

        layers = []
        for i in range(5):
            layer = NetworkLayer(
                name=network.layer_names[i],
                layer_type=network.layer_names[i],
                node_features=torch.randn(20, 32),
                edge_index=torch.randint(0, 20, (2, 30)),
                edge_features=torch.randn(30, 16),
                edge_weights=torch.rand(30),
                metadata={}
            )
            layers.append(layer)

        network.add_timestep(layers)
        assert len(network.layers_history) == 1

    def test_get_layer_sequence(self):
        """Test getting layer sequence over time"""
        network = MultiLayerTemporalNetwork(num_nodes=20, num_layers=5)

        for t in range(3):
            layers = []
            for i in range(5):
                layer = NetworkLayer(
                    name=network.layer_names[i],
                    layer_type=network.layer_names[i],
                    node_features=torch.randn(20, 32),
                    edge_index=torch.randint(0, 20, (2, 30)),
                    edge_features=torch.randn(30, 16),
                    edge_weights=torch.rand(30),
                    metadata={}
                )
                layers.append(layer)
            network.add_timestep(layers)

        physical_sequence = network.get_layer_sequence(0)
        assert len(physical_sequence) == 3

    def test_aggregated_network(self):
        """Test network aggregation across layers"""
        network = MultiLayerTemporalNetwork(num_nodes=20, num_layers=5)

        layers = []
        for i in range(5):
            layer = NetworkLayer(
                name=network.layer_names[i],
                layer_type=network.layer_names[i],
                node_features=torch.randn(20, 32),
                edge_index=torch.randint(0, 20, (2, 30)),
                edge_features=torch.randn(30, 16),
                edge_weights=torch.rand(30),
                metadata={}
            )
            layers.append(layer)
        network.add_timestep(layers)

        agg_edge_index, agg_weights = network.get_aggregated_network(0)
        assert agg_edge_index.size(0) == 2


class TestCriticalNodeDetector:
    """Test suite for CriticalNodeDetector"""

    def test_detector_initialization(self):
        """Test detector initializes correctly"""
        detector = CriticalNodeDetector()
        assert detector is not None

    def test_detect_critical_nodes_basic(self):
        """Test basic critical node detection"""
        detector = CriticalNodeDetector()
        num_nodes = 20
        edge_index = torch.randint(0, num_nodes, (2, 40))

        # Basic detection without embeddings
        critical_nodes, scores = detector.detect_critical_nodes(
            edge_index=edge_index,
            num_nodes=num_nodes,
            top_k=5
        )

        assert critical_nodes.size(0) == 5
        assert scores.size(0) == num_nodes
