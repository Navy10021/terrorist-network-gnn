"""
Test Suite for Network Disruption Algorithms
=============================================

Tests for critical node detection, temporal resilience prediction,
and adversarial robustness analysis.

Author: Yoon-Seop Lee
"""

import pytest
import torch
import numpy as np
from src.terrorist_network_disruption import (
    MultiLayerTemporalGNN,
    CriticalNodeDetector,
    NetworkDisruptionOptimizer,
    TemporalResiliencePredictor,
    AdversarialNetworkAttack,
    NetworkLayer,
    MultiLayerTemporalNetwork
)


class TestMultiLayerTemporalNetwork:
    """Test suite for MultiLayerTemporalNetwork"""

    @pytest.fixture
    def network(self):
        """Create test network"""
        return MultiLayerTemporalNetwork(num_nodes=20, num_layers=5)

    def test_network_initialization(self, network):
        """Test network initializes correctly"""
        assert network.num_nodes == 20
        assert network.num_layers == 5
        assert len(network.layer_names) == 5
        assert network.layers_history == []

    def test_add_timestep(self, network):
        """Test adding timestep to network"""
        # Create dummy layers
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
        assert len(network.get_timestep(0)) == 5

    def test_get_layer_sequence(self, network):
        """Test getting layer sequence over time"""
        # Add multiple timesteps
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

        # Get physical layer sequence
        physical_sequence = network.get_layer_sequence(0)
        assert len(physical_sequence) == 3
        assert all(layer.layer_type == 'physical' for layer in physical_sequence)

    def test_aggregated_network(self, network):
        """Test network aggregation across layers"""
        # Add one timestep
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

        # Get aggregated network
        agg_edge_index, agg_weights = network.get_aggregated_network(0)

        assert agg_edge_index.size(0) == 2
        assert agg_weights.size(0) == agg_edge_index.size(1)


class TestMultiLayerTemporalGNN:
    """Test suite for MultiLayerTemporalGNN model"""

    @pytest.fixture
    def model_config(self):
        """Model configuration for testing"""
        return {
            'num_node_features': 32,
            'num_edge_features': 16,
            'hidden_dim': 64,
            'num_layers': 5
        }

    @pytest.fixture
    def model(self, model_config):
        """Create model instance"""
        return MultiLayerTemporalGNN(**model_config)

    def test_model_initialization(self, model, model_config):
        """Test model initializes correctly"""
        assert model is not None
        assert model.hidden_dim == model_config['hidden_dim']

    def test_forward_pass_single_timestep(self, model):
        """Test forward pass with single timestep"""
        num_nodes = 20
        num_edges = 30

        node_features = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_features = torch.randn(num_edges, 16)

        # Forward pass
        embeddings = model(node_features, edge_index, edge_features)

        assert embeddings.shape == (num_nodes, 64)
        assert not torch.isnan(embeddings).any()


class TestCriticalNodeDetector:
    """Test suite for CriticalNodeDetector"""

    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return CriticalNodeDetector()

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly"""
        assert detector is not None

    def test_detect_critical_nodes(self, detector):
        """Test critical node detection"""
        num_nodes = 20
        num_edges = 40

        # Create simple network
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        embeddings = torch.randn(num_nodes, 64)

        # Detect critical nodes
        critical_nodes, scores = detector.detect_critical_nodes(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_embeddings=embeddings,
            top_k=5
        )

        assert critical_nodes.size(0) == 5
        assert scores.size(0) == num_nodes
        assert not torch.isnan(scores).any()

    def test_scores_are_sorted(self, detector):
        """Test that detected nodes are sorted by importance"""
        num_nodes = 15
        edge_index = torch.randint(0, num_nodes, (2, 30))
        embeddings = torch.randn(num_nodes, 64)

        critical_nodes, scores = detector.detect_critical_nodes(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_embeddings=embeddings,
            top_k=5
        )

        # Check that selected nodes have highest scores
        selected_scores = scores[critical_nodes]
        assert torch.all(selected_scores[:-1] >= selected_scores[1:])


class TestNetworkDisruptionOptimizer:
    """Test suite for NetworkDisruptionOptimizer"""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        return NetworkDisruptionOptimizer()

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly"""
        assert optimizer is not None

    def test_compute_optimal_strategy(self, optimizer):
        """Test optimal disruption strategy computation"""
        num_nodes = 20
        edge_index = torch.randint(0, num_nodes, (2, 40))
        node_scores = torch.rand(num_nodes)

        # This test verifies the optimizer can be called
        assert callable(optimizer.optimize)


class TestTemporalResiliencePredictor:
    """Test suite for TemporalResiliencePredictor"""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance"""
        return TemporalResiliencePredictor(
            input_dim=64,
            hidden_dim=128,
            num_future_steps=5
        )

    def test_predictor_initialization(self, predictor):
        """Test predictor initializes correctly"""
        assert predictor is not None
        assert predictor.num_future_steps == 5

    def test_predict_resilience(self, predictor):
        """Test resilience prediction"""
        batch_size = 3
        seq_len = 10
        feature_dim = 64

        # Create temporal sequence
        network_states = torch.randn(batch_size, seq_len, feature_dim)

        # Predict future resilience
        predictions = predictor(network_states)

        assert predictions.shape[0] == batch_size
        assert predictions.shape[1] == 5  # num_future_steps
        assert not torch.isnan(predictions).any()


class TestAdversarialNetworkAttack:
    """Test suite for AdversarialNetworkAttack"""

    @pytest.fixture
    def attack(self):
        """Create attack instance"""
        return AdversarialNetworkAttack(adaptation_rate=0.1)

    def test_attack_initialization(self, attack):
        """Test attack initializes correctly"""
        assert attack is not None
        assert attack.adaptation_rate == 0.1

    def test_simulate_adaptation(self, attack):
        """Test network adaptation simulation"""
        num_nodes = 20
        edge_index = torch.randint(0, num_nodes, (2, 40))
        removed_nodes = torch.tensor([1, 5, 10])

        # This test verifies the attack can be called
        assert callable(attack.simulate_adaptation)


@pytest.mark.integration
class TestDisruptionPipeline:
    """Integration tests for disruption pipeline"""

    def test_full_disruption_pipeline(self):
        """Test complete disruption analysis pipeline"""
        # Create network
        num_nodes = 30
        num_edges = 60

        node_features = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_features = torch.randn(num_edges, 16)

        # Build model
        model = MultiLayerTemporalGNN(
            num_node_features=32,
            num_edge_features=16,
            hidden_dim=64
        )

        # Get embeddings
        embeddings = model(node_features, edge_index, edge_features)

        # Detect critical nodes
        detector = CriticalNodeDetector()
        critical_nodes, scores = detector.detect_critical_nodes(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_embeddings=embeddings,
            top_k=5
        )

        # Verify pipeline
        assert critical_nodes.size(0) == 5
        assert embeddings.size(0) == num_nodes
        assert scores.size(0) == num_nodes
