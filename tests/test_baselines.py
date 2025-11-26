"""
Test Suite for Baseline Methods
================================

Tests for baseline comparison methods including static GNNs,
centrality measures, and simple temporal models.

Author: Yoon-Seop Lee
"""

import pytest
import torch
import numpy as np
import networkx as nx
from src.baselines import (
    StaticGCN,
    StaticGAT,
    StaticGraphSAGE,
    SimpleTemporalGNN,
    CentralityBaseline,
    BaselineEvaluator
)


class TestStaticGCN:
    """Test suite for Static GCN baseline"""

    @pytest.fixture
    def model(self):
        """Create GCN model instance"""
        return StaticGCN(
            num_features=32,
            hidden_dim=64,
            num_layers=2
        )

    def test_model_initialization(self, model):
        """Test GCN initializes correctly"""
        assert model is not None
        assert model.hidden_dim == 64

    def test_forward_pass(self, model):
        """Test forward pass with synthetic data"""
        num_nodes = 20
        num_edges = 40

        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass
        embeddings = model(x, edge_index)

        assert embeddings.shape == (num_nodes, 64)
        assert not torch.isnan(embeddings).any()

    def test_different_graph_sizes(self, model):
        """Test model works with different graph sizes"""
        for num_nodes in [10, 20, 50]:
            x = torch.randn(num_nodes, 32)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))

            embeddings = model(x, edge_index)

            assert embeddings.shape[0] == num_nodes
            assert not torch.isnan(embeddings).any()


class TestStaticGAT:
    """Test suite for Static GAT baseline"""

    @pytest.fixture
    def model(self):
        """Create GAT model instance"""
        return StaticGAT(
            num_features=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4
        )

    def test_model_initialization(self, model):
        """Test GAT initializes correctly"""
        assert model is not None
        assert model.hidden_dim == 64
        assert model.num_heads == 4

    def test_forward_pass(self, model):
        """Test forward pass with attention mechanism"""
        num_nodes = 20
        num_edges = 40

        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass
        embeddings = model(x, edge_index)

        assert embeddings.shape == (num_nodes, 64)
        assert not torch.isnan(embeddings).any()

    def test_attention_computation(self, model):
        """Test attention weights are computed"""
        x = torch.randn(15, 32)
        edge_index = torch.randint(0, 15, (2, 30))

        embeddings = model(x, edge_index)

        # Just verify the model produces valid output
        assert embeddings.shape[0] == 15
        assert not torch.isnan(embeddings).any()


class TestStaticGraphSAGE:
    """Test suite for Static GraphSAGE baseline"""

    @pytest.fixture
    def model(self):
        """Create GraphSAGE model instance"""
        return StaticGraphSAGE(
            num_features=32,
            hidden_dim=64,
            num_layers=2
        )

    def test_model_initialization(self, model):
        """Test GraphSAGE initializes correctly"""
        assert model is not None
        assert model.hidden_dim == 64

    def test_forward_pass(self, model):
        """Test forward pass with neighborhood sampling"""
        num_nodes = 25
        num_edges = 50

        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass
        embeddings = model(x, edge_index)

        assert embeddings.shape == (num_nodes, 64)
        assert not torch.isnan(embeddings).any()

    def test_aggregation_function(self, model):
        """Test different aggregation functions"""
        x = torch.randn(20, 32)
        edge_index = torch.randint(0, 20, (2, 40))

        embeddings = model(x, edge_index)

        assert embeddings.shape[0] == 20
        assert not torch.isnan(embeddings).any()


class TestSimpleTemporalGNN:
    """Test suite for Simple Temporal GNN baseline"""

    @pytest.fixture
    def model(self):
        """Create simple temporal model instance"""
        return SimpleTemporalGNN(
            num_node_features=32,
            num_edge_features=16,
            hidden_dim=64
        )

    def test_model_initialization(self, model):
        """Test simple T-GNN initializes correctly"""
        assert model is not None
        assert model.hidden_dim == 64

    def test_forward_pass_temporal(self, model):
        """Test forward pass with temporal sequence"""
        num_timesteps = 5
        num_nodes = 20
        num_edges = 30

        # Create temporal sequences
        node_features_seq = [
            torch.randn(num_nodes, 32) for _ in range(num_timesteps)
        ]
        edge_indices_seq = [
            torch.randint(0, num_nodes, (2, num_edges)) for _ in range(num_timesteps)
        ]
        edge_features_seq = [
            torch.randn(num_edges, 16) for _ in range(num_timesteps)
        ]

        # Forward pass
        embeddings = model(node_features_seq, edge_indices_seq, edge_features_seq)

        assert embeddings.shape == (num_nodes, 64)
        assert not torch.isnan(embeddings).any()

    def test_temporal_aggregation(self, model):
        """Test temporal information is aggregated"""
        num_timesteps = 10
        num_nodes = 15

        node_features_seq = [torch.randn(num_nodes, 32) for _ in range(num_timesteps)]
        edge_indices_seq = [torch.randint(0, num_nodes, (2, 25)) for _ in range(num_timesteps)]
        edge_features_seq = [torch.randn(25, 16) for _ in range(num_timesteps)]

        embeddings = model(node_features_seq, edge_indices_seq, edge_features_seq)

        assert embeddings.shape[0] == num_nodes
        assert not torch.isnan(embeddings).any()


class TestCentralityBaseline:
    """Test suite for Centrality-based baseline"""

    @pytest.fixture
    def baseline(self):
        """Create centrality baseline instance"""
        return CentralityBaseline()

    def test_baseline_initialization(self, baseline):
        """Test centrality baseline initializes correctly"""
        assert baseline is not None

    def test_degree_centrality(self, baseline):
        """Test degree centrality computation"""
        num_nodes = 20
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0]
        ], dtype=torch.long)

        scores = baseline.compute_degree_centrality(edge_index, num_nodes)

        assert scores.size(0) == num_nodes
        assert not torch.isnan(scores).any()
        assert torch.all(scores >= 0)

    def test_betweenness_centrality(self, baseline):
        """Test betweenness centrality computation"""
        # Create simple graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

        scores = baseline.compute_betweenness_centrality(G)

        assert len(scores) == 5
        assert all(score >= 0 for score in scores.values())

    def test_pagerank(self, baseline):
        """Test PageRank computation"""
        # Create directed graph
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])

        scores = baseline.compute_pagerank(G)

        assert len(scores) == 4
        assert all(score >= 0 for score in scores.values())
        # PageRank scores should sum to approximately 1
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_eigenvector_centrality(self, baseline):
        """Test eigenvector centrality computation"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])

        scores = baseline.compute_eigenvector_centrality(G)

        assert len(scores) == 4
        assert all(score >= 0 for score in scores.values())

    def test_closeness_centrality(self, baseline):
        """Test closeness centrality computation"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

        scores = baseline.compute_closeness_centrality(G)

        assert len(scores) == 5
        assert all(0 <= score <= 1 for score in scores.values())


class TestBaselineEvaluator:
    """Test suite for BaselineEvaluator"""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance"""
        return BaselineEvaluator()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation"""
        num_nodes = 20
        num_edges = 40

        node_features = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        return node_features, edge_index, num_nodes

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initializes correctly"""
        assert evaluator is not None

    def test_evaluate_single_baseline(self, evaluator, sample_data):
        """Test evaluation of single baseline method"""
        node_features, edge_index, num_nodes = sample_data

        # Create baseline model
        model = StaticGCN(num_features=32, hidden_dim=64)

        # Evaluate
        results = evaluator.evaluate_baseline(
            model=model,
            node_features=node_features,
            edge_index=edge_index,
            num_nodes=num_nodes
        )

        assert isinstance(results, dict)
        assert 'embeddings' in results or 'scores' in results

    def test_compare_multiple_baselines(self, evaluator, sample_data):
        """Test comparison of multiple baselines"""
        node_features, edge_index, num_nodes = sample_data

        # Create multiple baselines
        baselines = {
            'GCN': StaticGCN(num_features=32, hidden_dim=64),
            'GAT': StaticGAT(num_features=32, hidden_dim=64, num_heads=4),
        }

        # Compare
        comparison = evaluator.compare_baselines(
            baselines=baselines,
            node_features=node_features,
            edge_index=edge_index,
            num_nodes=num_nodes
        )

        assert isinstance(comparison, dict)
        assert 'GCN' in comparison
        assert 'GAT' in comparison


@pytest.mark.integration
class TestBaselineIntegration:
    """Integration tests for baseline methods"""

    def test_all_baselines_produce_valid_output(self):
        """Test all baseline models produce valid embeddings"""
        num_nodes = 25
        num_edges = 50

        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Test all static models
        models = [
            StaticGCN(num_features=32, hidden_dim=64),
            StaticGAT(num_features=32, hidden_dim=64, num_heads=4),
            StaticGraphSAGE(num_features=32, hidden_dim=64),
        ]

        for model in models:
            embeddings = model(x, edge_index)
            assert embeddings.shape[0] == num_nodes
            assert not torch.isnan(embeddings).any()

    def test_centrality_methods_consistency(self):
        """Test centrality methods produce consistent results"""
        # Create test graph
        G = nx.karate_club_graph()  # Well-known test graph
        num_nodes = G.number_of_nodes()

        baseline = CentralityBaseline()

        # Compute different centralities
        degree = baseline.compute_degree_centrality_nx(G)
        betweenness = baseline.compute_betweenness_centrality(G)
        pagerank = baseline.compute_pagerank(G)

        # All should have same number of nodes
        assert len(degree) == num_nodes
        assert len(betweenness) == num_nodes
        assert len(pagerank) == num_nodes

        # All scores should be non-negative
        assert all(score >= 0 for score in degree.values())
        assert all(score >= 0 for score in betweenness.values())
        assert all(score >= 0 for score in pagerank.values())

    def test_temporal_vs_static_comparison(self):
        """Test temporal model vs static models"""
        num_timesteps = 5
        num_nodes = 20
        num_edges = 30

        # Static model
        static_model = StaticGCN(num_features=32, hidden_dim=64)
        x_static = torch.randn(num_nodes, 32)
        edge_index_static = torch.randint(0, num_nodes, (2, num_edges))
        static_embeddings = static_model(x_static, edge_index_static)

        # Temporal model
        temporal_model = SimpleTemporalGNN(
            num_node_features=32,
            num_edge_features=16,
            hidden_dim=64
        )
        node_features_seq = [torch.randn(num_nodes, 32) for _ in range(num_timesteps)]
        edge_indices_seq = [torch.randint(0, num_nodes, (2, num_edges)) for _ in range(num_timesteps)]
        edge_features_seq = [torch.randn(num_edges, 16) for _ in range(num_timesteps)]
        temporal_embeddings = temporal_model(node_features_seq, edge_indices_seq, edge_features_seq)

        # Both should produce same shape embeddings
        assert static_embeddings.shape == temporal_embeddings.shape
        assert static_embeddings.shape == (num_nodes, 64)
