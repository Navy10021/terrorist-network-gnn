"""
Test Suite for Baseline Methods
================================

Simplified tests that match actual implementation.

Author: Yoon-Seop Lee
"""

import pytest
import torch
import networkx as nx
from src.baselines import (
    StaticGCN,
    StaticGAT,
    StaticGraphSAGE,
    SimpleTemporalGNN,
    CentralityBaseline
)


class TestStaticGCN:
    """Test suite for Static GCN baseline"""

    def test_model_initialization(self):
        """Test GCN initializes correctly"""
        model = StaticGCN(num_node_features=32, hidden_dim=64)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass with synthetic data"""
        model = StaticGCN(num_node_features=32, hidden_dim=64)
        num_nodes = 20

        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, 40))

        embeddings = model(x, edge_index)
        assert embeddings.shape[0] == num_nodes


class TestStaticGAT:
    """Test suite for Static GAT baseline"""

    def test_model_initialization(self):
        """Test GAT initializes correctly"""
        model = StaticGAT(num_node_features=32, hidden_dim=64, num_heads=4)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass with attention mechanism"""
        model = StaticGAT(num_node_features=32, hidden_dim=64, num_heads=4)
        num_nodes = 20

        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, 40))

        embeddings = model(x, edge_index)
        assert embeddings.shape[0] == num_nodes


class TestStaticGraphSAGE:
    """Test suite for Static GraphSAGE baseline"""

    def test_model_initialization(self):
        """Test GraphSAGE initializes correctly"""
        model = StaticGraphSAGE(num_node_features=32, hidden_dim=64)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass"""
        model = StaticGraphSAGE(num_node_features=32, hidden_dim=64)
        num_nodes = 20

        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, 40))

        embeddings = model(x, edge_index)
        assert embeddings.shape[0] == num_nodes


class TestSimpleTemporalGNN:
    """Test suite for Simple Temporal GNN baseline"""

    def test_model_initialization(self):
        """Test simple T-GNN initializes correctly"""
        model = SimpleTemporalGNN(num_node_features=32, hidden_dim=64)
        assert model is not None


class TestCentralityBaseline:
    """Test suite for Centrality-based baseline"""

    def test_baseline_initialization(self):
        """Test centrality baseline initializes correctly"""
        baseline = CentralityBaseline(method='degree')
        assert baseline is not None
        assert baseline.method == 'degree'

    def test_detect_critical_nodes(self):
        """Test critical node detection with centrality"""
        baseline = CentralityBaseline(method='degree')
        num_nodes = 20
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0]
        ], dtype=torch.long)

        critical_nodes, scores = baseline.detect_critical_nodes(
            edge_index=edge_index,
            num_nodes=num_nodes,
            top_k=5
        )

        assert critical_nodes.size(0) == 5
        assert scores.size(0) == num_nodes
