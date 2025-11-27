"""
Pytest configuration and shared fixtures
"""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def set_random_seed():
    """Set random seed for reproducibility across all tests"""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture
def device():
    """Get the device for testing (CPU or CUDA)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_graph_data():
    """Create sample graph data for testing"""
    num_nodes = 10
    num_edges = 15
    num_node_features = 64
    num_edge_features = 32

    node_features = torch.randn(num_nodes, num_node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, num_edge_features)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }


@pytest.fixture
def sample_temporal_data():
    """Create sample temporal graph data for testing"""
    batch_size = 2
    num_timesteps = 5
    num_nodes = 10
    num_edges = 15
    num_node_features = 64
    num_edge_features = 32

    temporal_graphs = []
    for t in range(num_timesteps):
        node_features = torch.randn(batch_size, num_nodes, num_node_features)
        edge_index = torch.randint(0, num_nodes, (batch_size, 2, num_edges))
        edge_attr = torch.randn(batch_size, num_edges, num_edge_features)
        timestamps = torch.tensor([t] * batch_size).float()

        temporal_graphs.append(
            {
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "timestamps": timestamps,
            }
        )

    return temporal_graphs


@pytest.fixture
def network_config():
    """Create a standard network configuration for testing"""
    from src.terrorist_network_dataset import NetworkConfig

    return NetworkConfig(
        initial_nodes=50,
        max_nodes=80,
        recruitment_rate=0.05,
        dropout_rate=0.02,
        initial_edges_per_node=3,
        communication_density=0.3,
        financial_flow_prob=0.15,
        ideological_similarity_threshold=0.6,
        operational_connection_prob=0.25,
    )


@pytest.fixture
def model_config():
    """Create a standard model configuration for testing"""
    return {
        "num_node_features": 64,
        "num_edge_features": 32,
        "hidden_dim": 128,
        "num_temporal_layers": 3,
        "num_graph_layers": 3,
        "num_attention_heads": 8,
        "memory_size": 100,
        "dropout": 0.1,
    }


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
