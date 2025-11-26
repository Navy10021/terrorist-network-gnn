"""
Test Suite for Advanced Temporal GNN
====================================

Example test file demonstrating testing patterns.

Author: Yoon-Seop Lee
"""

import pytest
import torch
import numpy as np
from src.advanced_tgnn import AdvancedTemporalGNN, AdaptiveTimeEncoding
from src.terrorist_network_disruption import CriticalNodeDetector, MultiLayerTemporalGNN


class TestAdvancedTemporalGNN:
    """Test suite for AdvancedTemporalGNN model"""
    
    @pytest.fixture
    def device(self):
        """Setup device for testing"""
        return torch.device('cpu')
    
    @pytest.fixture
    def model_config(self):
        """Model configuration for testing"""
        return {
            'num_node_features': 32,
            'num_edge_features': 16,
            'hidden_dim': 64,
            'num_temporal_layers': 2,
            'num_graph_layers': 2,
            'num_attention_heads': 4,
            'memory_size': 50
        }
    
    @pytest.fixture
    def model(self, model_config, device):
        """Create model instance"""
        model = AdvancedTemporalGNN(**model_config)
        return model.to(device)
    
    def test_model_initialization(self, model, model_config):
        """Test model initializes correctly"""
        assert model is not None
        assert model.hidden_dim == model_config['hidden_dim']
        assert len(model.temporal_attention) == model_config['num_temporal_layers']
        assert len(model.graph_layers) == model_config['num_graph_layers']
    
    def test_forward_pass(self, model, model_config, device):
        """Test forward pass with synthetic data"""
        num_timesteps = 5
        num_nodes = 20
        num_edges = 40
        
        # Generate synthetic temporal graph data
        node_features_seq = [
            torch.randn(num_nodes, model_config['num_node_features'], device=device)
            for _ in range(num_timesteps)
        ]
        edge_indices_seq = [
            torch.randint(0, num_nodes, (2, num_edges), device=device)
            for _ in range(num_timesteps)
        ]
        edge_features_seq = [
            torch.randn(num_edges, model_config['num_edge_features'], device=device)
            for _ in range(num_timesteps)
        ]
        timestamps = torch.arange(num_timesteps, dtype=torch.float32, device=device)
        
        # Forward pass
        embeddings = model(
            node_features_seq, edge_indices_seq, edge_features_seq, timestamps
        )
        
        # Check output shape
        assert embeddings.shape == (num_nodes, model_config['hidden_dim'])
        assert not torch.isnan(embeddings).any()
    
    def test_reconstruction(self, model, model_config, device):
        """Test node feature reconstruction"""
        num_nodes = 20
        embeddings = torch.randn(num_nodes, model_config['hidden_dim'], device=device)
        
        reconstructed = model.reconstruct_features(embeddings)
        
        assert reconstructed.shape == (num_nodes, model_config['num_node_features'])
        assert not torch.isnan(reconstructed).any()


class TestCriticalNodeDetector:
    """Test suite for CriticalNodeDetector"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return CriticalNodeDetector()
    
    @pytest.fixture
    def sample_graph(self):
        """Create sample graph"""
        num_nodes = 30
        edges = []
        # Create a simple connected graph
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
        # Add some random edges
        for _ in range(20):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                edges.append([src, dst])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index, num_nodes
    
    def test_degree_centrality(self, detector, sample_graph):
        """Test degree centrality computation"""
        edge_index, num_nodes = sample_graph
        
        centrality = detector.compute_degree_centrality(edge_index, num_nodes)
        
        assert centrality.shape == (num_nodes,)
        assert (centrality >= 0).all() and (centrality <= 1).all()
    
    def test_betweenness_centrality(self, detector, sample_graph):
        """Test betweenness centrality computation"""
        edge_index, num_nodes = sample_graph
        
        centrality = detector.compute_betweenness_centrality(edge_index, num_nodes)
        
        assert centrality.shape == (num_nodes,)
        assert (centrality >= 0).all() and (centrality <= 1).all()
    
    def test_critical_node_detection(self, detector, sample_graph):
        """Test critical node detection"""
        edge_index, num_nodes = sample_graph
        embeddings = torch.randn(num_nodes, 64)
        
        critical_nodes, scores = detector.detect_critical_nodes(
            edge_index, num_nodes, embeddings, top_k=5
        )
        
        assert critical_nodes.shape == (5,)
        assert len(scores) > 0
        assert all(0 <= node < num_nodes for node in critical_nodes.tolist())


class TestAdaptiveTimeEncoding:
    """Test suite for AdaptiveTimeEncoding"""
    
    def test_time_encoding_shape(self):
        """Test time encoding output shape"""
        d_model = 64
        encoder = AdaptiveTimeEncoding(d_model)
        
        # Single timestamp
        timestamps = torch.tensor([1.0])
        encoding = encoder(timestamps)
        assert encoding.shape == (1, d_model)
        
        # Multiple timestamps
        timestamps = torch.tensor([1.0, 2.0, 3.0])
        encoding = encoder(timestamps)
        assert encoding.shape == (3, d_model)
    
    def test_time_encoding_values(self):
        """Test time encoding produces valid values"""
        encoder = AdaptiveTimeEncoding(64)
        timestamps = torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0])
        
        encoding = encoder(timestamps)
        
        # Check no NaN or Inf
        assert not torch.isnan(encoding).any()
        assert not torch.isinf(encoding).any()
        
        # Check different timestamps produce different encodings
        assert not torch.allclose(encoding[0], encoding[1])
        assert not torch.allclose(encoding[1], encoding[2])


@pytest.mark.gpu
class TestGPUSupport:
    """Test GPU-specific functionality"""
    
    def test_model_on_gpu(self):
        """Test model runs on GPU if available"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        device = torch.device('cuda')
        model = AdvancedTemporalGNN(
            num_node_features=32,
            num_edge_features=16,
            hidden_dim=64
        ).to(device)
        
        # Test forward pass on GPU
        node_features_seq = [torch.randn(10, 32, device=device) for _ in range(3)]
        edge_indices_seq = [torch.randint(0, 10, (2, 20), device=device) for _ in range(3)]
        edge_features_seq = [torch.randn(20, 16, device=device) for _ in range(3)]
        timestamps = torch.arange(3, dtype=torch.float32, device=device)
        
        embeddings = model(
            node_features_seq, edge_indices_seq, edge_features_seq, timestamps
        )
        
        assert embeddings.device.type == 'cuda'
        assert embeddings.shape == (10, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
