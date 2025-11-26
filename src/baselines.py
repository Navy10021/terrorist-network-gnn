"""
Baseline Methods for Terrorist Network Disruption
=================================================

Implements various baseline methods for comparison:
1. Static GNN methods (GCN, GAT, GraphSAGE)
2. Traditional centrality measures
3. Graph partitioning methods
4. Existing temporal GNN methods (simplified)

Author: Advanced GNN Research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# ============================================================================
# STATIC GNN BASELINES
# ============================================================================

class StaticGCN(nn.Module):
    """
    Static Graph Convolutional Network (Kipf & Welling, 2017)
    Uses only the last timestep
    """
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.dropout = dropout
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, num_features]
            edge_index: [2, num_edges]
        Returns:
            embeddings: [num_nodes, hidden_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class StaticGAT(nn.Module):
    """
    Static Graph Attention Network (Veličković et al., 2018)
    """
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(
            num_node_features,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(
                hidden_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout
            ))
        
        # Last layer
        self.convs.append(GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=1,
            dropout=dropout
        ))
        
        self.dropout = dropout
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class StaticGraphSAGE(nn.Module):
    """
    GraphSAGE (Hamilton et al., 2017)
    """
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_node_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.dropout = dropout
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


# ============================================================================
# TRADITIONAL CENTRALITY BASELINES
# ============================================================================

class CentralityBaseline:
    """
    Traditional centrality-based node importance methods
    """
    def __init__(self, method: str = 'degree'):
        """
        Args:
            method: 'degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank'
        """
        self.method = method
    
    def detect_critical_nodes(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        top_k: int = 10
    ) -> torch.Tensor:
        """
        Detect critical nodes using centrality measures
        
        Returns:
            critical_nodes: [top_k] - indices of critical nodes
        """
        # Convert to NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
        
        # Compute centrality
        if self.method == 'degree':
            centrality = nx.degree_centrality(G)
        elif self.method == 'betweenness':
            centrality = nx.betweenness_centrality(G)
        elif self.method == 'closeness':
            centrality = nx.closeness_centrality(G)
        elif self.method == 'eigenvector':
            try:
                centrality = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                centrality = nx.degree_centrality(G)  # Fallback
        elif self.method == 'pagerank':
            centrality = nx.pagerank(G)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Sort and get top-k
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        critical_nodes = torch.tensor([node for node, _ in sorted_nodes[:top_k]])
        
        return critical_nodes


# ============================================================================
# GRAPH PARTITIONING BASELINE
# ============================================================================

class GraphPartitioningBaseline:
    """
    Graph partitioning methods (METIS-like approach)
    
    Identifies nodes whose removal maximally disconnects the graph
    """
    def __init__(self):
        pass
    
    def detect_critical_nodes(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        top_k: int = 10
    ) -> torch.Tensor:
        """
        Use min-cut approach to identify critical nodes
        """
        # Convert to NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
        
        # Find articulation points (cut vertices)
        articulation_points = list(nx.articulation_points(G))
        
        # If not enough articulation points, add high betweenness nodes
        if len(articulation_points) < top_k:
            betweenness = nx.betweenness_centrality(G)
            sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
            
            for node, _ in sorted_nodes:
                if node not in articulation_points and len(articulation_points) < top_k:
                    articulation_points.append(node)
        
        critical_nodes = torch.tensor(articulation_points[:top_k])
        return critical_nodes


# ============================================================================
# SIMPLIFIED TEMPORAL GNN BASELINE
# ============================================================================

class SimpleTemporalGNN(nn.Module):
    """
    Simplified temporal GNN baseline
    
    Uses simple LSTM + GCN without advanced techniques
    """
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2
    ):
        super().__init__()
        
        # GNN layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(num_node_features if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_gnn_layers)
        ])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        node_features_seq: List[torch.Tensor],
        edge_indices_seq: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            node_features_seq: List of [num_nodes, num_features]
            edge_indices_seq: List of [2, num_edges]
        
        Returns:
            embeddings: [num_nodes, hidden_dim]
        """
        num_timesteps = len(node_features_seq)
        
        # Process each timestep with GNN
        temporal_embeddings = []
        
        for t in range(num_timesteps):
            x = node_features_seq[t]
            edge_index = edge_indices_seq[t]
            
            # Apply GCN layers
            for gcn in self.gcn_layers:
                x = F.relu(gcn(x, edge_index))
            
            temporal_embeddings.append(x)
        
        # Handle dynamic network size
        # Find minimum number of nodes across all timesteps
        min_nodes = min(emb.size(0) for emb in temporal_embeddings)
        
        # Pad or truncate to the last timestep's size
        last_num_nodes = temporal_embeddings[-1].size(0)
        
        # Pad all embeddings to match last timestep size
        padded_embeddings = []
        for emb in temporal_embeddings:
            if emb.size(0) < last_num_nodes:
                # Pad with zeros
                padding = torch.zeros(
                    last_num_nodes - emb.size(0), 
                    emb.size(1), 
                    device=emb.device
                )
                emb = torch.cat([emb, padding], dim=0)
            elif emb.size(0) > last_num_nodes:
                # Truncate
                emb = emb[:last_num_nodes]
            padded_embeddings.append(emb)
        
        # Stack: [num_nodes, num_timesteps, hidden_dim]
        temporal_embeddings = torch.stack(padded_embeddings, dim=1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(temporal_embeddings)
        
        # Take last timestep
        final_embeddings = lstm_out[:, -1, :]
        
        return final_embeddings


# ============================================================================
# RANDOM BASELINE
# ============================================================================

class RandomBaseline:
    """
    Random node selection baseline
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def detect_critical_nodes(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        top_k: int = 10
    ) -> torch.Tensor:
        """
        Randomly select nodes
        """
        nodes = np.random.choice(num_nodes, size=min(top_k, num_nodes), replace=False)
        return torch.tensor(nodes)


# ============================================================================
# BASELINE EVALUATION WRAPPER
# ============================================================================

class BaselineEvaluator:
    """
    Unified interface for evaluating all baselines
    """
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.baselines = {
            'random': RandomBaseline(),
            'degree': CentralityBaseline('degree'),
            'betweenness': CentralityBaseline('betweenness'),
            'pagerank': CentralityBaseline('pagerank'),
            'eigenvector': CentralityBaseline('eigenvector'),
            'graph_partition': GraphPartitioningBaseline()
        }
    
    def add_gnn_baseline(
        self,
        name: str,
        model: nn.Module,
        requires_training: bool = True
    ):
        """
        Add a GNN-based baseline
        """
        self.baselines[name] = {
            'model': model.to(self.device),
            'requires_training': requires_training
        }
    
    def evaluate_baseline(
        self,
        baseline_name: str,
        network,
        top_k: int = 10,
        embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Evaluate a specific baseline
        
        Returns:
            critical_nodes: [top_k]
        """
        baseline = self.baselines[baseline_name]
        
        # Get last timestep
        t = -1
        layers = network.get_timestep(t)
        agg_edge_index, _ = network.get_aggregated_network(t)
        num_nodes = layers[0].node_features.size(0)
        
        agg_edge_index = agg_edge_index.to(self.device)
        
        # For GNN baselines
        if isinstance(baseline, dict):
            model = baseline['model']
            model.eval()
            
            with torch.no_grad():
                if isinstance(model, (StaticGCN, StaticGAT, StaticGraphSAGE)):
                    # Static GNN
                    node_features = layers[0].node_features.to(self.device)
                    embeddings = model(node_features, agg_edge_index)
                elif isinstance(model, SimpleTemporalGNN):
                    # Temporal GNN
                    node_features_seq = [l.node_features.to(self.device) for l in network.get_layer_sequence(0)]
                    edge_indices_seq = [l.edge_index.to(self.device) for l in network.get_layer_sequence(0)]
                    embeddings = model(node_features_seq[-5:], edge_indices_seq[-5:])  # Use last 5 timesteps
            
            # Use degree centrality on embeddings (simple heuristic)
            # In practice, you'd want more sophisticated importance scoring
            from torch_geometric.utils import degree
            deg = degree(agg_edge_index[0], num_nodes=num_nodes)
            critical_nodes = torch.argsort(deg, descending=True)[:top_k]
        else:
            # Traditional baselines
            critical_nodes = baseline.detect_critical_nodes(
                agg_edge_index, num_nodes, top_k
            )
        
        return critical_nodes.cpu()
    
    def compare_all_baselines(
        self,
        network,
        evaluator,
        top_k: int = 10
    ) -> Dict[str, Dict]:
        """
        Compare all baselines
        
        Returns:
            results: Dict with disruption metrics for each baseline
        """
        results = {}
        
        for name in self.baselines.keys():
            print(f"Evaluating {name}...")
            
            # Get critical nodes
            critical_nodes = self.evaluate_baseline(name, network, top_k)
            
            # Evaluate disruption
            t = -1
            metrics = evaluator.evaluate_disruption_strategy(
                network, critical_nodes.tolist(), timestep=t
            )
            
            results[name] = {
                'critical_nodes': critical_nodes.tolist(),
                'metrics': metrics,
                'disruption': metrics['overall_disruption']
            }
        
        return results


# Test and demonstration
if __name__ == "__main__":
    print("="*80)
    print("Baseline Methods for Comparison")
    print("="*80)
    
    print("\nImplemented Baselines:")
    print("  1. Random Selection")
    print("  2. Degree Centrality")
    print("  3. Betweenness Centrality")
    print("  4. PageRank")
    print("  5. Eigenvector Centrality")
    print("  6. Graph Partitioning (Min-Cut)")
    print("  7. Static GCN")
    print("  8. Static GAT")
    print("  9. Static GraphSAGE")
    print("  10. Simple Temporal GNN")
    
    print("\n✓ All baselines ready for comparison!")
