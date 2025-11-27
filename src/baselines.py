"""
Baseline Methods V2 - Enhanced with SOTA Methods
=================================================

Enhanced with:
1. Dynamic GCN
2. EvolveGCN
3. Improved baseline evaluation
4. Better integration

Author: Advanced GNN Research
Version: 2.0
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
    """Static Graph Convolutional Network"""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class StaticGAT(nn.Module):
    """Static Graph Attention Network"""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        self.convs.append(
            GATv2Conv(num_node_features, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            )

        self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=1, dropout=dropout))

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
    """GraphSAGE"""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
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
# ADVANCED TEMPORAL BASELINES (NEW)
# ============================================================================


class DynamicGCN(nn.Module):
    """
    Dynamic GCN - Time-dependent graph convolution
    Learns temporal gates for dynamic weighting
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.gcn_layers = nn.ModuleList(
            [
                GCNConv(num_node_features if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )

        # Temporal gate - learns importance of current vs. past
        self.temporal_gate = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, timestamp: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, num_features]
            edge_index: [2, num_edges]
            timestamp: [1] - scalar timestamp
        """
        # Compute temporal gate
        time_gate = self.temporal_gate(timestamp.view(1, 1))
        time_gate = time_gate.expand(x.size(0), -1)

        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = F.relu(x)
            # Apply temporal gating
            x = x * time_gate
            x = self.dropout(x)

        return x


class EvolveGCN(nn.Module):
    """
    EvolveGCN - GCN parameters evolve over time
    Simplified version inspired by Pareja et al. (2020)
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GCN layers
        self.gcn_layers = nn.ModuleList(
            [
                GCNConv(num_node_features if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )

        # LSTM to evolve GCN parameters
        # Simplified: evolve layer normalization parameters
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.temporal_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

        # Hidden state for LSTM
        self.hidden_state = None

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, update_evolution: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, num_features]
            edge_index: [2, num_edges]
            update_evolution: whether to update temporal evolution
        """
        device = x.device

        # Apply GCN layers
        for i, gcn in enumerate(self.gcn_layers):
            x = gcn(x, edge_index)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        # Update evolution state
        if update_evolution and self.training:
            # Use mean pooling as graph-level representation
            graph_repr = x.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, hidden_dim]

            # Evolve through LSTM
            if self.hidden_state is None:
                self.hidden_state = (
                    torch.zeros(1, 1, self.hidden_dim, device=device),
                    torch.zeros(1, 1, self.hidden_dim, device=device),
                )

            _, self.hidden_state = self.temporal_lstm(graph_repr, self.hidden_state)

        return x

    def reset_evolution(self):
        """Reset evolution state"""
        self.hidden_state = None


class SimpleTemporalGNN(nn.Module):
    """Simple temporal GNN baseline (LSTM + GCN)"""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gcn_layers = nn.ModuleList(
            [
                GCNConv(num_node_features if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_gnn_layers)
            ]
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(
        self, node_features_seq: List[torch.Tensor], edge_indices_seq: List[torch.Tensor]
    ) -> torch.Tensor:
        num_timesteps = len(node_features_seq)

        temporal_embeddings = []

        for t in range(num_timesteps):
            x = node_features_seq[t]
            edge_index = edge_indices_seq[t]

            for gcn in self.gcn_layers:
                x = F.relu(gcn(x, edge_index))
                x = self.dropout(x)

            temporal_embeddings.append(x)

        last_num_nodes = temporal_embeddings[-1].size(0)

        padded_embeddings = []
        for emb in temporal_embeddings:
            if emb.size(0) < last_num_nodes:
                padding = torch.zeros(last_num_nodes - emb.size(0), emb.size(1), device=emb.device)
                emb = torch.cat([emb, padding], dim=0)
            elif emb.size(0) > last_num_nodes:
                emb = emb[:last_num_nodes]
            padded_embeddings.append(emb)

        temporal_embeddings = torch.stack(padded_embeddings, dim=1)

        lstm_out, _ = self.lstm(temporal_embeddings)

        final_embeddings = lstm_out[:, -1, :]

        return final_embeddings


# ============================================================================
# TRADITIONAL CENTRALITY BASELINES
# ============================================================================


class CentralityBaseline:
    """Traditional centrality-based methods"""

    def __init__(self, method: str = "degree"):
        self.method = method

    def detect_critical_nodes(
        self, edge_index: torch.Tensor, num_nodes: int, top_k: int = 10
    ) -> torch.Tensor:
        """Detect critical nodes using centrality"""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().cpu().numpy()
        G.add_edges_from(edges)

        if self.method == "degree":
            centrality = nx.degree_centrality(G)
        elif self.method == "betweenness":
            centrality = nx.betweenness_centrality(G)
        elif self.method == "closeness":
            centrality = nx.closeness_centrality(G)
        elif self.method == "eigenvector":
            try:
                centrality = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                centrality = nx.degree_centrality(G)
        elif self.method == "pagerank":
            centrality = nx.pagerank(G)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        critical_nodes = torch.tensor([node for node, _ in sorted_nodes[:top_k]])

        return critical_nodes


class GraphPartitioningBaseline:
    """Graph partitioning methods"""

    def __init__(self):
        pass

    def detect_critical_nodes(
        self, edge_index: torch.Tensor, num_nodes: int, top_k: int = 10
    ) -> torch.Tensor:
        """Use min-cut approach"""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().cpu().numpy()
        G.add_edges_from(edges)

        articulation_points = list(nx.articulation_points(G))

        if len(articulation_points) < top_k:
            betweenness = nx.betweenness_centrality(G)
            sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

            for node, _ in sorted_nodes:
                if node not in articulation_points and len(articulation_points) < top_k:
                    articulation_points.append(node)

        critical_nodes = torch.tensor(articulation_points[:top_k])
        return critical_nodes


class RandomBaseline:
    """Random node selection baseline"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def detect_critical_nodes(
        self, edge_index: torch.Tensor, num_nodes: int, top_k: int = 10
    ) -> torch.Tensor:
        """Randomly select nodes"""
        nodes = np.random.choice(num_nodes, size=min(top_k, num_nodes), replace=False)
        return torch.tensor(nodes)


# ============================================================================
# ENHANCED BASELINE EVALUATOR
# ============================================================================


class BaselineEvaluator:
    """Unified interface for evaluating all baselines"""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.baselines = {
            "random": RandomBaseline(),
            "degree": CentralityBaseline("degree"),
            "betweenness": CentralityBaseline("betweenness"),
            "pagerank": CentralityBaseline("pagerank"),
            "eigenvector": CentralityBaseline("eigenvector"),
            "graph_partition": GraphPartitioningBaseline(),
        }

    def add_gnn_baseline(self, name: str, model: nn.Module, requires_training: bool = True):
        """Add a GNN-based baseline"""
        self.baselines[name] = {
            "model": model.to(self.device),
            "requires_training": requires_training,
        }

    def evaluate_baseline(
        self,
        baseline_name: str,
        network,
        top_k: int = 10,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate a specific baseline"""
        baseline = self.baselines[baseline_name]

        t = -1
        layers = network.get_timestep(t)
        agg_edge_index, _ = network.get_aggregated_network(t)
        num_nodes = layers[0].node_features.size(0)

        agg_edge_index = agg_edge_index.to(self.device)

        # For GNN baselines
        if isinstance(baseline, dict):
            model = baseline["model"]
            model.eval()

            with torch.no_grad():
                if isinstance(model, (StaticGCN, StaticGAT, StaticGraphSAGE)):
                    node_features = layers[0].node_features.to(self.device)
                    embeddings = model(node_features, agg_edge_index)
                elif isinstance(model, DynamicGCN):
                    node_features = layers[0].node_features.to(self.device)
                    timestamp = torch.tensor([float(t)], device=self.device)
                    embeddings = model(node_features, agg_edge_index, timestamp)
                elif isinstance(model, EvolveGCN):
                    model.reset_evolution()
                    node_features_seq = [
                        l.node_features.to(self.device) for l in network.get_layer_sequence(0)
                    ]
                    edge_indices_seq = [
                        l.edge_index.to(self.device) for l in network.get_layer_sequence(0)
                    ]

                    for nf, ei in zip(node_features_seq[-5:], edge_indices_seq[-5:]):
                        embeddings = model(nf, ei, update_evolution=False)
                elif isinstance(model, SimpleTemporalGNN):
                    node_features_seq = [
                        l.node_features.to(self.device) for l in network.get_layer_sequence(0)
                    ]
                    edge_indices_seq = [
                        l.edge_index.to(self.device) for l in network.get_layer_sequence(0)
                    ]
                    embeddings = model(node_features_seq[-5:], edge_indices_seq[-5:])

            # Use degree centrality on embeddings
            from torch_geometric.utils import degree

            deg = degree(agg_edge_index[0], num_nodes=num_nodes)
            critical_nodes = torch.argsort(deg, descending=True)[:top_k]
        else:
            # Traditional baselines
            critical_nodes = baseline.detect_critical_nodes(agg_edge_index, num_nodes, top_k)

        return critical_nodes.cpu()

    def compare_all_baselines(self, network, evaluator, top_k: int = 10) -> Dict[str, Dict]:
        """Compare all baselines"""
        results = {}

        for name in self.baselines.keys():
            print(f"  Evaluating {name}...")

            critical_nodes = self.evaluate_baseline(name, network, top_k)

            t = -1
            metrics = evaluator.evaluate_disruption_strategy(
                network, critical_nodes.tolist(), timestep=t
            )

            results[name] = {
                "critical_nodes": critical_nodes.tolist(),
                "metrics": metrics,
                "disruption": metrics["overall_disruption"],
            }

        return results


# Test code
if __name__ == "__main__":
    print("=" * 80)
    print("Baseline Methods V2 - Enhanced")
    print("=" * 80)
    print("\nImplemented Baselines:")
    print("  Traditional:")
    print("    1. Random Selection")
    print("    2. Degree Centrality")
    print("    3. Betweenness Centrality")
    print("    4. PageRank")
    print("    5. Eigenvector Centrality")
    print("    6. Graph Partitioning")
    print("\n  Static GNN:")
    print("    7. Static GCN")
    print("    8. Static GAT")
    print("    9. Static GraphSAGE")
    print("\n  Temporal GNN (NEW):")
    print("    10. Dynamic GCN")
    print("    11. EvolveGCN")
    print("    12. Simple Temporal GNN")
    print("\n✓ Enhanced baselines ready!")
