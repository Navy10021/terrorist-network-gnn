"""
Terrorist Network Disruption V2 - Enhanced Version
===================================================

Enhanced with:
1. Multi-layer centrality computation
2. Temporal importance tracking
3. Advanced disruption optimization
4. Improved resilience prediction

Author: Advanced GNN Research
Version: 2.0
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from torch_geometric.utils import degree, remove_self_loops, to_dense_adj

from .advanced_tgnn import AdvancedTemporalGNN

# ============================================================================
# MULTI-LAYER NETWORK SUPPORT
# ============================================================================


@dataclass
class NetworkLayer:
    """Represents one layer of a multi-layer terrorist network"""

    name: str
    layer_type: str
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_weights: torch.Tensor
    metadata: Dict


class MultiLayerTemporalNetwork:
    """Multi-layer temporal network for terrorist organizations"""

    def __init__(self, num_nodes: int, num_layers: int = 5):
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.layer_names = ["physical", "digital", "financial", "ideological", "operational"]
        self.layers_history = []

    def add_timestep(self, layers: List[NetworkLayer]):
        """Add network state at a timestep"""
        assert len(layers) == self.num_layers
        self.layers_history.append(layers)

    def get_timestep(self, t: int) -> List[NetworkLayer]:
        """Get all layers at timestep t"""
        return self.layers_history[t]

    def get_layer_sequence(self, layer_idx: int) -> List[NetworkLayer]:
        """Get temporal sequence of a specific layer"""
        return [timestep[layer_idx] for timestep in self.layers_history]

    def get_aggregated_network(self, t: int, weights: Optional[List[float]] = None):
        """Aggregate multiple layers into single network"""
        if weights is None:
            weights = [1.0] * self.num_layers

        layers = self.get_timestep(t)
        aggregated_edges = defaultdict(float)

        for layer, weight in zip(layers, weights):
            edge_index = layer.edge_index
            edge_weights = layer.edge_weights

            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                edge_weight = edge_weights[i].item() if edge_weights is not None else 1.0
                aggregated_edges[(src, dst)] += weight * edge_weight

        edges = list(aggregated_edges.keys())
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_weights = torch.tensor([aggregated_edges[e] for e in edges])
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty(0)

        return edge_index, edge_weights


class MultiLayerTemporalGNN(nn.Module):
    """Temporal GNN that processes multi-layer networks"""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int,
        num_layers: int = 5,
        layer_fusion: str = "attention",
        **kwargs,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.layer_fusion = layer_fusion

        # Per-layer GNNs
        self.layer_gnns = nn.ModuleList(
            [
                AdvancedTemporalGNN(num_node_features, num_edge_features, hidden_dim, **kwargs)
                for _ in range(num_layers)
            ]
        )

        # Layer fusion mechanism
        if layer_fusion == "attention":
            self.layer_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.layer_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        elif layer_fusion == "weighted_sum":
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        self.output_dim = hidden_dim

    def forward(
        self, multi_layer_network: MultiLayerTemporalNetwork, timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Process multi-layer temporal network"""
        num_timesteps = len(multi_layer_network.layers_history)

        layer_embeddings = []

        for layer_idx in range(self.num_layers):
            layer_sequence = multi_layer_network.get_layer_sequence(layer_idx)

            node_features_seq = [layer.node_features for layer in layer_sequence]
            edge_indices_seq = [layer.edge_index for layer in layer_sequence]
            edge_features_seq = [layer.edge_features for layer in layer_sequence]

            embeddings = self.layer_gnns[layer_idx](
                node_features_seq, edge_indices_seq, edge_features_seq, timestamps
            )
            layer_embeddings.append(embeddings)

        # Fuse layer embeddings
        if self.layer_fusion == "attention":
            layer_embeddings = torch.stack(layer_embeddings, dim=1)
            batch_size = layer_embeddings.size(0)
            query = self.layer_query.expand(batch_size, -1, -1)

            fused, attn_weights = self.layer_attention(query, layer_embeddings, layer_embeddings)
            fused = fused.squeeze(1)

        elif self.layer_fusion == "concat":
            fused = torch.cat(layer_embeddings, dim=-1)

        elif self.layer_fusion == "weighted_sum":
            weights = F.softmax(self.layer_weights, dim=0)
            fused = sum(w * emb for w, emb in zip(weights, layer_embeddings))

        return fused

    def reconstruct_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reconstruct node features from embeddings"""
        return self.layer_gnns[0].reconstruct_features(embeddings)


# ============================================================================
# ENHANCED CRITICAL NODE DETECTION
# ============================================================================


class EnhancedCriticalNodeDetector:
    """
    Enhanced critical node detector with multi-layer and temporal analysis
    """

    def __init__(self, importance_metrics: List[str] = None):
        if importance_metrics is None:
            self.importance_metrics = [
                "degree_centrality",
                "betweenness_centrality",
                "eigenvector_centrality",
                "pagerank",
                "structural_holes",
                "gnn_importance",
                "multilayer_centrality",
                "temporal_importance",
            ]
        else:
            self.importance_metrics = importance_metrics

    def compute_multilayer_centrality(
        self, network: MultiLayerTemporalNetwork, timestep: int = -1
    ) -> torch.Tensor:
        """
        Compute centrality across all layers with importance weighting

        Layer weights based on operational impact:
        - Operational: 30% (direct attack capability)
        - Financial: 25% (resource flow)
        - Physical: 20% (coordination)
        - Digital: 15% (communication)
        - Ideological: 10% (influence)
        """
        layers = network.get_timestep(timestep)
        num_nodes = layers[0].node_features.size(0)

        # Layer importance weights
        layer_weights = {
            "operational": 0.30,
            "financial": 0.25,
            "physical": 0.20,
            "digital": 0.15,
            "ideological": 0.10,
        }

        combined_importance = torch.zeros(num_nodes)

        for layer in layers:
            # Compute degree centrality for this layer
            layer_importance = self.compute_degree_centrality(layer.edge_index, num_nodes)

            # Weight by layer importance
            weight = layer_weights.get(layer.name, 0.1)
            combined_importance += weight * layer_importance

        return combined_importance

    def compute_temporal_importance(
        self, network: MultiLayerTemporalNetwork, window_size: int = 5
    ) -> torch.Tensor:
        """
        Compute node importance over temporal window

        Recent activity weighted more heavily
        """
        num_timesteps = len(network.layers_history)
        start_t = max(0, num_timesteps - window_size)

        temporal_scores = []
        for t in range(start_t, num_timesteps):
            score_t = self.compute_multilayer_centrality(network, t)
            temporal_scores.append(score_t)

        # Time-weighted average (more recent = more weight)
        weights = torch.linspace(0.5, 1.0, len(temporal_scores))
        weighted_avg = sum(w * s for w, s in zip(weights, temporal_scores))

        return weighted_avg / weights.sum()

    def compute_degree_centrality(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Degree centrality"""
        deg = degree(edge_index[0], num_nodes=num_nodes)
        return deg / max(num_nodes - 1, 1)

    def compute_betweenness_centrality(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Betweenness centrality"""
        G = self._to_networkx(edge_index, num_nodes)
        betweenness = nx.betweenness_centrality(G)
        device = edge_index.device
        return torch.tensor([betweenness[i] for i in range(num_nodes)], device=device)

    def compute_eigenvector_centrality(
        self, edge_index: torch.Tensor, num_nodes: int, max_iter: int = 100
    ) -> torch.Tensor:
        """Eigenvector centrality"""
        G = self._to_networkx(edge_index, num_nodes)
        device = edge_index.device
        try:
            eigen_cent = nx.eigenvector_centrality(G, max_iter=max_iter)
            return torch.tensor([eigen_cent[i] for i in range(num_nodes)], device=device)
        except:
            return self.compute_degree_centrality(edge_index, num_nodes)

    def compute_pagerank(
        self, edge_index: torch.Tensor, num_nodes: int, alpha: float = 0.85
    ) -> torch.Tensor:
        """PageRank"""
        G = self._to_networkx(edge_index, num_nodes)
        pagerank = nx.pagerank(G, alpha=alpha)
        device = edge_index.device
        return torch.tensor([pagerank[i] for i in range(num_nodes)], device=device)

    def compute_structural_holes(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Structural holes - nodes bridging communities"""
        G = self._to_networkx(edge_index, num_nodes)
        device = edge_index.device
        try:
            constraint = nx.constraint(G)
            importance = torch.tensor(
                [1.0 - constraint.get(i, 1.0) for i in range(num_nodes)], device=device
            )
            return importance
        except:
            return torch.zeros(num_nodes, device=device)

    def compute_gnn_importance(
        self, embeddings: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """GNN-based importance from learned embeddings"""
        embedding_norm = embeddings.norm(dim=1)

        num_embedding_nodes = embeddings.size(0)
        valid_mask = (edge_index[0] < num_embedding_nodes) & (edge_index[1] < num_embedding_nodes)
        valid_edge_index = edge_index[:, valid_mask]

        influence = torch.zeros(num_nodes, device=embeddings.device)
        for i in range(valid_edge_index.size(1)):
            src, dst = valid_edge_index[:, i]
            similarity = F.cosine_similarity(
                embeddings[src].unsqueeze(0), embeddings[dst].unsqueeze(0)
            )
            influence[src] += similarity.squeeze() * embeddings[dst].norm()

        importance = embedding_norm + influence[:num_embedding_nodes]
        return importance / importance.sum()

    def detect_critical_nodes(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        embeddings: Optional[torch.Tensor] = None,
        network: Optional[MultiLayerTemporalNetwork] = None,
        top_k: int = 10,
        aggregation: str = "weighted_sum",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced critical node detection with multi-layer and temporal analysis
        """
        importance_scores = {}

        # Traditional metrics
        if "degree_centrality" in self.importance_metrics:
            importance_scores["degree"] = self.compute_degree_centrality(edge_index, num_nodes)

        if "betweenness_centrality" in self.importance_metrics:
            importance_scores["betweenness"] = self.compute_betweenness_centrality(
                edge_index, num_nodes
            )

        if "eigenvector_centrality" in self.importance_metrics:
            importance_scores["eigenvector"] = self.compute_eigenvector_centrality(
                edge_index, num_nodes
            )

        if "pagerank" in self.importance_metrics:
            importance_scores["pagerank"] = self.compute_pagerank(edge_index, num_nodes)

        if "structural_holes" in self.importance_metrics:
            importance_scores["structural_holes"] = self.compute_structural_holes(
                edge_index, num_nodes
            )

        # GNN-based
        if "gnn_importance" in self.importance_metrics and embeddings is not None:
            importance_scores["gnn"] = self.compute_gnn_importance(
                embeddings, edge_index, num_nodes
            )

        # Multi-layer analysis (NEW)
        if "multilayer_centrality" in self.importance_metrics and network is not None:
            importance_scores["multilayer"] = self.compute_multilayer_centrality(network)

        # Temporal analysis (NEW)
        if "temporal_importance" in self.importance_metrics and network is not None:
            importance_scores["temporal"] = self.compute_temporal_importance(network)

        # Aggregate scores
        if aggregation == "weighted_sum":
            # Enhanced weights favoring multi-layer and temporal
            weights = {
                "degree": 0.10,
                "betweenness": 0.15,
                "eigenvector": 0.10,
                "pagerank": 0.10,
                "structural_holes": 0.10,
                "gnn": 0.15,
                "multilayer": 0.20,  # Higher weight for multi-layer
                "temporal": 0.10,  # Temporal consistency
            }

            normalized_scores = {}
            for metric, scores in importance_scores.items():
                if scores.size(0) < num_nodes:
                    padded_scores = torch.zeros(num_nodes, device=edge_index.device)
                    padded_scores[: scores.size(0)] = scores
                    scores = padded_scores
                elif scores.size(0) > num_nodes:
                    scores = scores[:num_nodes]

                if scores.sum() > 0:
                    normalized_scores[metric] = (scores / scores.sum()).to(edge_index.device)
                else:
                    normalized_scores[metric] = scores.to(edge_index.device)

            total_importance = torch.zeros(num_nodes, device=edge_index.device)
            for k in normalized_scores.keys():
                weight = weights.get(k, 0.1)
                total_importance += weight * normalized_scores[k]

        critical_nodes = total_importance.topk(top_k).indices

        return critical_nodes, importance_scores

    def _to_networkx(self, edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
        """Convert to NetworkX graph"""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
        return G


# Backward compatibility
class CriticalNodeDetector(EnhancedCriticalNodeDetector):
    """Alias for backward compatibility"""

    pass


# ============================================================================
# NETWORK DISRUPTION OPTIMIZATION
# ============================================================================


class NetworkDisruptionOptimizer:
    """Find optimal disruption strategy"""

    def __init__(self, disruption_metric: str = "largest_component", algorithm: str = "greedy"):
        self.disruption_metric = disruption_metric
        self.algorithm = algorithm
        self.critic_detector = EnhancedCriticalNodeDetector()

    def compute_disruption_score(
        self, edge_index: torch.Tensor, num_nodes: int, removed_nodes: Set[int]
    ) -> float:
        """Measure network disruption"""
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in removed_nodes or dst in removed_nodes:
                mask[i] = False

        subgraph_edges = edge_index[:, mask]
        active_nodes = set(range(num_nodes)) - removed_nodes

        if subgraph_edges.size(1) == 0 or len(active_nodes) == 0:
            return 1.0

        G = nx.Graph()
        G.add_nodes_from(active_nodes)
        edges = subgraph_edges.t().cpu().numpy()
        G.add_edges_from(edges)

        if self.disruption_metric == "largest_component":
            if len(G) == 0:
                return 1.0
            largest_cc = max(nx.connected_components(G), key=len)
            return 1.0 - len(largest_cc) / num_nodes

        elif self.disruption_metric == "num_components":
            num_components = nx.number_connected_components(G)
            return num_components / num_nodes

        elif self.disruption_metric == "network_efficiency":
            efficiency = nx.global_efficiency(G)
            return 1.0 - efficiency

        return 0.0

    def greedy_disruption(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        budget_k: int,
        embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], List[float]]:
        """Greedy disruption algorithm"""
        removed_nodes = set()
        disruption_history = []

        for step in range(budget_k):
            best_node = None
            best_disruption = -1

            candidates = set(range(num_nodes)) - removed_nodes

            for node in candidates:
                test_removed = removed_nodes | {node}
                disruption = self.compute_disruption_score(edge_index, num_nodes, test_removed)

                if disruption > best_disruption:
                    best_disruption = disruption
                    best_node = node

            if best_node is not None:
                removed_nodes.add(best_node)
                disruption_history.append(best_disruption)

        return list(removed_nodes), disruption_history

    def optimize_disruption(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        budget_k: int,
        embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], List[float], Dict]:
        """Main optimization interface"""
        if self.algorithm == "greedy":
            removed, scores = self.greedy_disruption(edge_index, num_nodes, budget_k, embeddings)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        metadata = {
            "algorithm": self.algorithm,
            "disruption_metric": self.disruption_metric,
            "final_disruption": scores[-1] if scores else 0.0,
        }

        return removed, scores, metadata


# ============================================================================
# TEMPORAL RESILIENCE PREDICTION
# ============================================================================


class TemporalResiliencePredictor(nn.Module):
    """Predict network reconstruction after disruption"""

    def __init__(self, base_tgnn, hidden_dim: int, prediction_horizon: int = 5):
        super().__init__()
        self.base_tgnn = base_tgnn
        self.prediction_horizon = prediction_horizon

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.recruitment_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def predict_edge_formation(
        self, embeddings: torch.Tensor, node_pairs: torch.Tensor
    ) -> torch.Tensor:
        """Predict edge formation probability"""
        src_emb = embeddings[node_pairs[:, 0]]
        dst_emb = embeddings[node_pairs[:, 1]]

        pair_features = torch.cat([src_emb, dst_emb], dim=-1)
        edge_probs = self.edge_predictor(pair_features)

        return edge_probs.squeeze(-1)

    def predict_resilience(
        self,
        network: MultiLayerTemporalNetwork,
        removed_nodes: List[int],
        timestamps: torch.Tensor,
        num_future_steps: int = 5,
    ) -> Dict[str, Any]:
        """Predict network resilience"""
        if isinstance(self.base_tgnn, MultiLayerTemporalGNN):
            embeddings = self.base_tgnn(network, timestamps)
        else:
            layer_sequence = network.get_layer_sequence(0)
            node_features_seq = [layer.node_features for layer in layer_sequence]
            edge_indices_seq = [layer.edge_index for layer in layer_sequence]
            edge_features_seq = [layer.edge_features for layer in layer_sequence]
            embeddings = self.base_tgnn(
                node_features_seq, edge_indices_seq, edge_features_seq, timestamps
            )

        num_nodes = embeddings.size(0)
        removed_set = set(removed_nodes)
        active_nodes = [i for i in range(num_nodes) if i not in removed_set]
        active_embeddings = embeddings[active_nodes]

        num_active = len(active_nodes)
        if num_active > 1:
            node_pairs = []
            for i in range(num_active):
                for j in range(i + 1, num_active):
                    node_pairs.append([active_nodes[i], active_nodes[j]])
            node_pairs = torch.tensor(node_pairs, device=embeddings.device)

            edge_probs = self.predict_edge_formation(embeddings, node_pairs)
        else:
            node_pairs = torch.empty((0, 2), dtype=torch.long, device=embeddings.device)
            edge_probs = torch.empty(0, device=embeddings.device)

        recruitment_probs = self.recruitment_predictor(active_embeddings)

        resilience_score = self._compute_resilience(
            edge_probs, recruitment_probs, num_nodes, removed_set
        )

        return {
            "predicted_edges": node_pairs,
            "edge_probabilities": edge_probs,
            "recruitment_probabilities": recruitment_probs,
            "resilience_score": resilience_score,
        }

    def _compute_resilience(
        self,
        edge_probs: torch.Tensor,
        recruitment_probs: torch.Tensor,
        num_nodes: int,
        removed_nodes: Set[int],
    ) -> torch.Tensor:
        """Compute overall resilience score"""
        expected_new_edges = edge_probs.sum()
        expected_recruits = recruitment_probs.sum()
        removal_fraction = len(removed_nodes) / num_nodes

        resilience = (
            0.4 * (expected_new_edges / max(1, num_nodes))
            + 0.3 * (expected_recruits / max(1, len(removed_nodes)))
            + 0.3 * (1 - removal_fraction)
        )

        return resilience.clamp(0, 1)


# ============================================================================
# ADVERSARIAL ROBUSTNESS
# ============================================================================


class AdversarialNetworkAttack:
    """Simulate adversarial attacks and network adaptation"""

    def __init__(self):
        self.attack_history = []

    def simulate_adaptive_response(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        removed_nodes: List[int],
        adaptation_strategy: str = "decentralize",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate network adaptation to disruption"""
        removed_set = set(removed_nodes)

        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        for i in range(edge_index.size(1)):
            if edge_index[0, i].item() in removed_set or edge_index[1, i].item() in removed_set:
                mask[i] = False

        new_edge_index = edge_index[:, mask]

        if adaptation_strategy == "decentralize":
            new_edge_index = self._add_redundant_edges(new_edge_index, num_nodes, removed_set)
        elif adaptation_strategy == "recruit":
            new_edge_index, new_nodes = self._recruit_new_nodes(
                new_edge_index, num_nodes, len(removed_nodes)
            )
            num_nodes += new_nodes
        elif adaptation_strategy == "go_dark":
            new_edge_index = self._reduce_communication(new_edge_index)
        elif adaptation_strategy == "subdivide":
            new_edge_index = self._create_cells(new_edge_index, num_nodes, removed_set)

        return new_edge_index, torch.tensor(num_nodes)

    def _add_redundant_edges(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        removed_nodes: Set[int],
        num_new_edges: int = None,
    ) -> torch.Tensor:
        """Add redundant connections"""
        if num_new_edges is None:
            num_new_edges = len(removed_nodes) * 2

        active_nodes = [i for i in range(num_nodes) if i not in removed_nodes]

        if len(active_nodes) < 2:
            return edge_index

        new_edges = []
        attempts = 0
        while len(new_edges) < num_new_edges and attempts < num_new_edges * 10:
            i = np.random.choice(active_nodes)
            j = np.random.choice(active_nodes)
            if i != j and [i, j] not in new_edges and [j, i] not in new_edges:
                new_edges.append([i, j])
            attempts += 1

        if new_edges:
            new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
            edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)

        return edge_index

    def _recruit_new_nodes(
        self, edge_index: torch.Tensor, num_nodes: int, num_new_nodes: int
    ) -> Tuple[torch.Tensor, int]:
        """Recruit new members"""
        new_node_ids = range(num_nodes, num_nodes + num_new_nodes)
        existing_nodes = list(range(num_nodes))
        new_edges = []

        for new_id in new_node_ids:
            num_connections = np.random.randint(2, 4)
            connections = np.random.choice(
                existing_nodes, size=min(num_connections, len(existing_nodes)), replace=False
            )
            for conn in connections:
                new_edges.append([new_id, conn])

        if new_edges:
            new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
            edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)

        return edge_index, num_new_nodes

    def _reduce_communication(
        self, edge_index: torch.Tensor, reduction_rate: float = 0.3
    ) -> torch.Tensor:
        """Reduce communication"""
        num_edges = edge_index.size(1)
        num_keep = int(num_edges * (1 - reduction_rate))

        indices = torch.randperm(num_edges)[:num_keep]
        return edge_index[:, indices]

    def _create_cells(
        self, edge_index: torch.Tensor, num_nodes: int, removed_nodes: Set[int], cell_size: int = 5
    ) -> torch.Tensor:
        """Create disconnected cells"""
        active_nodes = [i for i in range(num_nodes) if i not in removed_nodes]

        np.random.shuffle(active_nodes)
        cells = [active_nodes[i : i + cell_size] for i in range(0, len(active_nodes), cell_size)]

        new_edges = []
        for cell in cells:
            for i in range(len(cell)):
                for j in range(i + 1, len(cell)):
                    new_edges.append([cell[i], cell[j]])

        if new_edges:
            return torch.tensor(new_edges, dtype=torch.long).t()
        return torch.empty((2, 0), dtype=torch.long)


# Test code
if __name__ == "__main__":
    print("=" * 80)
    print("Terrorist Network Disruption V2 - Enhanced")
    print("=" * 80)
    print("\nEnhancements:")
    print("  ✓ Multi-layer Centrality Analysis")
    print("  ✓ Temporal Importance Tracking")
    print("  ✓ Enhanced Critical Node Detection")
    print("\n✓ Enhanced disruption module ready!")
