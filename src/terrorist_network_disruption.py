"""
Terrorist Network Disruption using Temporal GNN
================================================

Specialized implementation for:
1. Critical Node Detection
2. Temporal Resilience Prediction
3. Adversarial Robustness
4. Multi-Layer Network Analysis
5. Optimal Disruption Strategies

Research Questions:
- Q1: Which nodes, when removed, most effectively disrupt the network?
- Q2: How will the network reconstruct after node removal?
- Q3: How does the network adapt to disruption attempts?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, to_dense_adj, remove_self_loops
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict
import networkx as nx
from scipy.stats import entropy
from dataclasses import dataclass

from .advanced_tgnn import AdvancedTemporalGNN


# ============================================================================
# MULTI-LAYER NETWORK SUPPORT
# ============================================================================

@dataclass
class NetworkLayer:
    """
    Represents one layer of a multi-layer terrorist network
    """
    name: str
    layer_type: str  # physical, digital, financial, ideological, operational
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_weights: torch.Tensor
    metadata: Dict


class MultiLayerTemporalNetwork:
    """
    Multi-layer temporal network for terrorist organizations
    
    Layers:
    1. Physical: Face-to-face meetings, physical locations
    2. Digital: Online communications, social media
    3. Financial: Money transfers, resource sharing
    4. Ideological: Shared beliefs, radicalization paths
    5. Operational: Joint operations, attack planning
    """
    def __init__(self, num_nodes: int, num_layers: int = 5):
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.layer_names = [
            'physical', 'digital', 'financial', 'ideological', 'operational'
        ]
        self.layers_history = []  # List of timesteps, each containing layers
        
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
        """
        Aggregate multiple layers into single network at timestep t
        
        Args:
            weights: Importance weight for each layer
        """
        if weights is None:
            weights = [1.0] * self.num_layers
        
        layers = self.get_timestep(t)
        
        # Aggregate edges with weights
        aggregated_edges = defaultdict(float)
        for layer, weight in zip(layers, weights):
            edge_index = layer.edge_index
            edge_weights = layer.edge_weights
            
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                edge_weight = edge_weights[i].item() if edge_weights is not None else 1.0
                aggregated_edges[(src, dst)] += weight * edge_weight
        
        # Convert back to tensor format
        edges = list(aggregated_edges.keys())
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_weights = torch.tensor([aggregated_edges[e] for e in edges])
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty(0)
        
        return edge_index, edge_weights


class MultiLayerTemporalGNN(nn.Module):
    """
    Temporal GNN that processes multi-layer networks
    """
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int,
        num_layers: int = 5,
        layer_fusion: str = 'attention',  # 'attention', 'concat', 'weighted_sum'
        **kwargs
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.layer_fusion = layer_fusion
        
        # Per-layer GNNs
        self.layer_gnns = nn.ModuleList([
            AdvancedTemporalGNN(
                num_node_features, num_edge_features, hidden_dim, **kwargs
            )
            for _ in range(num_layers)
        ])
        
        # Layer fusion mechanism
        if layer_fusion == 'attention':
            self.layer_attention = nn.MultiheadAttention(
                hidden_dim, num_heads=4, batch_first=True
            )
            self.layer_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        elif layer_fusion == 'weighted_sum':
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        multi_layer_network: MultiLayerTemporalNetwork,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        Process multi-layer temporal network
        
        Returns:
            embeddings: [num_nodes, hidden_dim]
        """
        num_timesteps = len(multi_layer_network.layers_history)
        
        # Process each layer independently
        layer_embeddings = []
        
        for layer_idx in range(self.num_layers):
            # Get temporal sequence for this layer
            layer_sequence = multi_layer_network.get_layer_sequence(layer_idx)
            
            # Prepare data for temporal GNN
            node_features_seq = [layer.node_features for layer in layer_sequence]
            edge_indices_seq = [layer.edge_index for layer in layer_sequence]
            edge_features_seq = [layer.edge_features for layer in layer_sequence]
            
            # Process with layer-specific GNN
            embeddings = self.layer_gnns[layer_idx](
                node_features_seq, edge_indices_seq, 
                edge_features_seq, timestamps
            )
            layer_embeddings.append(embeddings)
        
        # Fuse layer embeddings
        if self.layer_fusion == 'attention':
            # Stack: [num_nodes, num_layers, hidden_dim]
            layer_embeddings = torch.stack(layer_embeddings, dim=1)
            
            # Attention-based fusion
            batch_size = layer_embeddings.size(0)
            query = self.layer_query.expand(batch_size, -1, -1)
            
            fused, attn_weights = self.layer_attention(
                query, layer_embeddings, layer_embeddings
            )
            fused = fused.squeeze(1)
            
        elif self.layer_fusion == 'concat':
            fused = torch.cat(layer_embeddings, dim=-1)
            
        elif self.layer_fusion == 'weighted_sum':
            # Normalize weights
            weights = F.softmax(self.layer_weights, dim=0)
            fused = sum(w * emb for w, emb in zip(weights, layer_embeddings))
        
        return fused
    
    def reconstruct_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct node features from embeddings
        
        Args:
            embeddings: [num_nodes, hidden_dim]
        Returns:
            reconstructed: [num_nodes, num_node_features]
        """
        # Use the first layer's GNN reconstructor
        return self.layer_gnns[0].reconstruct_features(embeddings)


# ============================================================================
# CRITICAL NODE DETECTION
# ============================================================================

class CriticalNodeDetector:
    """
    Detect critical nodes in terrorist networks using multiple strategies
    """
    def __init__(self, importance_metrics: List[str] = None):
        if importance_metrics is None:
            self.importance_metrics = [
                'degree_centrality',
                'betweenness_centrality',
                'eigenvector_centrality',
                'pagerank',
                'structural_holes',
                'gnn_importance'
            ]
        else:
            self.importance_metrics = importance_metrics
    
    def compute_degree_centrality(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int
    ) -> torch.Tensor:
        """Degree centrality: number of connections"""
        deg = degree(edge_index[0], num_nodes=num_nodes)
        return deg / (num_nodes - 1)  # Normalize
    
    def compute_betweenness_centrality(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Betweenness centrality: number of shortest paths through node"""
        # Convert to NetworkX for computation
        G = self._to_networkx(edge_index, num_nodes)
        betweenness = nx.betweenness_centrality(G)
        # Return on same device as input
        device = edge_index.device
        return torch.tensor([betweenness[i] for i in range(num_nodes)], device=device)
    
    def compute_eigenvector_centrality(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        max_iter: int = 100
    ) -> torch.Tensor:
        """Eigenvector centrality: influence based on connections' importance"""
        G = self._to_networkx(edge_index, num_nodes)
        device = edge_index.device
        try:
            eigen_cent = nx.eigenvector_centrality(G, max_iter=max_iter)
            return torch.tensor([eigen_cent[i] for i in range(num_nodes)], device=device)
        except:
            # Fallback to degree centrality if not converged
            return self.compute_degree_centrality(edge_index, num_nodes)
    
    def compute_pagerank(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        alpha: float = 0.85
    ) -> torch.Tensor:
        """PageRank: importance based on random walk"""
        G = self._to_networkx(edge_index, num_nodes)
        pagerank = nx.pagerank(G, alpha=alpha)
        device = edge_index.device
        return torch.tensor([pagerank[i] for i in range(num_nodes)], device=device)
    
    def compute_structural_holes(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Structural holes: nodes that bridge disconnected communities
        High constraint = embedded in dense neighborhood
        Low constraint = bridging structural holes (more important)
        """
        G = self._to_networkx(edge_index, num_nodes)
        device = edge_index.device
        try:
            constraint = nx.constraint(G)
            # Invert: lower constraint = higher importance
            importance = torch.tensor([
                1.0 - constraint.get(i, 1.0) for i in range(num_nodes)
            ], device=device)
            return importance
        except:
            return torch.zeros(num_nodes, device=device)
    
    def compute_gnn_importance(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        GNN-based importance: learned from embeddings
        Uses embedding norm and influence on neighbors
        """
        # Embedding magnitude
        embedding_norm = embeddings.norm(dim=1)
        
        # Filter edges to only include nodes within embeddings range
        num_embedding_nodes = embeddings.size(0)
        valid_mask = (edge_index[0] < num_embedding_nodes) & (edge_index[1] < num_embedding_nodes)
        valid_edge_index = edge_index[:, valid_mask]
        
        # Influence on neighbors
        influence = torch.zeros(num_nodes, device=embeddings.device)
        for i in range(valid_edge_index.size(1)):
            src, dst = valid_edge_index[:, i]
            # Measure influence as similarity * neighbor's embedding norm
            similarity = F.cosine_similarity(
                embeddings[src].unsqueeze(0),
                embeddings[dst].unsqueeze(0)
            )
            influence[src] += similarity.squeeze() * embeddings[dst].norm()
        
        # Combine
        importance = embedding_norm + influence[:num_embedding_nodes]
        return importance / importance.sum()  # Normalize
    
    def detect_critical_nodes(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        embeddings: Optional[torch.Tensor] = None,
        top_k: int = 10,
        aggregation: str = 'weighted_sum'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Detect critical nodes using ensemble of metrics
        
        Returns:
            critical_nodes: Indices of top-k critical nodes
            importance_scores: Dict of scores for each metric
        """
        importance_scores = {}
        
        # Compute all metrics
        if 'degree_centrality' in self.importance_metrics:
            importance_scores['degree'] = self.compute_degree_centrality(
                edge_index, num_nodes
            )
        
        if 'betweenness_centrality' in self.importance_metrics:
            importance_scores['betweenness'] = self.compute_betweenness_centrality(
                edge_index, num_nodes
            )
        
        if 'eigenvector_centrality' in self.importance_metrics:
            importance_scores['eigenvector'] = self.compute_eigenvector_centrality(
                edge_index, num_nodes
            )
        
        if 'pagerank' in self.importance_metrics:
            importance_scores['pagerank'] = self.compute_pagerank(
                edge_index, num_nodes
            )
        
        if 'structural_holes' in self.importance_metrics:
            importance_scores['structural_holes'] = self.compute_structural_holes(
                edge_index, num_nodes
            )
        
        if 'gnn_importance' in self.importance_metrics and embeddings is not None:
            importance_scores['gnn'] = self.compute_gnn_importance(
                embeddings, edge_index, num_nodes
            )
        
        # Aggregate scores
        if aggregation == 'weighted_sum':
            # Equal weights for simplicity
            weights = {k: 1.0 for k in importance_scores.keys()}
            
            # Normalize each metric to [0, 1]
            normalized_scores = {}
            for metric, scores in importance_scores.items():
                # Ensure all scores have size num_nodes
                if scores.size(0) < num_nodes:
                    # Pad with zeros
                    padded_scores = torch.zeros(num_nodes, device=edge_index.device)
                    padded_scores[:scores.size(0)] = scores
                    scores = padded_scores
                elif scores.size(0) > num_nodes:
                    # Truncate
                    scores = scores[:num_nodes]
                
                # Normalize
                if scores.sum() > 0:
                    normalized_scores[metric] = (scores / scores.sum()).to(edge_index.device)
                else:
                    normalized_scores[metric] = scores.to(edge_index.device)
            
            # Weighted sum
            total_importance = torch.zeros(num_nodes, device=edge_index.device)
            for k in normalized_scores.keys():
                total_importance += weights[k] * normalized_scores[k]
        
        elif aggregation == 'rank_fusion':
            # Borda count: sum of ranks across metrics
            ranks = torch.zeros(num_nodes, device=edge_index.device)
            for scores in importance_scores.values():
                # Ensure scores are on correct device
                scores = scores.to(edge_index.device)
                ranks += scores.argsort().argsort().float()
            total_importance = ranks
        
        # Get top-k nodes
        critical_nodes = total_importance.topk(top_k).indices
        
        return critical_nodes, importance_scores
    
    def _to_networkx(self, edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
        """Convert to NetworkX graph"""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
        return G


# ============================================================================
# NETWORK DISRUPTION OPTIMIZATION
# ============================================================================

class NetworkDisruptionOptimizer:
    """
    Find optimal disruption strategy (NP-hard problem)
    
    Problem: Given network G and budget k, find k nodes to remove
    that maximizes network disruption
    
    Approaches:
    1. Greedy: Iteratively remove highest-impact node
    2. Beam Search: Keep top-b partial solutions
    3. Genetic Algorithm: Evolutionary optimization
    4. Reinforcement Learning: Learn removal policy
    """
    
    def __init__(
        self,
        disruption_metric: str = 'largest_component',
        algorithm: str = 'greedy'
    ):
        self.disruption_metric = disruption_metric
        self.algorithm = algorithm
        self.critic_detector = CriticalNodeDetector()
    
    def compute_disruption_score(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        removed_nodes: Set[int]
    ) -> float:
        """
        Measure how disrupted the network is after removing nodes
        
        Metrics:
        - largest_component: Size of largest connected component (minimize)
        - num_components: Number of connected components (maximize)
        - avg_shortest_path: Average shortest path length (maximize)
        - network_efficiency: Global efficiency (minimize)
        """
        # Create subgraph without removed nodes
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in removed_nodes or dst in removed_nodes:
                mask[i] = False
        
        subgraph_edges = edge_index[:, mask]
        active_nodes = set(range(num_nodes)) - removed_nodes
        
        if subgraph_edges.size(1) == 0 or len(active_nodes) == 0:
            # Network completely disrupted
            return 1.0
        
        # Convert to NetworkX
        G = nx.Graph()
        G.add_nodes_from(active_nodes)
        edges = subgraph_edges.t().cpu().numpy()
        G.add_edges_from(edges)
        
        # Compute metrics
        if self.disruption_metric == 'largest_component':
            if len(G) == 0:
                return 1.0
            largest_cc = max(nx.connected_components(G), key=len)
            # Disruption = 1 - (size of largest component / original size)
            return 1.0 - len(largest_cc) / num_nodes
        
        elif self.disruption_metric == 'num_components':
            num_components = nx.number_connected_components(G)
            # More components = more disruption
            return num_components / num_nodes
        
        elif self.disruption_metric == 'avg_shortest_path':
            try:
                # For disconnected graph, use average over components
                avg_path_length = nx.average_shortest_path_length(G)
                # Longer paths = more disruption (normalized)
                return min(1.0, avg_path_length / num_nodes)
            except:
                # Graph is disconnected, high disruption
                return 1.0
        
        elif self.disruption_metric == 'network_efficiency':
            efficiency = nx.global_efficiency(G)
            # Lower efficiency = more disruption
            return 1.0 - efficiency
        
        return 0.0
    
    def greedy_disruption(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        budget_k: int,
        embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Greedy algorithm: Iteratively remove node with highest impact
        
        Returns:
            removed_nodes: List of removed node indices
            disruption_scores: Disruption after each removal
        """
        removed_nodes = set()
        disruption_history = []
        
        for step in range(budget_k):
            best_node = None
            best_disruption = -1
            
            # Try removing each remaining node
            candidates = set(range(num_nodes)) - removed_nodes
            
            for node in candidates:
                # Simulate removal
                test_removed = removed_nodes | {node}
                disruption = self.compute_disruption_score(
                    edge_index, num_nodes, test_removed
                )
                
                if disruption > best_disruption:
                    best_disruption = disruption
                    best_node = node
            
            # Remove best node
            if best_node is not None:
                removed_nodes.add(best_node)
                disruption_history.append(best_disruption)
        
        return list(removed_nodes), disruption_history
    
    def beam_search_disruption(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        budget_k: int,
        beam_width: int = 5,
        embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Beam search: Keep top-b partial solutions at each step
        
        More thorough than greedy but still tractable
        """
        # Initialize beam with empty solution
        beam = [(set(), 0.0)]  # (removed_nodes, disruption_score)
        
        for step in range(budget_k):
            candidates = []
            
            # Expand each solution in beam
            for removed_nodes, _ in beam:
                # Try adding each node
                remaining = set(range(num_nodes)) - removed_nodes
                
                for node in remaining:
                    new_removed = removed_nodes | {node}
                    disruption = self.compute_disruption_score(
                        edge_index, num_nodes, new_removed
                    )
                    candidates.append((new_removed, disruption))
            
            # Keep top-beam_width solutions
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]
        
        # Return best solution
        best_solution = max(beam, key=lambda x: x[1])
        removed_nodes = list(best_solution[0])
        
        # Compute disruption history
        disruption_history = []
        for i in range(1, len(removed_nodes) + 1):
            disruption = self.compute_disruption_score(
                edge_index, num_nodes, set(removed_nodes[:i])
            )
            disruption_history.append(disruption)
        
        return removed_nodes, disruption_history
    
    def optimize_disruption(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        budget_k: int,
        embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[List[int], List[float], Dict]:
        """
        Main interface: Find optimal disruption strategy
        
        Returns:
            removed_nodes: Nodes to remove
            disruption_scores: Disruption after each removal
            metadata: Additional information
        """
        if self.algorithm == 'greedy':
            removed, scores = self.greedy_disruption(
                edge_index, num_nodes, budget_k, embeddings
            )
        elif self.algorithm == 'beam_search':
            removed, scores = self.beam_search_disruption(
                edge_index, num_nodes, budget_k, 
                beam_width=5, embeddings=embeddings
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        metadata = {
            'algorithm': self.algorithm,
            'disruption_metric': self.disruption_metric,
            'final_disruption': scores[-1] if scores else 0.0
        }
        
        return removed, scores, metadata


# ============================================================================
# TEMPORAL RESILIENCE PREDICTION
# ============================================================================

class TemporalResiliencePredictor(nn.Module):
    """
    Predict how network will reconstruct after node removal
    
    Q2: How will the network adapt and rebuild connections?
    """
    def __init__(
        self,
        base_tgnn: AdvancedTemporalGNN,
        hidden_dim: int,
        prediction_horizon: int = 5
    ):
        super().__init__()
        self.base_tgnn = base_tgnn
        self.prediction_horizon = prediction_horizon
        
        # Edge reconstruction predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # New node recruitment predictor
        self.recruitment_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def predict_edge_formation(
        self,
        embeddings: torch.Tensor,
        node_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict probability of edge formation between node pairs
        
        Args:
            embeddings: [num_nodes, hidden_dim]
            node_pairs: [num_pairs, 2] - pairs of nodes
        """
        src_emb = embeddings[node_pairs[:, 0]]
        dst_emb = embeddings[node_pairs[:, 1]]
        
        pair_features = torch.cat([src_emb, dst_emb], dim=-1)
        edge_probs = self.edge_predictor(pair_features)
        
        return edge_probs.squeeze(-1)
    
    def predict_reconstruction(
        self,
        node_features_seq: List[torch.Tensor],
        edge_indices_seq: List[torch.Tensor],
        edge_features_seq: List[torch.Tensor],
        timestamps: torch.Tensor,
        removed_nodes: Set[int],
        num_future_steps: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict network reconstruction after node removal
        
        Returns:
            predictions: Dict containing:
                - new_edges: Predicted new edge probabilities
                - recruitment_prob: Probability of recruiting new members
                - resilience_score: Overall network resilience
        """
        if num_future_steps is None:
            num_future_steps = self.prediction_horizon
        
        # Get embeddings from history
        # Check if base_tgnn is MultiLayerTemporalGNN or AdvancedTemporalGNN
        if isinstance(self.base_tgnn, MultiLayerTemporalGNN):
            # For MultiLayerTemporalGNN, we need to create a temporary network
            # This is a simplified approach - just use the model directly if available
            # In practice, this path shouldn't be called directly
            raise NotImplementedError(
                "predict_reconstruction should be called via predict_resilience for MultiLayerTemporalGNN"
            )
        else:
            # For AdvancedTemporalGNN
            embeddings = self.base_tgnn(
                node_features_seq, edge_indices_seq,
                edge_features_seq, timestamps
            )
        
        # Active nodes (not removed)
        num_nodes = embeddings.size(0)
        active_nodes = [i for i in range(num_nodes) if i not in removed_nodes]
        active_embeddings = embeddings[active_nodes]
        
        # Predict new edge formation among active nodes
        num_active = len(active_nodes)
        if num_active > 1:
            # Generate all possible pairs
            node_pairs = []
            for i in range(num_active):
                for j in range(i + 1, num_active):
                    node_pairs.append([active_nodes[i], active_nodes[j]])
            node_pairs = torch.tensor(node_pairs, device=embeddings.device)
            
            # Predict edge probabilities
            edge_probs = self.predict_edge_formation(embeddings, node_pairs)
        else:
            node_pairs = torch.empty((0, 2), dtype=torch.long)
            edge_probs = torch.empty(0)
        
        # Predict recruitment probability
        recruitment_probs = self.recruitment_predictor(active_embeddings)
        
        # Compute resilience score
        resilience_score = self._compute_resilience(
            edge_probs, recruitment_probs, num_nodes, removed_nodes
        )
        
        return {
            'predicted_edges': node_pairs,
            'edge_probabilities': edge_probs,
            'recruitment_probabilities': recruitment_probs,
            'resilience_score': resilience_score
        }
    
    def _compute_resilience(
        self,
        edge_probs: torch.Tensor,
        recruitment_probs: torch.Tensor,
        num_nodes: int,
        removed_nodes: Set[int]
    ) -> torch.Tensor:
        """
        Compute overall resilience score
        
        High resilience = network can easily reconstruct
        Low resilience = network is significantly weakened
        """
        # Factors:
        # 1. Expected number of new edges
        expected_new_edges = edge_probs.sum()
        
        # 2. Expected new recruits
        expected_recruits = recruitment_probs.sum()
        
        # 3. Fraction of network removed
        removal_fraction = len(removed_nodes) / num_nodes
        
        # Combine into resilience score (0-1, higher = more resilient)
        resilience = (
            0.4 * (expected_new_edges / max(1, num_nodes)) +
            0.3 * (expected_recruits / max(1, len(removed_nodes))) +
            0.3 * (1 - removal_fraction)
        )
        
        return resilience.clamp(0, 1)
    
    def predict_resilience(
        self,
        network: 'MultiLayerTemporalNetwork',
        removed_nodes: List[int],
        timestamps: torch.Tensor,
        num_future_steps: int = 5
    ) -> Dict[str, Any]:
        """
        High-level interface for resilience prediction
        
        Args:
            network: MultiLayerTemporalNetwork
            removed_nodes: List of node IDs to remove
            timestamps: Temporal timestamps
            num_future_steps: Number of future timesteps to predict
            
        Returns:
            Dict with resilience metrics
        """
        # Get embeddings directly from base_tgnn
        if isinstance(self.base_tgnn, MultiLayerTemporalGNN):
            embeddings = self.base_tgnn(network, timestamps)
        else:
            # For AdvancedTemporalGNN, use layer sequence
            layer_sequence = network.get_layer_sequence(0)
            node_features_seq = [layer.node_features for layer in layer_sequence]
            edge_indices_seq = [layer.edge_index for layer in layer_sequence]
            edge_features_seq = [layer.edge_features for layer in layer_sequence]
            embeddings = self.base_tgnn(
                node_features_seq, edge_indices_seq,
                edge_features_seq, timestamps
            )
        
        # Active nodes (not removed)
        num_nodes = embeddings.size(0)
        removed_set = set(removed_nodes)
        active_nodes = [i for i in range(num_nodes) if i not in removed_set]
        active_embeddings = embeddings[active_nodes]
        
        # Predict new edge formation among active nodes
        num_active = len(active_nodes)
        if num_active > 1:
            # Generate all possible pairs
            node_pairs = []
            for i in range(num_active):
                for j in range(i + 1, num_active):
                    node_pairs.append([active_nodes[i], active_nodes[j]])
            node_pairs = torch.tensor(node_pairs, device=embeddings.device)
            
            # Predict edge probabilities
            edge_probs = self.predict_edge_formation(embeddings, node_pairs)
        else:
            node_pairs = torch.empty((0, 2), dtype=torch.long, device=embeddings.device)
            edge_probs = torch.empty(0, device=embeddings.device)
        
        # Predict recruitment probability
        recruitment_probs = self.recruitment_predictor(active_embeddings)
        
        # Compute resilience score
        resilience_score = self._compute_resilience(
            edge_probs, recruitment_probs, num_nodes, removed_set
        )
        
        return {
            'predicted_edges': node_pairs,
            'edge_probabilities': edge_probs,
            'recruitment_probabilities': recruitment_probs,
            'resilience_score': resilience_score
        }


# ============================================================================
# ADVERSARIAL ROBUSTNESS
# ============================================================================

class AdversarialNetworkAttack:
    """
    Simulate adversarial attacks and network adaptation
    
    Q3: How does the network adapt to disruption attempts?
    
    Strategies:
    1. Proactive: Network anticipates and adapts before attack
    2. Reactive: Network responds after attack
    3. Deceptive: Network creates fake nodes to mislead
    """
    
    def __init__(self):
        self.attack_history = []
    
    def simulate_adaptive_response(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        removed_nodes: List[int],
        adaptation_strategy: str = 'decentralize'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate network's adaptive response to node removal
        
        Strategies:
        - decentralize: Create more redundant connections
        - recruit: Add new nodes to replace removed ones
        - go_dark: Reduce communication, harder to detect
        - subdivide: Split into smaller cells
        """
        removed_set = set(removed_nodes)
        
        # Filter edges
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        for i in range(edge_index.size(1)):
            if edge_index[0, i].item() in removed_set or \
               edge_index[1, i].item() in removed_set:
                mask[i] = False
        
        new_edge_index = edge_index[:, mask]
        
        if adaptation_strategy == 'decentralize':
            # Add redundant connections between remaining nodes
            new_edges = self._add_redundant_edges(
                new_edge_index, num_nodes, removed_set
            )
            
        elif adaptation_strategy == 'recruit':
            # Add new nodes
            new_edge_index, new_nodes = self._recruit_new_nodes(
                new_edge_index, num_nodes, len(removed_nodes)
            )
            num_nodes += new_nodes
            
        elif adaptation_strategy == 'go_dark':
            # Remove some edges to reduce detectability
            new_edge_index = self._reduce_communication(new_edge_index)
            
        elif adaptation_strategy == 'subdivide':
            # Create disconnected cells
            new_edge_index = self._create_cells(
                new_edge_index, num_nodes, removed_set
            )
        
        return new_edge_index, torch.tensor(num_nodes)
    
    def _add_redundant_edges(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        removed_nodes: Set[int],
        num_new_edges: int = None
    ) -> torch.Tensor:
        """Add redundant connections to increase robustness"""
        if num_new_edges is None:
            num_new_edges = len(removed_nodes) * 2
        
        active_nodes = [i for i in range(num_nodes) if i not in removed_nodes]
        
        if len(active_nodes) < 2:
            return edge_index
        
        # Randomly add edges between active nodes
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
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_new_nodes: int
    ) -> Tuple[torch.Tensor, int]:
        """Recruit new members to replace removed ones"""
        new_node_ids = range(num_nodes, num_nodes + num_new_nodes)
        
        # Connect new nodes to existing network
        existing_nodes = list(range(num_nodes))
        new_edges = []
        
        for new_id in new_node_ids:
            # Connect to 2-3 existing nodes
            num_connections = np.random.randint(2, 4)
            connections = np.random.choice(
                existing_nodes, size=min(num_connections, len(existing_nodes)),
                replace=False
            )
            for conn in connections:
                new_edges.append([new_id, conn])
        
        if new_edges:
            new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
            edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)
        
        return edge_index, num_new_nodes
    
    def _reduce_communication(
        self,
        edge_index: torch.Tensor,
        reduction_rate: float = 0.3
    ) -> torch.Tensor:
        """Reduce communication to avoid detection"""
        num_edges = edge_index.size(1)
        num_keep = int(num_edges * (1 - reduction_rate))
        
        # Randomly keep edges
        indices = torch.randperm(num_edges)[:num_keep]
        return edge_index[:, indices]
    
    def _create_cells(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        removed_nodes: Set[int],
        cell_size: int = 5
    ) -> torch.Tensor:
        """Create disconnected cells"""
        active_nodes = [i for i in range(num_nodes) if i not in removed_nodes]
        
        # Divide into cells
        np.random.shuffle(active_nodes)
        cells = [active_nodes[i:i+cell_size] 
                for i in range(0, len(active_nodes), cell_size)]
        
        # Create edges only within cells
        new_edges = []
        for cell in cells:
            for i in range(len(cell)):
                for j in range(i + 1, len(cell)):
                    new_edges.append([cell[i], cell[j]])
        
        if new_edges:
            return torch.tensor(new_edges, dtype=torch.long).t()
        return torch.empty((2, 0), dtype=torch.long)


# Test and demonstration code
if __name__ == "__main__":
    print("="*80)
    print("Terrorist Network Disruption - Specialized Module")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample network
    num_nodes = 50
    num_edges = 150
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    embeddings = torch.randn(num_nodes, 128, device=device)
    
    print("\n1. Critical Node Detection")
    print("-" * 80)
    detector = CriticalNodeDetector()
    critical_nodes, scores = detector.detect_critical_nodes(
        edge_index, num_nodes, embeddings, top_k=10
    )
    print(f"Top 10 critical nodes: {critical_nodes.tolist()}")
    print(f"Metrics computed: {list(scores.keys())}")
    
    print("\n2. Network Disruption Optimization")
    print("-" * 80)
    optimizer = NetworkDisruptionOptimizer(algorithm='greedy')
    removed, disruption_scores, metadata = optimizer.optimize_disruption(
        edge_index, num_nodes, budget_k=5
    )
    print(f"Optimal removal sequence: {removed}")
    print(f"Disruption scores: {[f'{s:.3f}' for s in disruption_scores]}")
    print(f"Final disruption: {metadata['final_disruption']:.3f}")
    
    print("\n3. Adversarial Network Adaptation")
    print("-" * 80)
    adversarial = AdversarialNetworkAttack()
    adapted_edges, new_num_nodes = adversarial.simulate_adaptive_response(
        edge_index, num_nodes, removed, adaptation_strategy='decentralize'
    )
    print(f"Original edges: {edge_index.size(1)}")
    print(f"Adapted edges: {adapted_edges.size(1)}")
    print(f"Adaptation: Network added {adapted_edges.size(1) - edge_index.size(1) + len(removed)*2} redundant connections")
    
    print("\n" + "="*80)
    print("Specialized Module Ready!")
    print("="*80)
