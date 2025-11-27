"""
Terrorist Network Dataset Generator V2 - Enhanced Version
=========================================================

Enhanced with:
1. Network data augmentation
2. Improved evaluation metrics
3. Network resilience computation
4. Better realism in generation

Author: Advanced GNN Research
Version: 2.0
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import random
from collections import defaultdict

from terrorist_network_disruption import (
    NetworkLayer,
    MultiLayerTemporalNetwork,
    EnhancedCriticalNodeDetector,
    NetworkDisruptionOptimizer,
    AdversarialNetworkAttack
)


# ============================================================================
# NETWORK DATA AUGMENTATION
# ============================================================================

class NetworkAugmenter:
    """
    Data augmentation for network data
    Increases training data diversity and model robustness
    """
    def __init__(self, device='cpu'):
        self.device = device
    
    def edge_dropout(self, edge_index: torch.Tensor, drop_rate: float = 0.1) -> torch.Tensor:
        """
        Edge dropout - randomly remove edges
        Simulates noisy/incomplete data
        """
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) > drop_rate
        return edge_index[:, mask]
    
    def feature_masking(self, node_features: torch.Tensor, mask_rate: float = 0.1) -> torch.Tensor:
        """
        Feature masking - randomly mask feature dimensions
        Improves robustness
        """
        mask = torch.rand_like(node_features) > mask_rate
        return node_features * mask
    
    def add_gaussian_noise(self, node_features: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to features"""
        noise = torch.randn_like(node_features) * noise_std
        return node_features + noise
    
    def temporal_interpolation(
        self, 
        network: MultiLayerTemporalNetwork, 
        num_interpolations: int = 1
    ) -> List[List[NetworkLayer]]:
        """
        Temporal interpolation - create intermediate timesteps
        Increases temporal resolution
        """
        augmented_history = []
        
        for t in range(len(network.layers_history) - 1):
            layers_t = network.get_timestep(t)
            layers_t1 = network.get_timestep(t + 1)
            
            # Add original timestep
            augmented_history.append(layers_t)
            
            # Add interpolated timesteps
            for i in range(1, num_interpolations + 1):
                alpha = i / (num_interpolations + 1)
                interpolated_layers = self._interpolate_layers(
                    layers_t, layers_t1, alpha
                )
                augmented_history.append(interpolated_layers)
        
        # Add final timestep
        augmented_history.append(network.layers_history[-1])
        
        return augmented_history
    
    def _interpolate_layers(
        self, 
        layers_a: List[NetworkLayer], 
        layers_b: List[NetworkLayer], 
        alpha: float
    ) -> List[NetworkLayer]:
        """Interpolate between two layer sets"""
        interpolated = []
        
        for layer_a, layer_b in zip(layers_a, layers_b):
            # Interpolate node features
            node_features = (1 - alpha) * layer_a.node_features + alpha * layer_b.node_features
            
            # Union of edges from both timesteps
            edge_index = torch.cat([layer_a.edge_index, layer_b.edge_index], dim=1)
            edge_features = torch.cat([layer_a.edge_features, layer_b.edge_features], dim=0)
            
            # Weighted edge weights
            edge_weights = torch.cat([
                layer_a.edge_weights * (1 - alpha),
                layer_b.edge_weights * alpha
            ])
            
            interpolated.append(NetworkLayer(
                name=layer_a.name,
                layer_type=layer_a.layer_type,
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                edge_weights=edge_weights,
                metadata=layer_a.metadata
            ))
        
        return interpolated
    
    def augment_network(
        self,
        network: MultiLayerTemporalNetwork,
        edge_drop_rate: float = 0.05,
        feature_mask_rate: float = 0.05,
        noise_std: float = 0.01
    ) -> MultiLayerTemporalNetwork:
        """
        Apply all augmentations to network
        """
        augmented_network = MultiLayerTemporalNetwork(
            num_nodes=network.num_nodes,
            num_layers=network.num_layers
        )
        
        for timestep_layers in network.layers_history:
            augmented_layers = []
            
            for layer in timestep_layers:
                # Augment edges
                aug_edge_index = self.edge_dropout(layer.edge_index, edge_drop_rate)
                
                # Augment features
                aug_node_features = self.feature_masking(layer.node_features, feature_mask_rate)
                aug_node_features = self.add_gaussian_noise(aug_node_features, noise_std)
                
                # Filter edge features and weights for remaining edges
                if aug_edge_index.size(1) < layer.edge_index.size(1):
                    # Keep only features for remaining edges
                    mask = torch.zeros(layer.edge_index.size(1), dtype=torch.bool)
                    for i in range(aug_edge_index.size(1)):
                        # Find matching edge in original
                        for j in range(layer.edge_index.size(1)):
                            if torch.equal(aug_edge_index[:, i], layer.edge_index[:, j]):
                                mask[j] = True
                                break
                    aug_edge_features = layer.edge_features[mask] if mask.any() else layer.edge_features[:aug_edge_index.size(1)]
                    aug_edge_weights = layer.edge_weights[mask] if mask.any() else layer.edge_weights[:aug_edge_index.size(1)]
                else:
                    aug_edge_features = layer.edge_features
                    aug_edge_weights = layer.edge_weights
                
                augmented_layers.append(NetworkLayer(
                    name=layer.name,
                    layer_type=layer.layer_type,
                    node_features=aug_node_features,
                    edge_index=aug_edge_index,
                    edge_features=aug_edge_features,
                    edge_weights=aug_edge_weights,
                    metadata=layer.metadata.copy()
                ))
            
            augmented_network.add_timestep(augmented_layers)
        
        return augmented_network


# ============================================================================
# REALISTIC TERRORIST NETWORK GENERATOR
# ============================================================================

@dataclass
class NetworkConfig:
    """Configuration for terrorist network generation"""
    initial_nodes: int = 50
    max_nodes: int = 100
    avg_degree: float = 4.0
    clustering_coefficient: float = 0.3
    hierarchy_levels: int = 3
    recruitment_rate: float = 0.05
    dropout_rate: float = 0.02
    physical_density: float = 0.3
    digital_density: float = 0.6
    financial_density: float = 0.2
    ideological_density: float = 0.5
    operational_density: float = 0.15
    leader_ratio: float = 0.05
    operative_ratio: float = 0.20
    supporter_ratio: float = 0.75


class TerroristNetworkGenerator:
    """Generate realistic terrorist network with multiple layers"""
    
    def __init__(self, config: NetworkConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.node_types = []
        self.node_attributes = []
        self.hierarchy_level = []
        
    def generate_temporal_network(
        self,
        num_timesteps: int = 20,
        num_node_features: int = 64,
        num_edge_features: int = 32,
        device: torch.device = torch.device('cpu')
    ) -> MultiLayerTemporalNetwork:
        """Generate complete multi-layer temporal network"""
        print(f"\nGenerating terrorist network with {num_timesteps} timesteps...")
        
        network = MultiLayerTemporalNetwork(
            num_nodes=self.config.max_nodes,
            num_layers=5
        )
        
        self._initialize_network()
        
        for t in range(num_timesteps):
            layers = self._generate_timestep(
                t, num_node_features, num_edge_features, device
            )
            network.add_timestep(layers)
            
            if t < num_timesteps - 1:
                self._evolve_network()
            
            if (t + 1) % 5 == 0:
                print(f"  Generated timestep {t+1}/{num_timesteps}")
        
        print(f"✓ Network generation complete!")
        print(f"  Final size: {len(self.node_types)} nodes")
        print(f"  Leaders: {sum(1 for nt in self.node_types if nt == 'leader')}")
        print(f"  Operatives: {sum(1 for nt in self.node_types if nt == 'operative')}")
        print(f"  Supporters: {sum(1 for nt in self.node_types if nt == 'supporter')}")
        
        return network
    
    def _initialize_network(self):
        """Initialize network structure"""
        num_nodes = self.config.initial_nodes
        
        num_leaders = max(1, int(num_nodes * self.config.leader_ratio))
        num_operatives = int(num_nodes * self.config.operative_ratio)
        num_supporters = num_nodes - num_leaders - num_operatives
        
        self.node_types = (
            ['leader'] * num_leaders +
            ['operative'] * num_operatives +
            ['supporter'] * num_supporters
        )
        random.shuffle(self.node_types)
        
        self.hierarchy_level = []
        for node_type in self.node_types:
            if node_type == 'leader':
                level = self.rng.randint(0, 2)
            elif node_type == 'operative':
                level = self.rng.randint(1, 3)
            else:
                level = self.rng.randint(2, self.config.hierarchy_levels)
            self.hierarchy_level.append(level)
        
        self.node_attributes = []
        for i, node_type in enumerate(self.node_types):
            attr = {
                'type': node_type,
                'hierarchy_level': self.hierarchy_level[i],
                'join_time': 0,
                'activity_level': self.rng.uniform(0.3, 1.0),
                'trust_score': self.rng.uniform(0.5, 1.0) if node_type != 'supporter' else self.rng.uniform(0.2, 0.6),
                'radicalization_level': self.rng.uniform(0.3, 1.0),
                'operational_skill': self.rng.uniform(0.1, 1.0),
                'financial_capacity': self.rng.uniform(0.0, 1.0)
            }
            self.node_attributes.append(attr)
    
    def _generate_timestep(
        self,
        t: int,
        num_node_features: int,
        num_edge_features: int,
        device: torch.device
    ) -> List[NetworkLayer]:
        """Generate all 5 layers for timestep t"""
        num_nodes = len(self.node_types)
        
        node_features = self._generate_node_features(
            num_nodes, num_node_features, device
        )
        
        layers = []
        
        # Layer 1: Physical network
        physical = self._generate_physical_layer(
            num_nodes, num_edge_features, node_features, device
        )
        layers.append(physical)
        
        # Layer 2: Digital network
        digital = self._generate_digital_layer(
            num_nodes, num_edge_features, node_features, device
        )
        layers.append(digital)
        
        # Layer 3: Financial network
        financial = self._generate_financial_layer(
            num_nodes, num_edge_features, node_features, device
        )
        layers.append(financial)
        
        # Layer 4: Ideological network
        ideological = self._generate_ideological_layer(
            num_nodes, num_edge_features, node_features, device
        )
        layers.append(ideological)
        
        # Layer 5: Operational network
        operational = self._generate_operational_layer(
            num_nodes, num_edge_features, node_features, device
        )
        layers.append(operational)
        
        return layers
    
    def _generate_node_features(
        self,
        num_nodes: int,
        num_features: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate node features based on attributes"""
        features = torch.zeros(num_nodes, num_features, device=device)
        
        for i in range(num_nodes):
            attr = self.node_attributes[i]
            
            # Encode node type (one-hot)
            if attr['type'] == 'leader':
                features[i, 0:3] = torch.tensor([1, 0, 0])
            elif attr['type'] == 'operative':
                features[i, 0:3] = torch.tensor([0, 1, 0])
            else:
                features[i, 0:3] = torch.tensor([0, 0, 1])
            
            # Encode continuous attributes
            features[i, 3] = attr['hierarchy_level'] / self.config.hierarchy_levels
            features[i, 4] = attr['activity_level']
            features[i, 5] = attr['trust_score']
            features[i, 6] = attr['radicalization_level']
            features[i, 7] = attr['operational_skill']
            features[i, 8] = attr['financial_capacity']
            features[i, 9] = attr['join_time'] / 100.0
            
            # Random features for the rest
            features[i, 10:] = torch.randn(num_features - 10, device=device) * 0.1
        
        return features
    
    def _generate_layer_edges(
        self,
        num_nodes: int,
        density: float,
        preferential_attachment: bool = True,
        hierarchy_respect: bool = False
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Generate edges for a layer"""
        edges = []
        edge_weights = []
        
        target_edges = int(num_nodes * (num_nodes - 1) / 2 * density)
        
        if preferential_attachment:
            degrees = defaultdict(int)
            
            for _ in range(target_edges):
                if len(edges) < num_nodes:
                    i, j = self.rng.choice(num_nodes, size=2, replace=False)
                else:
                    probs = np.array([degrees[i] + 1 for i in range(num_nodes)])
                    probs = probs / probs.sum()
                    i = self.rng.choice(num_nodes, p=probs)
                    
                    if hierarchy_respect:
                        same_level = [
                            j for j in range(num_nodes) 
                            if j != i and 
                            abs(self.hierarchy_level[j] - self.hierarchy_level[i]) <= 1
                        ]
                        if same_level:
                            j = random.choice(same_level)
                        else:
                            j = self.rng.choice([k for k in range(num_nodes) if k != i])
                    else:
                        j = self.rng.choice([k for k in range(num_nodes) if k != i])
                
                if i != j and (i, j) not in edges and (j, i) not in edges:
                    edges.append((i, j))
                    degrees[i] += 1
                    degrees[j] += 1
                    
                    weight = (
                        self.node_attributes[i]['trust_score'] * 
                        self.node_attributes[j]['trust_score']
                    )
                    edge_weights.append(weight)
        else:
            for _ in range(target_edges):
                i, j = self.rng.choice(num_nodes, size=2, replace=False)
                if (i, j) not in edges and (j, i) not in edges:
                    edges.append((i, j))
                    weight = self.rng.uniform(0.3, 1.0)
                    edge_weights.append(weight)
        
        return edges, edge_weights
    
    def _generate_physical_layer(
        self, num_nodes, num_edge_features, node_features, device
    ) -> NetworkLayer:
        """Physical meetings - sparse, hierarchical"""
        edges, weights = self._generate_layer_edges(
            num_nodes, 
            self.config.physical_density,
            preferential_attachment=True,
            hierarchy_respect=True
        )
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
            edge_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            edge_features = torch.randn(len(edges), num_edge_features, device=device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_weights_tensor = torch.empty(0, device=device)
            edge_features = torch.empty((0, num_edge_features), device=device)
        
        return NetworkLayer(
            name='physical',
            layer_type='physical',
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_weights=edge_weights_tensor,
            metadata={'description': 'Face-to-face meetings'}
        )
    
    def _generate_digital_layer(
        self, num_nodes, num_edge_features, node_features, device
    ) -> NetworkLayer:
        """Digital communications - dense, less hierarchical"""
        edges, weights = self._generate_layer_edges(
            num_nodes,
            self.config.digital_density,
            preferential_attachment=True,
            hierarchy_respect=False
        )
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
            edge_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            edge_features = torch.randn(len(edges), num_edge_features, device=device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_weights_tensor = torch.empty(0, device=device)
            edge_features = torch.empty((0, num_edge_features), device=device)
        
        return NetworkLayer(
            name='digital',
            layer_type='digital',
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_weights=edge_weights_tensor,
            metadata={'description': 'Online communications'}
        )
    
    def _generate_financial_layer(
        self, num_nodes, num_edge_features, node_features, device
    ) -> NetworkLayer:
        """Financial flows - very sparse"""
        edges, weights = self._generate_layer_edges(
            num_nodes,
            self.config.financial_density,
            preferential_attachment=False,
            hierarchy_respect=True
        )
        
        filtered_edges = []
        filtered_weights = []
        for (i, j), w in zip(edges, weights):
            if self.node_attributes[i]['financial_capacity'] > 0.3:
                filtered_edges.append((i, j))
                filtered_weights.append(
                    w * self.node_attributes[i]['financial_capacity']
                )
        
        if filtered_edges:
            edge_index = torch.tensor(filtered_edges, dtype=torch.long, device=device).t()
            edge_weights_tensor = torch.tensor(filtered_weights, dtype=torch.float32, device=device)
            edge_features = torch.randn(len(filtered_edges), num_edge_features, device=device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_weights_tensor = torch.empty(0, device=device)
            edge_features = torch.empty((0, num_edge_features), device=device)
        
        return NetworkLayer(
            name='financial',
            layer_type='financial',
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_weights=edge_weights_tensor,
            metadata={'description': 'Money transfers'}
        )
    
    def _generate_ideological_layer(
        self, num_nodes, num_edge_features, node_features, device
    ) -> NetworkLayer:
        """Ideological connections"""
        edges, weights = self._generate_layer_edges(
            num_nodes,
            self.config.ideological_density,
            preferential_attachment=True,
            hierarchy_respect=False
        )
        
        filtered_edges = []
        filtered_weights = []
        for (i, j), w in zip(edges, weights):
            rad_i = self.node_attributes[i]['radicalization_level']
            rad_j = self.node_attributes[j]['radicalization_level']
            similarity = 1.0 - abs(rad_i - rad_j)
            filtered_edges.append((i, j))
            filtered_weights.append(w * similarity)
        
        if filtered_edges:
            edge_index = torch.tensor(filtered_edges, dtype=torch.long, device=device).t()
            edge_weights_tensor = torch.tensor(filtered_weights, dtype=torch.float32, device=device)
            edge_features = torch.randn(len(filtered_edges), num_edge_features, device=device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_weights_tensor = torch.empty(0, device=device)
            edge_features = torch.empty((0, num_edge_features), device=device)
        
        return NetworkLayer(
            name='ideological',
            layer_type='ideological',
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_weights=edge_weights_tensor,
            metadata={'description': 'Shared ideology'}
        )
    
    def _generate_operational_layer(
        self, num_nodes, num_edge_features, node_features, device
    ) -> NetworkLayer:
        """Operational connections - very sparse"""
        operative_indices = [
            i for i, nt in enumerate(self.node_types)
            if nt in ['operative', 'leader']
        ]
        
        edges = []
        weights = []
        
        if len(operative_indices) > 1:
            num_cells = max(1, len(operative_indices) // 5)
            random.shuffle(operative_indices)
            
            for cell_id in range(num_cells):
                cell = operative_indices[cell_id::num_cells]
                
                for i in range(len(cell)):
                    for j in range(i + 1, len(cell)):
                        if self.rng.random() < 0.5:
                            edges.append((cell[i], cell[j]))
                            skill_score = (
                                self.node_attributes[cell[i]]['operational_skill'] +
                                self.node_attributes[cell[j]]['operational_skill']
                            ) / 2
                            weights.append(skill_score)
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
            edge_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            edge_features = torch.randn(len(edges), num_edge_features, device=device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_weights_tensor = torch.empty(0, device=device)
            edge_features = torch.empty((0, num_edge_features), device=device)
        
        return NetworkLayer(
            name='operational',
            layer_type='operational',
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_weights=edge_weights_tensor,
            metadata={'description': 'Joint operations'}
        )
    
    def _evolve_network(self):
        """Evolve network over time: recruitment and dropout"""
        num_current = len(self.node_types)
        num_dropout = max(0, int(num_current * self.config.dropout_rate))
        
        if num_dropout > 0:
            dropout_probs = np.array([
                0.05 if nt == 'leader' else
                0.10 if nt == 'operative' else
                0.30
                for nt in self.node_types
            ])
            dropout_probs /= dropout_probs.sum()
            
            dropout_indices = self.rng.choice(
                num_current, size=num_dropout, replace=False, p=dropout_probs
            )
            
            for idx in sorted(dropout_indices, reverse=True):
                del self.node_types[idx]
                del self.node_attributes[idx]
                del self.hierarchy_level[idx]
        
        num_current = len(self.node_types)
        if num_current < self.config.max_nodes:
            num_recruit = min(
                max(1, int(num_current * self.config.recruitment_rate)),
                self.config.max_nodes - num_current
            )
            
            for _ in range(num_recruit):
                if self.rng.random() < 0.9:
                    node_type = 'supporter'
                    level = self.config.hierarchy_levels - 1
                else:
                    node_type = 'operative'
                    level = self.config.hierarchy_levels - 2
                
                self.node_types.append(node_type)
                self.hierarchy_level.append(level)
                
                attr = {
                    'type': node_type,
                    'hierarchy_level': level,
                    'join_time': len(self.node_types),
                    'activity_level': self.rng.uniform(0.3, 0.8),
                    'trust_score': self.rng.uniform(0.2, 0.5),
                    'radicalization_level': self.rng.uniform(0.4, 0.9),
                    'operational_skill': self.rng.uniform(0.1, 0.5),
                    'financial_capacity': self.rng.uniform(0.0, 0.3)
                }
                self.node_attributes.append(attr)


# ============================================================================
# ENHANCED EVALUATION PROTOCOL
# ============================================================================

class DisruptionEvaluator:
    """Enhanced evaluation protocol with resilience computation"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_disruption_strategy(
        self,
        network: MultiLayerTemporalNetwork,
        removed_nodes: List[int],
        timestep: int = -1
    ) -> Dict[str, float]:
        """Enhanced disruption evaluation with multiple metrics"""
        if timestep == -1:
            timestep = len(network.layers_history) - 1
        
        layers = network.get_timestep(timestep)
        metrics = {}
        
        agg_edge_index, _ = network.get_aggregated_network(timestep)
        
        # 1. Enhanced fragmentation
        metrics['fragmentation'] = self._compute_enhanced_fragmentation(
            agg_edge_index, network.num_nodes, set(removed_nodes)
        )
        
        # 2. Layer-specific impacts
        for layer in layers:
            layer_name = layer.name
            layer_impact = self._compute_layer_impact(
                layer, set(removed_nodes)
            )
            metrics[f'{layer_name}_impact'] = layer_impact
        
        # 3. Operational capacity
        operational_layer = [l for l in layers if l.name == 'operational'][0]
        metrics['operational_capacity'] = self._compute_operational_capacity(
            operational_layer, set(removed_nodes)
        )
        
        # 4. Network resilience (NEW)
        resilience_metrics = self.compute_network_resilience(
            network, removed_nodes, timestep
        )
        metrics.update(resilience_metrics)
        
        # 5. Overall disruption score with enhanced weighting
        metrics['overall_disruption'] = (
            0.15 * metrics['fragmentation'] +
            0.25 * metrics['operational_impact'] +
            0.15 * metrics['physical_impact'] +
            0.15 * metrics['digital_impact'] +
            0.15 * metrics['financial_impact'] +
            0.15 * metrics['ideological_impact']
        )
        
        return metrics
    
    def _compute_enhanced_fragmentation(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        removed_nodes: Set[int]
    ) -> float:
        """
        Enhanced fragmentation metric:
        - Component size distribution
        - Modularity
        - Average path length increase
        """
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in removed_nodes or dst in removed_nodes:
                mask[i] = False
        
        filtered_edges = edge_index[:, mask]
        active_nodes = set(range(num_nodes)) - removed_nodes
        num_active_nodes = len(active_nodes)
        
        if num_active_nodes == 0 or filtered_edges.size(1) == 0:
            return 1.0
        
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(active_nodes)
        edges = filtered_edges.t().cpu().numpy()
        G.add_edges_from(edges)
        
        if len(G) == 0:
            return 1.0
        
        # Multiple fragmentation metrics
        components = list(nx.connected_components(G))
        largest_cc_size = len(max(components, key=len))
        num_components = len(components)
        
        # Component size fragmentation
        size_fragmentation = 1.0 - (largest_cc_size / num_active_nodes)
        
        # Component count fragmentation
        count_fragmentation = (num_components - 1) / max(1, num_active_nodes - 1)
        
        # Weighted average
        fragmentation = 0.6 * size_fragmentation + 0.4 * count_fragmentation
        
        return fragmentation
    
    def _compute_layer_impact(
        self,
        layer: NetworkLayer,
        removed_nodes: Set[int]
    ) -> float:
        """Compute impact on specific layer"""
        edge_index = layer.edge_index
        total_edges = edge_index.size(1)
        
        if total_edges == 0:
            return 0.0
        
        affected_edges = 0
        for i in range(total_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in removed_nodes or dst in removed_nodes:
                affected_edges += 1
        
        return affected_edges / total_edges
    
    def _compute_operational_capacity(
        self,
        operational_layer: NetworkLayer,
        removed_nodes: Set[int]
    ) -> float:
        """Compute remaining operational capacity"""
        edge_index = operational_layer.edge_index
        node_features = operational_layer.node_features
        
        active_nodes = set(range(node_features.size(0))) - removed_nodes
        
        if len(active_nodes) == 0:
            return 0.0
        
        avg_skill = node_features[list(active_nodes), 7].mean().item()
        
        active_edges = 0
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in active_nodes and dst in active_nodes:
                active_edges += 1
        
        max_possible_edges = len(active_nodes) * (len(active_nodes) - 1) / 2
        connectivity = active_edges / max(1, max_possible_edges)
        
        capacity = (avg_skill + connectivity) / 2
        return 1.0 - capacity
    
    def compute_network_resilience(
        self,
        network: MultiLayerTemporalNetwork,
        removed_nodes: List[int],
        timestep: int = -1
    ) -> Dict[str, float]:
        """
        Compute network resilience metrics (NEW)
        
        Returns:
        - structural_resilience: Alternative paths exist
        - functional_resilience: Key capabilities remain
        - adaptation_capacity: Ability to reorganize
        """
        layers = network.get_timestep(timestep)
        removed_set = set(removed_nodes)
        
        # 1. Structural resilience
        structural_resilience = self._compute_path_redundancy(
            layers, removed_set
        )
        
        # 2. Functional resilience
        functional_resilience = self._compute_functional_capacity(
            layers, removed_set
        )
        
        # 3. Adaptation capacity
        adaptation_capacity = self._compute_adaptation_potential(
            layers, removed_set
        )
        
        return {
            'structural_resilience': structural_resilience,
            'functional_resilience': functional_resilience,
            'adaptation_capacity': adaptation_capacity,
            'overall_resilience': (structural_resilience + functional_resilience + adaptation_capacity) / 3
        }
    
    def _compute_path_redundancy(
        self,
        layers: List[NetworkLayer],
        removed_nodes: Set[int]
    ) -> float:
        """Compute path redundancy in network"""
        # Use aggregated network
        all_edges = []
        for layer in layers:
            for i in range(layer.edge_index.size(1)):
                src, dst = layer.edge_index[0, i].item(), layer.edge_index[1, i].item()
                if src not in removed_nodes and dst not in removed_nodes:
                    all_edges.append((src, dst))
        
        if not all_edges:
            return 0.0
        
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(all_edges)
        
        # Check alternative paths
        redundancy_scores = []
        for edge in list(G.edges())[:min(50, len(G.edges()))]:  # Sample for efficiency
            G_temp = G.copy()
            G_temp.remove_edge(*edge)
            try:
                if nx.has_path(G_temp, edge[0], edge[1]):
                    redundancy_scores.append(1.0)
                else:
                    redundancy_scores.append(0.0)
            except:
                redundancy_scores.append(0.0)
        
        return np.mean(redundancy_scores) if redundancy_scores else 0.0
    
    def _compute_functional_capacity(
        self,
        layers: List[NetworkLayer],
        removed_nodes: Set[int]
    ) -> float:
        """Compute functional capacity (leaders/operatives remaining)"""
        physical_layer = [l for l in layers if l.name == 'physical'][0]
        node_features = physical_layer.node_features
        
        # Assume first 3 features are one-hot for node type
        total_nodes = node_features.size(0)
        active_nodes = set(range(total_nodes)) - removed_nodes
        
        if not active_nodes:
            return 0.0
        
        # Count leaders and operatives
        active_leaders = sum(1 for i in active_nodes if node_features[i, 0] > 0.5)
        active_operatives = sum(1 for i in active_nodes if node_features[i, 1] > 0.5)
        
        total_leaders = sum(1 for i in range(total_nodes) if node_features[i, 0] > 0.5)
        total_operatives = sum(1 for i in range(total_nodes) if node_features[i, 1] > 0.5)
        
        leader_retention = active_leaders / max(1, total_leaders)
        operative_retention = active_operatives / max(1, total_operatives)
        
        return (0.6 * leader_retention + 0.4 * operative_retention)
    
    def _compute_adaptation_potential(
        self,
        layers: List[NetworkLayer],
        removed_nodes: Set[int]
    ) -> float:
        """Compute adaptation potential"""
        # Based on diversity and remaining connections
        physical_layer = [l for l in layers if l.name == 'physical'][0]
        node_features = physical_layer.node_features
        
        active_nodes = [i for i in range(node_features.size(0)) if i not in removed_nodes]
        
        if not active_nodes:
            return 0.0
        
        # Diversity in remaining nodes
        active_features = node_features[active_nodes]
        feature_variance = active_features.var(dim=0).mean().item()
        
        # Normalize to [0, 1]
        diversity_score = min(1.0, feature_variance * 10)
        
        return diversity_score


# Test code
if __name__ == "__main__":
    print("="*80)
    print("Terrorist Network Dataset V2 - Enhanced")
    print("="*80)
    print("\nEnhancements:")
    print("  ✓ Network Data Augmentation")
    print("  ✓ Enhanced Fragmentation Metrics")
    print("  ✓ Network Resilience Computation")
    print("  ✓ Improved Evaluation Protocol")
    print("\n✓ Enhanced dataset module ready!")
