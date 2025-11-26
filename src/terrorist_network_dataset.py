"""
Terrorist Network Dataset Generator & Evaluation Protocol
=========================================================

Generates realistic multi-layer temporal terrorist networks with:
1. Network evolution patterns
2. Recruitment dynamics
3. Communication patterns
4. Financial flows
5. Operational structure

Based on real-world terrorist network characteristics from research literature.
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
    CriticalNodeDetector,
    NetworkDisruptionOptimizer,
    AdversarialNetworkAttack
)


# ============================================================================
# REALISTIC TERRORIST NETWORK GENERATOR
# ============================================================================

@dataclass
class NetworkConfig:
    """Configuration for terrorist network generation"""
    # Network size
    initial_nodes: int = 50
    max_nodes: int = 100
    
    # Structural properties
    avg_degree: float = 4.0  # Average connections per node
    clustering_coefficient: float = 0.3  # Community structure
    hierarchy_levels: int = 3  # Leadership hierarchy
    
    # Temporal dynamics
    recruitment_rate: float = 0.05  # New members per timestep
    dropout_rate: float = 0.02  # Members leaving
    
    # Layer-specific properties
    physical_density: float = 0.3  # Physical meetings are sparse
    digital_density: float = 0.6  # Digital comm is dense
    financial_density: float = 0.2  # Money flows are sparse
    ideological_density: float = 0.5  # Shared ideology
    operational_density: float = 0.15  # Operations are secretive
    
    # Node types distribution
    leader_ratio: float = 0.05
    operative_ratio: float = 0.20
    supporter_ratio: float = 0.75


class TerroristNetworkGenerator:
    """
    Generate realistic terrorist network with multiple layers
    
    Based on research characteristics:
    - Scale-free structure (power-law degree distribution)
    - High clustering (small-world property)
    - Hierarchical organization
    - Cell structure for operations
    - Temporal evolution with recruitment/dropout
    """
    
    def __init__(self, config: NetworkConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.node_types = []  # 'leader', 'operative', 'supporter'
        self.node_attributes = []  # Dict of attributes per node
        self.hierarchy_level = []  # 0 (top) to N (bottom)
        
    def generate_temporal_network(
        self,
        num_timesteps: int = 20,
        num_node_features: int = 64,
        num_edge_features: int = 32,
        device: torch.device = torch.device('cpu')
    ) -> MultiLayerTemporalNetwork:
        """
        Generate complete multi-layer temporal network
        """
        print(f"\nGenerating terrorist network with {num_timesteps} timesteps...")
        
        network = MultiLayerTemporalNetwork(
            num_nodes=self.config.max_nodes,
            num_layers=5
        )
        
        # Initialize first timestep
        self._initialize_network()
        
        for t in range(num_timesteps):
            # Generate layers for this timestep
            layers = self._generate_timestep(
                t, num_node_features, num_edge_features, device
            )
            network.add_timestep(layers)
            
            # Evolve network
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
        """Initialize network structure at t=0"""
        num_nodes = self.config.initial_nodes
        
        # Assign node types based on ratios
        num_leaders = max(1, int(num_nodes * self.config.leader_ratio))
        num_operatives = int(num_nodes * self.config.operative_ratio)
        num_supporters = num_nodes - num_leaders - num_operatives
        
        self.node_types = (
            ['leader'] * num_leaders +
            ['operative'] * num_operatives +
            ['supporter'] * num_supporters
        )
        random.shuffle(self.node_types)
        
        # Assign hierarchy levels
        self.hierarchy_level = []
        for node_type in self.node_types:
            if node_type == 'leader':
                level = self.rng.randint(0, 2)  # Top 2 levels
            elif node_type == 'operative':
                level = self.rng.randint(1, 3)  # Middle levels
            else:
                level = self.rng.randint(2, self.config.hierarchy_levels)
            self.hierarchy_level.append(level)
        
        # Initialize node attributes
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
        
        # Generate node features
        node_features = self._generate_node_features(
            num_nodes, num_node_features, device
        )
        
        # Generate each layer
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
            features[i, 9] = attr['join_time'] / 100.0  # Normalized time
            
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
        """
        Generate edges for a layer
        
        Args:
            density: Target edge density
            preferential_attachment: Use power-law degree distribution
            hierarchy_respect: Edges more likely within same hierarchy level
        """
        edges = []
        edge_weights = []
        
        target_edges = int(num_nodes * (num_nodes - 1) / 2 * density)
        
        if preferential_attachment:
            # Scale-free network generation (Barabási-Albert style)
            degrees = defaultdict(int)
            
            for _ in range(target_edges):
                if len(edges) < num_nodes:
                    # Initial phase: random edges
                    i, j = self.rng.choice(num_nodes, size=2, replace=False)
                else:
                    # Preferential attachment based on degree
                    # Higher degree nodes more likely to get new edges
                    probs = np.array([degrees[i] + 1 for i in range(num_nodes)])
                    probs = probs / probs.sum()
                    i = self.rng.choice(num_nodes, p=probs)
                    
                    # Second node: prefer similar hierarchy if enabled
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
                    
                    # Edge weight based on node importance
                    weight = (
                        self.node_attributes[i]['trust_score'] * 
                        self.node_attributes[j]['trust_score']
                    )
                    edge_weights.append(weight)
        else:
            # Random graph
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
        """Financial flows - very sparse, directed by hierarchy"""
        edges, weights = self._generate_layer_edges(
            num_nodes,
            self.config.financial_density,
            preferential_attachment=False,
            hierarchy_respect=True
        )
        
        # Filter by financial capacity
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
        """Ideological connections - moderate density"""
        edges, weights = self._generate_layer_edges(
            num_nodes,
            self.config.ideological_density,
            preferential_attachment=True,
            hierarchy_respect=False
        )
        
        # Weight by radicalization similarity
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
        """Operational connections - very sparse, only operatives"""
        # Only connect operatives and leaders
        operative_indices = [
            i for i, nt in enumerate(self.node_types)
            if nt in ['operative', 'leader']
        ]
        
        edges = []
        weights = []
        
        if len(operative_indices) > 1:
            num_cells = max(1, len(operative_indices) // 5)
            random.shuffle(operative_indices)
            
            # Create cells
            for cell_id in range(num_cells):
                cell = operative_indices[cell_id::num_cells]
                
                # Connect within cell
                for i in range(len(cell)):
                    for j in range(i + 1, len(cell)):
                        if self.rng.random() < 0.5:  # Sparse even within cells
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
        # Dropout: some members leave
        num_current = len(self.node_types)
        num_dropout = max(0, int(num_current * self.config.dropout_rate))
        
        if num_dropout > 0:
            # Supporters more likely to drop out
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
            
            # Remove from end to avoid index issues
            for idx in sorted(dropout_indices, reverse=True):
                del self.node_types[idx]
                del self.node_attributes[idx]
                del self.hierarchy_level[idx]
        
        # Recruitment: add new members
        num_current = len(self.node_types)
        if num_current < self.config.max_nodes:
            num_recruit = min(
                max(1, int(num_current * self.config.recruitment_rate)),
                self.config.max_nodes - num_current
            )
            
            for _ in range(num_recruit):
                # New members are usually supporters
                if self.rng.random() < 0.9:
                    node_type = 'supporter'
                    level = self.config.hierarchy_levels - 1
                else:
                    node_type = 'operative'
                    level = self.config.hierarchy_levels - 2
                
                self.node_types.append(node_type)
                self.hierarchy_level.append(level)
                
                # New member attributes
                attr = {
                    'type': node_type,
                    'hierarchy_level': level,
                    'join_time': len(self.node_types),
                    'activity_level': self.rng.uniform(0.3, 0.8),
                    'trust_score': self.rng.uniform(0.2, 0.5),  # Low initially
                    'radicalization_level': self.rng.uniform(0.4, 0.9),
                    'operational_skill': self.rng.uniform(0.1, 0.5),
                    'financial_capacity': self.rng.uniform(0.0, 0.3)
                }
                self.node_attributes.append(attr)


# ============================================================================
# EVALUATION PROTOCOL
# ============================================================================

class DisruptionEvaluator:
    """
    Comprehensive evaluation protocol for network disruption
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_disruption_strategy(
        self,
        network: MultiLayerTemporalNetwork,
        removed_nodes: List[int],
        timestep: int = -1
    ) -> Dict[str, float]:
        """
        Evaluate effectiveness of disruption strategy
        
        Metrics:
        1. Network fragmentation
        2. Information flow disruption
        3. Operational capacity reduction
        4. Financial flow interruption
        5. Recruitment capability impact
        """
        if timestep == -1:
            timestep = len(network.layers_history) - 1
        
        layers = network.get_timestep(timestep)
        
        metrics = {}
        
        # Aggregate network
        agg_edge_index, _ = network.get_aggregated_network(timestep)
        
        # 1. Network fragmentation
        metrics['fragmentation'] = self._compute_fragmentation(
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
        
        # 4. Overall disruption score
        # Adjusted weights: Less emphasis on fragmentation (hard to achieve),
        # More on operational/layer impacts (more realistic)
        metrics['overall_disruption'] = (
            0.15 * metrics['fragmentation'] +          # 30% → 15%
            0.25 * metrics['operational_impact'] +     # 20% → 25%
            0.15 * metrics['physical_impact'] +        # 15% (same)
            0.15 * metrics['digital_impact'] +         # 15% (same)
            0.15 * metrics['financial_impact'] +       # 10% → 15%
            0.15 * metrics['ideological_impact']       # 10% → 15%
        )
        
        return metrics
    
    def _compute_fragmentation(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        removed_nodes: Set[int]
    ) -> float:
        """
        Compute network fragmentation score
        
        1.0 = completely fragmented (largest component is very small relative to remaining nodes)
        0.0 = no fragmentation (all remaining nodes in one component)
        """
        # Filter edges
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in removed_nodes or dst in removed_nodes:
                mask[i] = False
        
        filtered_edges = edge_index[:, mask]
        
        # Calculate active nodes
        active_nodes = set(range(num_nodes)) - removed_nodes
        num_active_nodes = len(active_nodes)
        
        if num_active_nodes == 0:
            return 1.0
        
        if filtered_edges.size(1) == 0:
            return 1.0
        
        # Convert to NetworkX for component analysis
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(active_nodes)
        edges = filtered_edges.t().cpu().numpy()
        G.add_edges_from(edges)
        
        # Fragmentation = 1 - (size of largest component / number of active nodes)
        # This correctly measures fragmentation relative to remaining network
        if len(G) == 0:
            return 1.0
        
        largest_cc = max(nx.connected_components(G), key=len)
        fragmentation = 1.0 - len(largest_cc) / num_active_nodes
        
        return fragmentation
    
    def _compute_layer_impact(
        self,
        layer: NetworkLayer,
        removed_nodes: Set[int]
    ) -> float:
        """Compute impact on specific layer"""
        edge_index = layer.edge_index
        
        # Count affected edges
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
        """
        Compute remaining operational capacity
        
        Based on:
        - Number of operational cells remaining
        - Connectivity within cells
        - Skill levels of remaining operatives
        """
        edge_index = operational_layer.edge_index
        node_features = operational_layer.node_features
        
        # Filter to active nodes
        active_nodes = set(range(node_features.size(0))) - removed_nodes
        
        if len(active_nodes) == 0:
            return 0.0
        
        # Operational capacity = average skill * connectivity
        avg_skill = node_features[list(active_nodes), 7].mean().item()  # Skill at index 7
        
        # Connectivity among active operatives
        active_edges = 0
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in active_nodes and dst in active_nodes:
                active_edges += 1
        
        max_possible_edges = len(active_nodes) * (len(active_nodes) - 1) / 2
        connectivity = active_edges / max(1, max_possible_edges)
        
        capacity = (avg_skill + connectivity) / 2
        return 1.0 - capacity  # Return as reduction (higher = more disrupted)


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("Terrorist Network Dataset Generator & Evaluation")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    config = NetworkConfig(
        initial_nodes=50,
        max_nodes=80,
        recruitment_rate=0.05,
        dropout_rate=0.02
    )
    
    # Generate network
    print("\n1. Generating Multi-Layer Temporal Network")
    print("-" * 80)
    generator = TerroristNetworkGenerator(config)
    network = generator.generate_temporal_network(
        num_timesteps=15,
        num_node_features=64,
        num_edge_features=32,
        device=device
    )
    
    # Analyze network structure
    print("\n2. Network Structure Analysis")
    print("-" * 80)
    t = -1  # Last timestep
    layers = network.get_timestep(t)
    
    for layer in layers:
        print(f"\n{layer.name.upper()} Layer:")
        print(f"  Nodes: {layer.node_features.size(0)}")
        print(f"  Edges: {layer.edge_index.size(1)}")
        print(f"  Density: {layer.edge_index.size(1) / (layer.node_features.size(0) * (layer.node_features.size(0) - 1) / 2):.3f}")
    
    # Test disruption optimization
    print("\n3. Testing Disruption Strategies")
    print("-" * 80)
    
    # Get aggregated network
    agg_edge_index, _ = network.get_aggregated_network(t)
    num_nodes = layers[0].node_features.size(0)
    
    # Detect critical nodes
    detector = CriticalNodeDetector()
    critical_nodes, _ = detector.detect_critical_nodes(
        agg_edge_index, num_nodes, top_k=10
    )
    
    print(f"Top 10 critical nodes: {critical_nodes.tolist()}")
    
    # Find optimal disruption
    optimizer = NetworkDisruptionOptimizer(algorithm='greedy')
    removed, disruption_scores, metadata = optimizer.optimize_disruption(
        agg_edge_index, num_nodes, budget_k=5
    )
    
    print(f"\nOptimal removal sequence: {removed}")
    print(f"Disruption progression: {[f'{s:.3f}' for s in disruption_scores]}")
    
    # Evaluate disruption
    print("\n4. Disruption Evaluation")
    print("-" * 80)
    evaluator = DisruptionEvaluator()
    metrics = evaluator.evaluate_disruption_strategy(
        network, removed, timestep=t
    )
    
    print("\nDisruption Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print("\n" + "="*80)
    print("Dataset Generation & Evaluation Complete!")
    print("="*80)
