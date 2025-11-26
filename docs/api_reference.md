# API Reference

Complete API documentation for the Terrorist Network GNN package.

## Table of Contents

- [Core Models](#core-models)
- [Network Generation](#network-generation)
- [Disruption Analysis](#disruption-analysis)
- [Training](#training)
- [Baseline Methods](#baseline-methods)
- [Statistical Analysis](#statistical-analysis)
- [Utilities](#utilities)

---

## Core Models

### AdvancedTemporalGNN

```python
class AdvancedTemporalGNN(
    num_node_features: int,
    num_edge_features: int,
    hidden_dim: int = 128,
    num_temporal_layers: int = 3,
    num_graph_layers: int = 3,
    num_attention_heads: int = 8,
    memory_size: int = 100,
    dropout: float = 0.1,
    max_seq_len: int = 50
)
```

Advanced Temporal Graph Neural Network with state-of-the-art components.

**Parameters:**
- `num_node_features` (int): Number of input node features
- `num_edge_features` (int): Number of input edge features
- `hidden_dim` (int): Hidden dimension size (default: 128)
- `num_temporal_layers` (int): Number of temporal layers (default: 3)
- `num_graph_layers` (int): Number of graph layers (default: 3)
- `num_attention_heads` (int): Number of attention heads (default: 8)
- `memory_size` (int): Size of memory bank (default: 100)
- `dropout` (float): Dropout rate (default: 0.1)
- `max_seq_len` (int): Maximum sequence length (default: 50)

**Methods:**

#### forward()

```python
def forward(
    node_features_seq: List[torch.Tensor],
    edge_indices_seq: List[torch.Tensor],
    edge_features_seq: List[torch.Tensor],
    timestamps: torch.Tensor,
    batch_seq: Optional[List[torch.Tensor]] = None
) -> torch.Tensor
```

Forward pass through the network.

**Parameters:**
- `node_features_seq`: List of node feature tensors for each timestep
- `edge_indices_seq`: List of edge index tensors
- `edge_features_seq`: List of edge feature tensors
- `timestamps`: Tensor of timestamps
- `batch_seq`: Optional batch assignments

**Returns:**
- `embeddings`: Node embeddings tensor [num_nodes, hidden_dim]

**Example:**

```python
from src.advanced_tgnn import AdvancedTemporalGNN

# Initialize model
model = AdvancedTemporalGNN(
    num_node_features=64,
    num_edge_features=32,
    hidden_dim=128
)

# Prepare data
node_features_seq = [torch.randn(50, 64) for _ in range(10)]
edge_indices_seq = [torch.randint(0, 50, (2, 100)) for _ in range(10)]
edge_features_seq = [torch.randn(100, 32) for _ in range(10)]
timestamps = torch.arange(10, dtype=torch.float32)

# Forward pass
embeddings = model(
    node_features_seq,
    edge_indices_seq,
    edge_features_seq,
    timestamps
)

print(f"Output shape: {embeddings.shape}")  # [50, 128]
```

#### reconstruct_features()

```python
def reconstruct_features(embeddings: torch.Tensor) -> torch.Tensor
```

Reconstruct node features from embeddings (for self-supervised learning).

**Parameters:**
- `embeddings`: Node embeddings [num_nodes, hidden_dim]

**Returns:**
- `reconstructed_features`: [num_nodes, num_node_features]

---

### MultiLayerTemporalGNN

```python
class MultiLayerTemporalGNN(
    num_node_features: int,
    num_edge_features: int,
    hidden_dim: int,
    num_layers: int = 5,
    layer_fusion: str = 'attention',
    **kwargs
)
```

Multi-layer temporal GNN for processing multi-layer networks.

**Parameters:**
- `num_node_features` (int): Number of node features
- `num_edge_features` (int): Number of edge features
- `hidden_dim` (int): Hidden dimension
- `num_layers` (int): Number of network layers (default: 5)
- `layer_fusion` (str): Fusion method - 'attention', 'concat', or 'weighted_sum'

**Example:**

```python
from src.terrorist_network_disruption import MultiLayerTemporalGNN

model = MultiLayerTemporalGNN(
    num_node_features=64,
    num_edge_features=32,
    hidden_dim=128,
    num_layers=5,
    layer_fusion='attention'
)

# Forward pass with multi-layer network
embeddings = model(multi_layer_network, timestamps)
```

---

## Network Generation

### TerroristNetworkGenerator

```python
class TerroristNetworkGenerator(config: NetworkConfig, seed: int = 42)
```

Generate realistic multi-layer temporal terrorist networks.

**Parameters:**
- `config`: NetworkConfig object with generation parameters
- `seed`: Random seed for reproducibility

**Methods:**

#### generate_temporal_network()

```python
def generate_temporal_network(
    num_timesteps: int = 20,
    num_node_features: int = 64,
    num_edge_features: int = 32,
    device: torch.device = torch.device('cpu')
) -> MultiLayerTemporalNetwork
```

Generate complete temporal network.

**Returns:**
- `network`: MultiLayerTemporalNetwork object

**Example:**

```python
from src.terrorist_network_dataset import TerroristNetworkGenerator, NetworkConfig

# Configure network
config = NetworkConfig(
    initial_nodes=50,
    max_nodes=80,
    recruitment_rate=0.05,
    dropout_rate=0.02,
    physical_density=0.15,
    digital_density=0.35
)

# Generate network
generator = TerroristNetworkGenerator(config, seed=42)
network = generator.generate_temporal_network(
    num_timesteps=20,
    num_node_features=64,
    num_edge_features=32
)

print(f"Generated {len(network.layers_history)} timesteps")
```

### NetworkConfig

```python
@dataclass
class NetworkConfig:
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
```

Configuration dataclass for network generation.

---

## Disruption Analysis

### CriticalNodeDetector

```python
class CriticalNodeDetector(importance_metrics: List[str] = None)
```

Detect critical nodes using ensemble of metrics.

**Parameters:**
- `importance_metrics`: List of metrics to use. Available:
  - 'degree_centrality'
  - 'betweenness_centrality'
  - 'eigenvector_centrality'
  - 'pagerank'
  - 'structural_holes'
  - 'gnn_importance'

**Methods:**

#### detect_critical_nodes()

```python
def detect_critical_nodes(
    edge_index: torch.Tensor,
    num_nodes: int,
    embeddings: Optional[torch.Tensor] = None,
    top_k: int = 10,
    aggregation: str = 'weighted_sum'
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
```

Detect top-k critical nodes.

**Returns:**
- `critical_nodes`: Tensor of node indices
- `importance_scores`: Dict of scores for each metric

**Example:**

```python
from src.terrorist_network_disruption import CriticalNodeDetector

detector = CriticalNodeDetector(
    importance_metrics=[
        'degree_centrality',
        'betweenness_centrality',
        'gnn_importance'
    ]
)

critical_nodes, scores = detector.detect_critical_nodes(
    edge_index=network_edges,
    num_nodes=50,
    embeddings=node_embeddings,
    top_k=10
)

print(f"Critical nodes: {critical_nodes.tolist()}")
print(f"Metrics used: {list(scores.keys())}")
```

### DisruptionEvaluator

```python
class DisruptionEvaluator()
```

Evaluate effectiveness of disruption strategies.

#### evaluate_disruption_strategy()

```python
def evaluate_disruption_strategy(
    network: MultiLayerTemporalNetwork,
    removed_nodes: List[int],
    timestep: int = -1
) -> Dict[str, float]
```

Evaluate disruption impact.

**Returns:**
- Dictionary with metrics:
  - 'overall_disruption': Overall score
  - 'fragmentation': Network fragmentation
  - 'operational_capacity': Remaining capacity
  - Layer-specific impacts

**Example:**

```python
from src.terrorist_network_dataset import DisruptionEvaluator

evaluator = DisruptionEvaluator()
metrics = evaluator.evaluate_disruption_strategy(
    network=test_network,
    removed_nodes=[1, 5, 10, 15, 20],
    timestep=-1
)

print(f"Overall disruption: {metrics['overall_disruption']:.4f}")
print(f"Fragmentation: {metrics['fragmentation']:.4f}")
```

---

## Training

### TemporalGNNTrainer

```python
class TemporalGNNTrainer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device = None
)
```

Trainer for temporal GNN with self-supervised learning.

**Methods:**

#### fit()

```python
def fit(
    train_networks: List,
    val_networks: List,
    num_epochs: int = 50,
    early_stopping_patience: int = 10,
    verbose: bool = True
) -> Dict
```

Train model with early stopping.

**Returns:**
- `history`: Dictionary with training metrics

**Example:**

```python
from src.training import TemporalGNNTrainer

trainer = TemporalGNNTrainer(
    model=model,
    learning_rate=1e-3,
    weight_decay=1e-5
)

history = trainer.fit(
    train_networks=train_data,
    val_networks=val_data,
    num_epochs=50,
    early_stopping_patience=10
)

print(f"Best val loss: {min(history['val_loss']):.4f}")
```

---

## Baseline Methods

### StaticGCN

```python
class StaticGCN(
    num_node_features: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.1
)
```

Static Graph Convolutional Network baseline.

### StaticGAT

```python
class StaticGAT(
    num_node_features: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout: float = 0.1
)
```

Static Graph Attention Network baseline.

### BaselineEvaluator

```python
class BaselineEvaluator(device: torch.device = None)
```

Unified interface for evaluating all baselines.

**Example:**

```python
from src.baselines import BaselineEvaluator, StaticGCN, StaticGAT

evaluator = BaselineEvaluator()

# Add baselines
evaluator.add_gnn_baseline('gcn', StaticGCN(64, 128))
evaluator.add_gnn_baseline('gat', StaticGAT(64, 128))

# Compare methods
results = evaluator.compare_all_baselines(
    network=test_network,
    evaluator=disruption_evaluator,
    top_k=10
)
```

---

## Statistical Analysis

### StatisticalAnalyzer

```python
class StatisticalAnalyzer(alpha: float = 0.05)
```

Statistical significance testing framework.

**Methods:**

#### compare_multiple_methods()

```python
def compare_multiple_methods(
    results_dict: Dict[str, List[float]],
    reference_method: str = None
) -> Dict[str, Dict]
```

Compare multiple methods with statistical tests.

**Example:**

```python
from src.statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(alpha=0.05)

results = {
    'our_method': [0.75, 0.78, 0.76, ...],
    'baseline_1': [0.65, 0.68, 0.66, ...],
    'baseline_2': [0.70, 0.72, 0.71, ...]
}

comparison = analyzer.compare_multiple_methods(
    results_dict=results,
    reference_method='our_method'
)

for method, stats in comparison['comparisons'].items():
    print(f"{method}: p={stats['t_test'].p_value:.4f}")
```

### ResultVisualizer

Static methods for result visualization.

#### plot_comparison_boxplot()

```python
@staticmethod
def plot_comparison_boxplot(
    results_dict: Dict[str, List[float]],
    title: str = "Method Comparison",
    ylabel: str = "Performance",
    save_path: Optional[str] = None
)
```

---

## Utilities

### Logging

```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Training started")
```

### Device Management

```python
import torch

# Auto-detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model = model.to(device)

# Move data to device
data = data.to(device)
```

### Checkpointing

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## Complete Example

```python
"""
Complete workflow example
"""

import torch
from src.terrorist_network_dataset import (
    TerroristNetworkGenerator, NetworkConfig, DisruptionEvaluator
)
from src.terrorist_network_disruption import (
    MultiLayerTemporalGNN, CriticalNodeDetector
)
from src.training import TemporalGNNTrainer

# 1. Configure and generate network
config = NetworkConfig(
    initial_nodes=50,
    max_nodes=80,
    recruitment_rate=0.05
)

generator = TerroristNetworkGenerator(config, seed=42)
network = generator.generate_temporal_network(
    num_timesteps=20,
    num_node_features=64,
    num_edge_features=32
)

# 2. Create and train model
model = MultiLayerTemporalGNN(
    num_node_features=64,
    num_edge_features=32,
    hidden_dim=128
)

trainer = TemporalGNNTrainer(model)
history = trainer.fit(
    train_networks=[network],
    val_networks=[network],
    num_epochs=50
)

# 3. Detect critical nodes
detector = CriticalNodeDetector()
timestamps = torch.arange(20, dtype=torch.float32)
embeddings = model(network, timestamps)

t = -1
agg_edge_index, _ = network.get_aggregated_network(t)
num_nodes = network.get_timestep(t)[0].node_features.size(0)

critical_nodes, scores = detector.detect_critical_nodes(
    agg_edge_index,
    num_nodes,
    embeddings,
    top_k=10
)

# 4. Evaluate disruption
evaluator = DisruptionEvaluator()
metrics = evaluator.evaluate_disruption_strategy(
    network,
    critical_nodes.tolist(),
    timestep=t
)

print(f"Critical nodes: {critical_nodes.tolist()}")
print(f"Disruption score: {metrics['overall_disruption']:.4f}")
```

---

For more examples, see the [notebooks](../notebooks/) directory.
