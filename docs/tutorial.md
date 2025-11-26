# Tutorial: Getting Started with Terrorist Network GNN

This comprehensive tutorial will guide you through using the Terrorist Network GNN system, from basic usage to advanced features.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Intermediate Examples](#intermediate-examples)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: Optional but recommended (CUDA 11.0+)

### Knowledge Prerequisites

- Basic Python programming
- Understanding of graph concepts
- Familiarity with PyTorch (helpful but not required)

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/Navy10021/terrorist-network-gnn.git
cd terrorist-network-gnn
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

#### CPU-only Installation

```bash
pip install -r requirements.txt
pip install -e .
```

#### GPU Installation (Recommended)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install remaining dependencies
pip install -r requirements.txt
pip install -e .
```

### Step 4: Verify Installation

```python
# test_installation.py
import torch
from src.terrorist_network_dataset import TerroristNetworkGenerator, NetworkConfig
from src.advanced_tgnn import AdvancedTemporalGNN

print("✓ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Basic Usage

### Example 1: Generate a Network

```python
from src.terrorist_network_dataset import TerroristNetworkGenerator, NetworkConfig

# Configure network parameters
config = NetworkConfig(
    initial_nodes=30,           # Starting number of nodes
    max_nodes=50,               # Maximum nodes
    recruitment_rate=0.05,      # New member rate per timestep
    dropout_rate=0.02           # Member dropout rate
)

# Create generator with fixed seed for reproducibility
generator = TerroristNetworkGenerator(config, seed=42)

# Generate temporal multi-layer network
network = generator.generate_temporal_network(
    num_timesteps=10,           # Number of time steps
    num_node_features=64,       # Node feature dimension
    num_edge_features=32        # Edge feature dimension
)

print(f"Generated network with {len(network.layers_history)} timesteps")
print(f"Number of layers: {network.num_layers}")
print(f"Layer names: {network.layer_names}")
```

**Output**:
```
Generated network with 10 timesteps
Number of layers: 5
Layer names: ['physical', 'digital', 'financial', 'ideological', 'operational']
```

---

### Example 2: Access Network Layers

```python
# Get network state at specific timestep
t = 0
layers = network.get_timestep(t)

# Examine each layer
for i, layer in enumerate(layers):
    print(f"\nLayer {i}: {layer.name}")
    print(f"  Type: {layer.layer_type}")
    print(f"  Nodes: {layer.node_features.size(0)}")
    print(f"  Edges: {layer.edge_index.size(1)}")
    print(f"  Node features shape: {layer.node_features.shape}")
    print(f"  Edge features shape: {layer.edge_features.shape}")
```

**Output**:
```
Layer 0: physical
  Type: physical
  Nodes: 30
  Edges: 75
  Node features shape: torch.Size([30, 64])
  Edge features shape: torch.Size([75, 32])

Layer 1: digital
  Type: digital
  Nodes: 30
  Edges: 82
  ...
```

---

### Example 3: Build and Use Model

```python
from src.terrorist_network_disruption import MultiLayerTemporalGNN
import torch

# Create model
model = MultiLayerTemporalGNN(
    num_node_features=64,
    num_edge_features=32,
    hidden_dim=128,
    num_layers=5
)

# Get data from last timestep
t = -1
layers = network.get_timestep(t)
physical_layer = layers[0]

# Forward pass
node_embeddings = model(
    physical_layer.node_features,
    physical_layer.edge_index,
    physical_layer.edge_features
)

print(f"Node embeddings shape: {node_embeddings.shape}")
print(f"Embedding dimension: {node_embeddings.size(1)}")
```

**Output**:
```
Node embeddings shape: torch.Size([30, 128])
Embedding dimension: 128
```

---

### Example 4: Detect Critical Nodes

```python
from src.terrorist_network_disruption import CriticalNodeDetector

# Create detector
detector = CriticalNodeDetector()

# Get aggregated network (combines all layers)
agg_edge_index, agg_weights = network.get_aggregated_network(t=-1)
num_nodes = layers[0].node_features.size(0)

# Detect top-k critical nodes
top_k = 5
critical_nodes, scores = detector.detect_critical_nodes(
    edge_index=agg_edge_index,
    num_nodes=num_nodes,
    node_embeddings=node_embeddings,
    top_k=top_k
)

print(f"\nTop {top_k} critical nodes:")
for i, node_id in enumerate(critical_nodes.tolist()):
    print(f"  {i+1}. Node {node_id:2d} - Score: {scores[node_id]:.4f}")
```

**Output**:
```
Top 5 critical nodes:
  1. Node 12 - Score: 0.8745
  2. Node  5 - Score: 0.8432
  3. Node 23 - Score: 0.8201
  4. Node  8 - Score: 0.7988
  5. Node 17 - Score: 0.7756
```

---

## Intermediate Examples

### Example 5: Train Model with Self-Supervised Learning

```python
from src.training import TemporalGNNTrainer
from src.advanced_tgnn import AdvancedTemporalGNN

# Generate training and validation data
train_networks = [
    generator.generate_temporal_network(15, 64, 32)
    for _ in range(5)
]
val_networks = [
    generator.generate_temporal_network(15, 64, 32)
    for _ in range(2)
]

# Create model
model = AdvancedTemporalGNN(
    num_node_features=64,
    num_edge_features=32,
    hidden_dim=128,
    num_temporal_layers=2,
    num_graph_layers=2,
    num_attention_heads=4
)

# Create trainer
trainer = TemporalGNNTrainer(
    model=model,
    learning_rate=1e-3,
    weight_decay=1e-5
)

# Train model
history = trainer.fit(
    train_networks=train_networks,
    val_networks=val_networks,
    num_epochs=50,
    verbose=True
)

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.savefig('training_history.png')
plt.show()
```

---

### Example 6: Compare with Baselines

```python
from src.baselines import (
    StaticGCN, StaticGAT, StaticGraphSAGE,
    CentralityBaseline, BaselineEvaluator
)

# Create baseline models
baselines = {
    'GCN': StaticGCN(num_features=64, hidden_dim=128),
    'GAT': StaticGAT(num_features=64, hidden_dim=128, num_heads=4),
    'GraphSAGE': StaticGraphSAGE(num_features=64, hidden_dim=128),
}

# Evaluate each baseline
evaluator = BaselineEvaluator()
results = {}

for name, model in baselines.items():
    print(f"\nEvaluating {name}...")

    # Get embeddings
    embeddings = model(
        physical_layer.node_features,
        physical_layer.edge_index
    )

    # Detect critical nodes
    critical_nodes, scores = detector.detect_critical_nodes(
        edge_index=agg_edge_index,
        num_nodes=num_nodes,
        node_embeddings=embeddings,
        top_k=5
    )

    results[name] = {
        'critical_nodes': critical_nodes,
        'scores': scores
    }

# Compare results
print("\n=== Comparison ===")
for name, result in results.items():
    print(f"{name}: {result['critical_nodes'].tolist()}")
```

---

### Example 7: Evaluate Disruption Impact

```python
from src.terrorist_network_dataset import DisruptionEvaluator
import networkx as nx

# Create evaluator
evaluator = DisruptionEvaluator()

# Convert to NetworkX for analysis
def to_networkx(edge_index, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    return G

# Original network
G_original = to_networkx(agg_edge_index, num_nodes)

# Simulate removal of critical nodes
G_disrupted = G_original.copy()
G_disrupted.remove_nodes_from(critical_nodes.tolist())

# Compute metrics
print("\n=== Disruption Metrics ===")
print(f"Original nodes: {G_original.number_of_nodes()}")
print(f"Original edges: {G_original.number_of_edges()}")
print(f"Removed nodes: {len(critical_nodes)}")

print(f"\nAfter disruption:")
print(f"Remaining nodes: {G_disrupted.number_of_nodes()}")
print(f"Remaining edges: {G_disrupted.number_of_edges()}")

# Connected components
components_before = nx.number_connected_components(G_original)
components_after = nx.number_connected_components(G_disrupted)
print(f"\nConnected components: {components_before} → {components_after}")

# Largest component
lcc_before = len(max(nx.connected_components(G_original), key=len))
lcc_after = len(max(nx.connected_components(G_disrupted), key=len)) if G_disrupted.number_of_nodes() > 0 else 0
print(f"Largest component: {lcc_before} → {lcc_after} ({-100*(lcc_before-lcc_after)/lcc_before:.1f}%)")

# Efficiency
eff_before = nx.global_efficiency(G_original)
eff_after = nx.global_efficiency(G_disrupted)
print(f"Global efficiency: {eff_before:.3f} → {eff_after:.3f} ({-100*(eff_before-eff_after)/eff_before:.1f}%)")
```

**Output**:
```
=== Disruption Metrics ===
Original nodes: 30
Original edges: 75
Removed nodes: 5

After disruption:
Remaining nodes: 25
Remaining edges: 52

Connected components: 1 → 3
Largest component: 30 → 18 (-40.0%)
Global efficiency: 0.512 → 0.287 (-43.9%)
```

---

## Advanced Features

### Example 8: Predict Temporal Resilience

```python
from src.terrorist_network_disruption import TemporalResiliencePredictor

# Create predictor
predictor = TemporalResiliencePredictor(
    input_dim=128,
    hidden_dim=256,
    num_future_steps=5
)

# Prepare historical network states
network_states = []
for t in range(10):
    layers = network.get_timestep(t)
    embeddings = model(
        layers[0].node_features,
        layers[0].edge_index,
        layers[0].edge_features
    )
    # Aggregate to network-level representation
    network_state = embeddings.mean(dim=0)
    network_states.append(network_state)

# Stack into sequence
network_states_tensor = torch.stack(network_states).unsqueeze(0)

# Predict future resilience
with torch.no_grad():
    resilience_predictions = predictor(network_states_tensor)

print("\nResilience predictions for next 5 timesteps:")
for t, score in enumerate(resilience_predictions[0].tolist()):
    print(f"  t+{t+1}: {score:.4f}")
```

---

### Example 9: Run Complete Experiment

```python
# scripts/run_full_experiment.py
from src.main_experiment import EnhancedExperiment

# Configure experiment
config = {
    'num_networks': 10,
    'num_timesteps': 20,
    'initial_nodes': 50,
    'max_nodes': 80,
    'hidden_dim': 128,
    'num_epochs': 50,
    'output_dir': 'experiments/my_experiment'
}

# Create experiment
experiment = EnhancedExperiment(
    num_networks=config['num_networks'],
    num_timesteps=config['num_timesteps'],
    initial_nodes=config['initial_nodes'],
    max_nodes=config['max_nodes'],
    hidden_dim=config['hidden_dim']
)

# Run all phases
print("Phase 1: Network Generation")
experiment.generate_networks()

print("\nPhase 2: Model Training")
experiment.train_model(num_epochs=config['num_epochs'])

print("\nPhase 3: Baseline Comparison")
experiment.run_baselines()

print("\nPhase 4: Critical Node Detection")
experiment.detect_critical_nodes()

print("\nPhase 5: Statistical Analysis")
results = experiment.analyze_results()

print("\nPhase 6: Save Results")
experiment.save_results(config['output_dir'])

print(f"\nExperiment complete! Results saved to {config['output_dir']}")
```

---

### Example 10: Visualize Results

```python
from src.scripts.visualize_results import ResultVisualizer
import matplotlib.pyplot as plt
import networkx as nx

# Create visualizer
visualizer = ResultVisualizer()

# Visualize network with critical nodes highlighted
fig, ax = plt.subplots(figsize=(12, 8))

# Convert to NetworkX
G = to_networkx(agg_edge_index, num_nodes)

# Node colors (red for critical, blue for others)
node_colors = ['red' if i in critical_nodes else 'lightblue'
               for i in range(num_nodes)]

# Node sizes (larger for critical)
node_sizes = [1000 if i in critical_nodes else 300
              for i in range(num_nodes)]

# Draw network
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                       node_size=node_sizes, alpha=0.8, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

ax.set_title('Network with Critical Nodes Highlighted (Red)', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize temporal evolution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for t, ax in enumerate(axes):
    layers = network.get_timestep(t)
    num_nodes_t = layers[0].node_features.size(0)
    num_edges_t = layers[0].edge_index.size(1)

    G_t = to_networkx(layers[0].edge_index, num_nodes_t)
    pos_t = nx.spring_layout(G_t, seed=42)

    nx.draw(G_t, pos_t, node_size=100, node_color='lightblue',
            alpha=0.6, with_labels=False, ax=ax)
    ax.set_title(f't={t}: {num_nodes_t} nodes, {num_edges_t} edges')

plt.tight_layout()
plt.savefig('temporal_evolution.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```python
# 1. Reduce batch size
trainer.batch_size = 16  # Instead of 32

# 2. Reduce hidden dimension
model = AdvancedTemporalGNN(hidden_dim=64)  # Instead of 128

# 3. Use gradient accumulation
trainer.accumulation_steps = 4

# 4. Use CPU instead
device = torch.device('cpu')
model = model.to(device)
```

---

#### Issue 2: Import Errors

**Error**:
```
ModuleNotFoundError: No module named 'torch_geometric'
```

**Solution**:
```bash
# Install PyTorch Geometric properly
pip install torch-geometric
pip install torch-scatter torch-sparse
```

---

#### Issue 3: Slow Training

**Symptoms**: Training takes very long

**Solutions**:
```python
# 1. Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Reduce number of networks
num_networks = 5  # Instead of 50

# 3. Reduce timesteps
num_timesteps = 10  # Instead of 20

# 4. Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

---

## Best Practices

### 1. Always Set Random Seeds

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

### 2. Use Validation Set

```python
# Split data into train/val/test
train_networks = networks[:70]    # 70%
val_networks = networks[70:85]     # 15%
test_networks = networks[85:]      # 15%
```

### 3. Save Checkpoints

```python
# During training
if epoch % 10 == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
```

### 4. Monitor GPU Memory

```python
if torch.cuda.is_available():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### 5. Log Experiments

```python
import json
from datetime import datetime

experiment_log = {
    'timestamp': datetime.now().isoformat(),
    'config': config,
    'results': results,
    'metrics': metrics
}

with open('experiment_log.json', 'w') as f:
    json.dump(experiment_log, f, indent=2)
```

---

## Next Steps

1. **Read the [Architecture Documentation](architecture.md)** for system design details
2. **Read the [Methodology](methodology.md)** for research background
3. **Explore the [API Reference](api_reference.md)** for detailed function documentation
4. **Check out the [Jupyter Notebooks](../notebooks/)** for interactive examples
5. **Join the [Discussion Forum](https://github.com/Navy10021/terrorist-network-gnn/discussions)** for questions

---

## Additional Resources

- **Paper**: [Link to paper when published]
- **Video Tutorial**: [Link to video tutorial]
- **Slides**: [Link to presentation slides]
- **GitHub**: https://github.com/Navy10021/terrorist-network-gnn

---

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search [existing issues](https://github.com/Navy10021/terrorist-network-gnn/issues)
3. Create a new issue with detailed information
4. Join discussions on GitHub

---

**Happy experimenting! 🚀**
