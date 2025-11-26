# Quick Start Guide

Get started with Terrorist Network GNN in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/terrorist-network-gnn.git
cd terrorist-network-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Generate a Network

```python
from src.terrorist_network_dataset import TerroristNetworkGenerator, NetworkConfig

# Configure network
config = NetworkConfig(
    initial_nodes=50,
    max_nodes=80,
    recruitment_rate=0.05
)

# Generate network
generator = TerroristNetworkGenerator(config, seed=42)
network = generator.generate_temporal_network(
    num_timesteps=20,
    num_node_features=64,
    num_edge_features=32
)

print(f"Generated network with {len(network.layers_history)} timesteps")
```

### 2. Build and Train Model

```python
from src.terrorist_network_disruption import MultiLayerTemporalGNN
from src.training import TemporalGNNTrainer

# Build model
model = MultiLayerTemporalGNN(
    num_node_features=64,
    num_edge_features=32,
    hidden_dim=128
)

# Train model
trainer = TemporalGNNTrainer(model, learning_rate=1e-3)
history = trainer.fit(
    train_networks=[network],  # Use your training data
    val_networks=[network],     # Use your validation data
    num_epochs=50
)

print(f"Training complete! Final loss: {history['train_loss'][-1]:.4f}")
```

### 3. Detect Critical Nodes

```python
from src.terrorist_network_disruption import CriticalNodeDetector
import torch

# Get network data
t = -1  # Last timestep
layers = network.get_timestep(t)
agg_edge_index, _ = network.get_aggregated_network(t)
num_nodes = layers[0].node_features.size(0)

# Get embeddings
timestamps = torch.arange(len(network.layers_history), dtype=torch.float32)
embeddings = model(network, timestamps)

# Detect critical nodes
detector = CriticalNodeDetector()
critical_nodes, scores = detector.detect_critical_nodes(
    agg_edge_index,
    num_nodes,
    embeddings,
    top_k=10
)

print(f"Top 10 critical nodes: {critical_nodes.tolist()}")
```

### 4. Evaluate Disruption

```python
from src.terrorist_network_dataset import DisruptionEvaluator

# Evaluate disruption strategy
evaluator = DisruptionEvaluator()
metrics = evaluator.evaluate_disruption_strategy(
    network,
    critical_nodes.tolist(),
    timestep=t
)

print(f"Disruption Score: {metrics['overall_disruption']:.4f}")
print(f"Fragmentation: {metrics['fragmentation']:.4f}")
print(f"Operational Impact: {metrics['operational_impact']:.4f}")
```

## Running Full Experiments

### Quick Test (5-10 minutes)

```bash
python scripts/run_experiment.py \
    --num-networks 3 \
    --num-timesteps 10 \
    --output-dir experiments/quick_test
```

### Full Experiment (30-60 minutes)

```bash
python scripts/run_experiment.py \
    --num-networks 10 \
    --num-timesteps 20 \
    --train-model \
    --run-baselines \
    --run-ablation \
    --output-dir experiments/phase1_full
```

## Using Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/terrorist-network-gnn/blob/main/notebooks/terrorist_network_gnn_demo.ipynb)

1. Click the badge above
2. Run all cells
3. Experiment with the code!

## Common Tasks

### Load a Trained Model

```python
import torch

# Load model
model = MultiLayerTemporalGNN(
    num_node_features=64,
    num_edge_features=32,
    hidden_dim=128
)
model.load_state_dict(torch.load('path/to/model.pt'))
model.eval()
```

### Compare Multiple Methods

```python
from src.baselines import BaselineEvaluator, StaticGCN, StaticGAT

# Initialize evaluator
evaluator = BaselineEvaluator()

# Add baseline methods
evaluator.add_gnn_baseline('gcn', StaticGCN(64, 128))
evaluator.add_gnn_baseline('gat', StaticGAT(64, 128))

# Compare all methods
results = evaluator.compare_all_baselines(
    network=test_network,
    evaluator=disruption_evaluator,
    top_k=10
)

# Print results
for method, metrics in sorted(
    results.items(), 
    key=lambda x: x[1]['disruption'], 
    reverse=True
):
    print(f"{method}: {metrics['disruption']:.4f}")
```

### Visualize Results

```python
from src.statistical_analysis import ResultVisualizer

# Create comparison plot
ResultVisualizer.plot_comparison_boxplot(
    results_dict=baseline_results,
    title="Method Comparison",
    ylabel="Disruption Score",
    save_path="comparison.png"
)
```

## Next Steps

- 📖 Read the [full documentation](docs/)
- 🎓 Try the [tutorial notebooks](notebooks/)
- 🔬 Run the [ablation study](docs/methodology.md#ablation-study)
- 📊 Explore [statistical analysis](docs/methodology.md#statistical-testing)

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or network size:
```python
config = NetworkConfig(
    initial_nodes=30,  # Reduced from 50
    max_nodes=50       # Reduced from 80
)
```

### Import Errors

Make sure you're in the correct directory:
```bash
cd terrorist-network-gnn
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Slow Training

Use GPU if available:
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

## Getting Help

- 📝 [Open an issue](https://github.com/yourusername/terrorist-network-gnn/issues)
- 💬 [Start a discussion](https://github.com/yourusername/terrorist-network-gnn/discussions)
- 📧 Email: your.email@example.com

Happy researching! 🚀
