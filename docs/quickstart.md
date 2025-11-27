# Quick Start Guide

Get up and running with Terrorist Network T-GNN in 5 minutes!

## Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU)
- 8GB RAM minimum

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/terrorist-network-tgnn.git
cd terrorist-network-tgnn
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (choose your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch_geometric; print('PyG installed successfully')"
```

## Running Your First Experiment

### Option 1: Interactive Notebook (Easiest)

1. Open `notebooks/terrorist_network_gnn_demo.ipynb`
2. Run cells sequentially
3. Results saved to `results/` directory

### Option 2: Python Script

Create `quick_demo.py`:

```python
import sys
sys.path.append('./src')

from terrorist_network_dataset import NetworkConfig, TerroristNetworkGenerator
from main_experiment import EnhancedExperiment

# Quick demo configuration
config = NetworkConfig(
    initial_nodes=30,
    max_nodes=50
)

model_config = {
    'num_node_features': 64,
    'num_edge_features': 32,
    'hidden_dim': 128
}

# Run experiment
experiment = EnhancedExperiment(
    config=config,
    model_config=model_config,
    output_dir='results/quick_demo'
)

experiment.run_complete_experiment(
    num_networks=3,
    num_timesteps=10,
    train_model=True,
    run_baselines=True,
    run_ablation=False
)

print("✅ Experiment complete! Check results/quick_demo/")
```

Run it:

```bash
python quick_demo.py
```

### Option 3: Command Line

```bash
cd src
python main_experiment.py --mode demo --output ../results/demo
```

## Understanding the Output

After running, you'll find in `results/`:

```
results/
├── quick_demo/
│   ├── figures/
│   │   ├── q1_performance.png          # Critical node results
│   │   ├── q2_resilience.png           # Resilience predictions
│   │   ├── q3_adversarial.png          # Adversarial analysis
│   │   ├── baseline_comparison.png     # Method comparison
│   │   └── statistical_analysis.png    # Statistical tests
│   ├── results_phase1.json             # All metrics
│   └── experiment_log.txt              # Detailed log
```

## Next Steps

### 1. Customize Network Configuration

```python
config = NetworkConfig(
    initial_nodes=50,        # More nodes
    max_nodes=80,
    recruitment_rate=0.05,   # 5% growth per timestep
    dropout_rate=0.02,       # 2% attrition
    physical_density=0.15,   # Sparse meetings
    digital_density=0.35     # Dense online communication
)
```

### 2. Tune Model Architecture

```python
model_config = {
    'hidden_dim': 256,              # Larger model
    'num_temporal_layers': 4,       # Deeper temporal processing
    'num_attention_heads': 16,      # More attention heads
    'memory_size': 200              # Larger memory
}
```

### 3. Run Full Experiment

```python
experiment.run_complete_experiment(
    num_networks=10,      # More networks for statistical power
    num_timesteps=20,     # Longer temporal sequences
    train_model=True,
    run_baselines=True,
    run_ablation=True     # Full ablation study
)
```

## Common Issues

### CUDA Out of Memory

```python
# Reduce model size
model_config['hidden_dim'] = 64  # Instead of 128
```

### Slow Training

```python
# Use fewer networks/timesteps
num_networks = 5
num_timesteps = 10
```

### Import Errors

```python
# Ensure src is in path
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))
```

## Getting Help

- 📖 Read [README.md](../README.md) for detailed documentation
- 🏗️ Check [architecture.md](architecture.md) for system design
- 🐛 Report bugs on [GitHub Issues](https://github.com/yourusername/terrorist-network-tgnn/issues)
- 💬 Ask questions in [Discussions](https://github.com/yourusername/terrorist-network-tgnn/discussions)

## Congratulations! 🎉

You've successfully run your first terrorist network disruption experiment!

For more advanced usage, explore:
- [Advanced Usage Guide](advanced_usage.md)
- [API Reference](api_reference.md)
- [Research Questions Details](research_questions.md)
