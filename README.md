# Terrorist Network Disruption using Temporal Graph Neural Networks

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/)

(https://github.com/Navy10021/terrorist-network-gnn/actions)
[![codecov](https://codecov.io/gh/Navy10021/terrorist-network-gnn/branch/main/graph/badge.svg)](https://codecov.io/gh/Navy10021/terrorist-network-gnn)
[![Documentation Status](https://readthedocs.org/projects/terrorist-network-gnn/badge/?version=latest)](https://terrorist-network-gnn.readthedocs.io/)

**Advanced Temporal Graph Neural Networks for Critical Node Detection and Network Resilience Analysis**

[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Documentation](docs/) •
[Examples](notebooks/) •
[Paper](https://arxiv.org/) •
[Citation](#-citation)

</div>

---

## 🎯 Overview

Terrorist networks pose significant security challenges due to their adaptive, decentralized nature. This research develops **advanced machine learning techniques** to identify critical vulnerabilities and predict network resilience patterns.

### Key Innovations

🧠 **State-of-the-Art T-GNN Architecture**
- Multi-head temporal attention mechanisms
- Memory-augmented recurrent networks
- Graph transformer layers
- Adaptive time encoding

🌐 **Multi-Layer Network Modeling**
- Physical, Digital, Financial, Ideological, Operational layers
- Realistic temporal evolution
- Cross-layer dependency analysis

📊 **Comprehensive Evaluation Framework**
- 10+ baseline comparisons (GCN, GAT, GraphSAGE, etc.)
- Statistical significance testing
- Ablation studies
- Publication-ready visualizations

## 🔍 Research Questions

| Question | Focus | Approach |
|----------|-------|----------|
| **Q1** | Critical Node Detection | Ensemble of centrality + GNN-based importance |
| **Q2** | Temporal Resilience | Prediction of network reconstruction patterns |
| **Q3** | Adversarial Robustness | Analysis of adaptive network responses |

## 📈 Results Preview

<div align="center">

| Method | Disruption Score | Improvement | Statistical Sig. |
|--------|-----------------|-------------|------------------|
| **Our T-GNN** | **0.7823** | - | - |
| Simple T-GNN | 0.7145 | +9.5% | ✓ p < 0.001 |
| Static GAT | 0.6832 | +14.5% | ✓ p < 0.001 |
| Static GCN | 0.6453 | +21.2% | ✓ p < 0.001 |
| PageRank | 0.5934 | +31.8% | ✓ p < 0.001 |

</div>

## 🚀 Installation

### Quick Install

```bash
git clone https://github.com/Navy10021/terrorist-network-gnn.git
cd terrorist-network-gnn
pip install -r requirements.txt
pip install -e .
```

### With GPU Support

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install remaining dependencies
pip install -r requirements.txt
pip install -e .
```

### Google Colab (No Installation!)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Navy10021/terrorist-network-gnn/blob/main/notebooks/terrorist_network_gnn_demo.ipynb)

For detailed installation instructions, see [Installation Guide](docs/installation.md).

## ⚡ Quick Start

### Basic Usage

```python
from src.terrorist_network_dataset import TerroristNetworkGenerator, NetworkConfig
from src.terrorist_network_disruption import MultiLayerTemporalGNN, CriticalNodeDetector

# Generate network
config = NetworkConfig(initial_nodes=50, max_nodes=80)
generator = TerroristNetworkGenerator(config)
network = generator.generate_temporal_network(num_timesteps=20)

# Build and train model
model = MultiLayerTemporalGNN(num_node_features=64, num_edge_features=32, hidden_dim=128)

# Detect critical nodes
detector = CriticalNodeDetector()
critical_nodes, scores = detector.detect_critical_nodes(
    edge_index, num_nodes, embeddings, top_k=10
)

print(f"Critical nodes: {critical_nodes.tolist()}")
```

### Command Line

```bash
# Quick test (5-10 minutes)
python scripts/run_experiment.py --num-networks 3 --num-timesteps 10

# Full experiment (30-60 minutes)
python scripts/run_experiment.py \
    --num-networks 10 \
    --num-timesteps 20 \
    --train-model \
    --run-baselines \
    --run-ablation
```

See [Quick Start Guide](QUICKSTART.md) for more examples.

## 📁 Project Structure

```
terrorist-network-gnn/
├── src/                    # Source code
│   ├── advanced_tgnn.py   # T-GNN architecture
│   ├── terrorist_network_disruption.py
│   └── ...
├── scripts/               # Executable scripts
│   ├── run_experiment.py
│   ├── evaluate_model.py
│   └── visualize_results.py
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks
├── docs/                # Documentation
└── experiments/         # Output directory
```

## 📊 Key Features

### Multi-Layer Network Support

```python
# Access different network layers
physical_layer = network.get_layer_sequence(0)      # Face-to-face meetings
digital_layer = network.get_layer_sequence(1)       # Online communications
financial_layer = network.get_layer_sequence(2)     # Money transfers
ideological_layer = network.get_layer_sequence(3)   # Shared beliefs
operational_layer = network.get_layer_sequence(4)   # Joint operations
```

### Self-Supervised Learning

```python
from src.training import TemporalGNNTrainer

trainer = TemporalGNNTrainer(model)
history = trainer.fit(
    train_networks=train_data,
    val_networks=val_data,
    num_epochs=50
)
# Trains using: temporal link prediction, contrastive learning, node reconstruction
```

### Statistical Analysis

```python
from src.statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(alpha=0.05)
comparison = analyzer.compare_multiple_methods(results_dict)

# Includes: t-tests, Wilcoxon, effect sizes, Bonferroni correction
```

## 🧪 Experiments

### Running Experiments

```bash
# Standard experiment
python scripts/run_experiment.py --output-dir experiments/exp1

# With custom configuration
python scripts/run_experiment.py \
    --num-networks 15 \
    --hidden-dim 256 \
    --initial-nodes 60
```

### Evaluating Models

```bash
python scripts/evaluate_model.py \
    --model-path checkpoints/model.pt \
    --num-test-networks 20
```

### Visualizing Results

```bash
python scripts/visualize_results.py \
    --input-dir experiments/exp1 \
    --output-dir visualizations/ \
    --format png
```

## 📚 Documentation

- [Installation Guide](docs/installation.md) - Detailed setup instructions
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Quick Start](QUICKSTART.md) - Get started in 5 minutes
- [Contributing](CONTRIBUTING.md) - How to contribute
- [Project Structure](PROJECT_STRUCTURE.md) - Code organization

## 🔬 Research Context

### Defensive Research Purpose

This research is conducted for **defensive intelligence and security applications**:
- Protect civilian populations
- Support law enforcement and intelligence agencies
- Advance academic understanding of network security

### Ethical Considerations

✅ Uses **synthetic data** only  
✅ **Responsible disclosure** practices  
✅ Intended for **defensive applications**  
❌ No real terrorist network data  

### Responsible Use Policy

Users must:
- Comply with applicable laws
- Use for legitimate purposes only
- Consider ethical implications
- Not harm individuals or organizations

See [LICENSE](LICENSE) for full terms.

## 📈 Performance

### Disruption Effectiveness

- **+31.8%** improvement over PageRank
- **+21.2%** improvement over GCN
- **+14.5%** improvement over GAT
- **+9.5%** improvement over simple T-GNN

### Component Contributions

1. Temporal Attention: 12.3% of performance
2. Memory Bank: 9.7% of performance
3. Multi-layer Networks: 8.9% of performance
4. Graph Transformer: 7.2% of performance

See [CHANGELOG](CHANGELOG.md) for version history.

## 🤝 Contributing

We welcome contributions! See [Contributing Guidelines](CONTRIBUTING.md) for:

- Code style requirements
- Testing procedures
- Pull request process
- Development setup

```bash
# Setup development environment
pip install -r requirements-dev.txt
pre-commit install

# Run tests
pytest tests/ -v

# Check code style
black src/ --check
flake8 src/
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{lee2025terrorist,
  title={Terrorist Network Disruption using Temporal Graph Neural Networks},
  author={Lee, Yoon-Seop},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License with additional terms for responsible use.

See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- PyTorch Geometric team for the excellent graph neural network library
- Research community for foundational work on temporal GNNs
- Academic advisors for guidance on responsible AI research

## 📧 Contact

- **Author**: Yoon-Seop Lee
- **Email**: iyunseob4@gmail.com
- **Institution**: Big Data Research Institute, Ministry of National Defense
- **Project Page**: [https://Navy10021.github.io/terrorist-network-gnn/](https://Navy10021.github.io/terrorist-network-gnn/)

## 🔗 Links

- [Documentation](https://terrorist-network-gnn.readthedocs.io/)
- [Issue Tracker](https://github.com/Navy10021/terrorist-network-gnn/issues)
- [Discussions](https://github.com/Navy10021/terrorist-network-gnn/discussions)

---

<div align="center">

**Star ⭐ this repo if you find it useful!**

Made with ❤️ for defensive security research

</div>
