# Project Structure

This document provides a detailed overview of the project structure and organization.

## Directory Layout

```
terrorist-network-gnn/
│
├── README.md                          # Main project documentation
├── LICENSE                            # MIT License
├── CONTRIBUTING.md                    # Contribution guidelines
├── requirements.txt                   # Python dependencies
├── requirements-dev.txt               # Development dependencies
├── setup.py                          # Package installation
├── .gitignore                        # Git ignore rules
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── advanced_tgnn.py             # Advanced T-GNN architecture
│   ├── terrorist_network_disruption.py  # Network disruption algorithms
│   ├── terrorist_network_dataset.py  # Dataset generation
│   ├── training.py                   # Self-supervised training
│   ├── baselines.py                  # Baseline methods
│   ├── statistical_analysis.py       # Statistical testing
│   └── ablation_study.py            # Component analysis
│
├── scripts/                          # Executable scripts
│   ├── __init__.py
│   ├── run_experiment.py            # Main experiment runner
│   ├── evaluate_model.py            # Model evaluation
│   └── visualize_results.py         # Result visualization
│
├── notebooks/                        # Jupyter notebooks
│   ├── terrorist_network_gnn_demo.ipynb  # Interactive demo
│   ├── exploratory_analysis.ipynb   # Data exploration
│   └── results_visualization.ipynb  # Result visualization
│
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_tgnn.py                # T-GNN tests
│   ├── test_dataset.py             # Dataset tests
│   ├── test_disruption.py          # Disruption algorithm tests
│   ├── test_training.py            # Training tests
│   └── test_baselines.py           # Baseline tests
│
├── docs/                           # Documentation
│   ├── architecture.md             # System architecture
│   ├── methodology.md              # Research methodology
│   ├── api_reference.md            # API documentation
│   ├── installation.md             # Installation guide
│   └── tutorial.md                 # Usage tutorial
│
├── experiments/                    # Experiment outputs
│   ├── .gitkeep
│   └── README.md                  # Experiment documentation
│
├── assets/                        # Images and resources
│   ├── architecture.png           # Architecture diagram
│   ├── logo.png                   # Project logo
│   ├── results/                   # Result images
│   └── demo.gif                   # Demo animation
│
└── .github/                       # GitHub configuration
    ├── workflows/                 # CI/CD workflows
    │   ├── tests.yml             # Automated testing
    │   └── docs.yml              # Documentation build
    ├── ISSUE_TEMPLATE/           # Issue templates
    └── PULL_REQUEST_TEMPLATE.md  # PR template
```

## Module Descriptions

### Core Modules (`src/`)

#### `advanced_tgnn.py`
Advanced Temporal Graph Neural Network implementation with:
- Multi-head temporal attention
- Memory-augmented networks
- Graph transformer layers
- Adaptive time encoding
- Causal temporal convolution

**Key Classes:**
- `AdvancedTemporalGNN`: Main T-GNN architecture
- `AdaptiveTimeEncoding`: Time representation learning
- `TemporalMemoryBank`: Memory module
- `MultiHeadTemporalAttention`: Attention mechanism
- `GraphTransformerLayer`: Graph convolution layer

#### `terrorist_network_disruption.py`
Network disruption algorithms and analysis:
- Critical node detection
- Multi-layer network support
- Temporal resilience prediction
- Adversarial robustness analysis

**Key Classes:**
- `MultiLayerTemporalGNN`: Multi-layer network processor
- `CriticalNodeDetector`: Node importance computation
- `NetworkDisruptionOptimizer`: Optimal disruption strategy
- `TemporalResiliencePredictor`: Resilience forecasting
- `AdversarialNetworkAttack`: Adaptation simulation

#### `terrorist_network_dataset.py`
Synthetic dataset generation:
- Realistic network structures
- Temporal evolution
- Multi-layer networks
- Evaluation protocols

**Key Classes:**
- `TerroristNetworkGenerator`: Network creation
- `NetworkConfig`: Configuration dataclass
- `DisruptionEvaluator`: Performance metrics

#### `training.py`
Self-supervised learning framework:
- Temporal link prediction
- Contrastive learning
- Node reconstruction
- Training utilities

**Key Classes:**
- `TemporalGNNTrainer`: Training orchestration
- `TemporalLinkPredictionLoss`: Link prediction loss
- `ContrastiveLoss`: Temporal consistency loss
- `NodeReconstructionLoss`: Feature reconstruction

#### `baselines.py`
Baseline method implementations:
- Static GNN methods (GCN, GAT, GraphSAGE)
- Centrality measures
- Graph partitioning
- Simple temporal GNN

**Key Classes:**
- `StaticGCN`, `StaticGAT`, `StaticGraphSAGE`: Static GNNs
- `CentralityBaseline`: Traditional metrics
- `SimpleTemporalGNN`: Basic temporal model
- `BaselineEvaluator`: Unified evaluation

#### `statistical_analysis.py`
Statistical significance testing:
- Hypothesis tests
- Effect size calculations
- Multiple comparison corrections
- Result visualization

**Key Classes:**
- `StatisticalAnalyzer`: Statistical tests
- `ResultVisualizer`: Result plotting

#### `ablation_study.py`
Component importance analysis:
- Systematic component removal
- Performance impact measurement
- Importance ranking

**Key Classes:**
- `AblationStudy`: Ablation orchestration

### Scripts (`scripts/`)

#### `run_experiment.py`
Main experiment runner with command-line interface for:
- Network generation
- Model training
- Baseline comparison
- Ablation studies

#### `evaluate_model.py`
Model evaluation utilities for:
- Trained model assessment
- Performance metrics
- Comparison with baselines

#### `visualize_results.py`
Result visualization including:
- Training curves
- Performance comparisons
- Statistical plots
- Network visualizations

### Tests (`tests/`)

Comprehensive test suite covering:
- Unit tests for all modules
- Integration tests
- Performance benchmarks
- Edge case handling

### Documentation (`docs/`)

#### `architecture.md`
System architecture documentation:
- Component relationships
- Data flow
- Design decisions

#### `methodology.md`
Research methodology:
- Problem formulation
- Approach description
- Experimental design

#### `api_reference.md`
API documentation:
- Function signatures
- Parameter descriptions
- Usage examples

## File Naming Conventions

- **Python modules**: `lowercase_with_underscores.py`
- **Classes**: `PascalCase`
- **Functions**: `lowercase_with_underscores()`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- **Private methods**: `_leading_underscore()`

## Code Organization Principles

1. **Modularity**: Each module has a single, well-defined purpose
2. **Separation of Concerns**: Clear boundaries between components
3. **Reusability**: Common functionality in shared modules
4. **Testability**: All components designed for easy testing
5. **Documentation**: Comprehensive docstrings and comments

## Import Structure

```python
# Standard library imports
import os
import sys
from typing import List, Dict, Tuple

# Third-party imports
import torch
import numpy as np
import networkx as nx

# Local imports
from src.advanced_tgnn import AdvancedTemporalGNN
from src.terrorist_network_disruption import CriticalNodeDetector
```

## Configuration Management

Configuration is managed through:
- `NetworkConfig` dataclass for network parameters
- Command-line arguments for experiment settings
- Model configuration dictionaries
- Environment variables for sensitive settings

## Data Flow

1. **Network Generation**: `TerroristNetworkGenerator` → `MultiLayerTemporalNetwork`
2. **Model Training**: Network data → `TemporalGNNTrainer` → Trained model
3. **Evaluation**: Trained model → `CriticalNodeDetector` → Results
4. **Analysis**: Results → `StatisticalAnalyzer` → Insights

## Adding New Features

To add a new feature:

1. Create new module in `src/` if needed
2. Add corresponding tests in `tests/`
3. Update documentation in `docs/`
4. Add usage examples in `notebooks/`
5. Update README.md if user-facing

## Best Practices

- Keep modules focused and cohesive
- Write comprehensive docstrings
- Add type hints to all functions
- Create tests for new functionality
- Update documentation with changes
- Follow existing code style

---
