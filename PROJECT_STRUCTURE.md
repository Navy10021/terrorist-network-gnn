# Project Structure

## Complete Directory Tree

```
terrorist-network-tgnn/
│
├── .github/                              # GitHub configuration
│   └── workflows/
│       └── python-tests.yml              # CI/CD pipeline
│
├── src/                                  # Source code (207KB total)
│   ├── __init__.py                       # Package initialization
│   ├── advanced_tgnn.py                  # Core T-GNN architecture (25KB)
│   ├── terrorist_network_disruption.py   # Disruption algorithms (30KB)
│   ├── terrorist_network_dataset.py      # Network generation (37KB)
│   ├── training.py                       # Training loops (23KB)
│   ├── baselines.py                      # Comparison methods (18KB)
│   ├── statistical_analysis.py           # Statistical tests (18KB)
│   ├── ablation_study.py                 # Component analysis (20KB)
│   └── main_experiment.py                # Complete pipeline (36KB)
│
├── examples/                             # Usage examples
│   └── terrorist_network_gnn_demo.ipynb  # Interactive demo (2.0MB)
│
├── tests/                                # Unit tests
│   ├── __init__.py
│   └── test_tgnn.py                      # T-GNN tests
│
├── docs/                                 # Documentation
│   ├── architecture.md                   # System architecture
│   └── quickstart.md                     # Quick start guide
│
├── data/                                 # Data directory
│   └── .gitkeep                          # (Synthetic data only)
│
├── results/                              # Experiment results
│   └── .gitkeep                          # (Generated outputs)
│
├── .gitignore                            # Git ignore rules
├── CHANGELOG.md                          # Version history
├── CONTRIBUTING.md                       # Contribution guidelines
├── LICENSE                               # MIT License
├── README.md                             # Main documentation
├── requirements.txt                      # Python dependencies
└── setup.py                              # Package setup

```

## File Descriptions

### Root Files (9 files)

| File | Size | Description |
|------|------|-------------|
| `README.md` | 40KB | Main project documentation with overview, installation, usage |
| `LICENSE` | 1KB | MIT License |
| `CONTRIBUTING.md` | 12KB | Guidelines for contributors |
| `CHANGELOG.md` | 5KB | Version history and release notes |
| `requirements.txt` | 1KB | Python package dependencies |
| `setup.py` | 1KB | Package installation configuration |
| `.gitignore` | 1KB | Git ignore patterns |

### Source Code (`src/` - 8 files, 207KB)

| File | Lines | Size | Description |
|------|-------|------|-------------|
| `advanced_tgnn.py` | 800+ | 25KB | **Core T-GNN Architecture**<br>- HierarchicalTemporalPooling<br>- EnhancedTemporalMemoryBank<br>- AdaptiveTimeEncoding<br>- AdvancedTemporalGNN |
| `terrorist_network_disruption.py` | 1000+ | 30KB | **Disruption Analysis**<br>- EnhancedCriticalNodeDetector<br>- MultiLayerTemporalGNN<br>- TemporalResiliencePredictor<br>- AdversarialNetworkAttack |
| `terrorist_network_dataset.py` | 1200+ | 37KB | **Data Generation**<br>- TerroristNetworkGenerator<br>- NetworkAugmenter<br>- DisruptionEvaluator<br>- Multi-layer network synthesis |
| `training.py` | 800+ | 23KB | **Model Training**<br>- EnhancedTemporalGNNTrainer<br>- TemporalAutoencoderLoss<br>- GraphReconstructionLoss<br>- Self-supervised learning |
| `baselines.py` | 600+ | 18KB | **Baseline Methods**<br>- Traditional centrality metrics<br>- Static GNN models<br>- Temporal GNN baselines<br>- 12 comparison methods |
| `statistical_analysis.py` | 600+ | 18KB | **Statistical Validation**<br>- Hypothesis testing<br>- Effect size calculation<br>- Multiple comparison correction<br>- Result visualization |
| `ablation_study.py` | 600+ | 20KB | **Ablation Analysis**<br>- Component removal experiments<br>- Importance ranking<br>- Performance contribution<br>- Sensitivity analysis |
| `main_experiment.py` | 1200+ | 36KB | **Complete Pipeline**<br>- 9-phase experimental workflow<br>- Network generation<br>- Model training<br>- Evaluation and visualization |
| `__init__.py` | 100+ | 3KB | Package initialization and exports |

### Tests (`tests/` - 2 files)

| File | Description |
|------|-------------|
| `test_tgnn.py` | Unit tests for core T-GNN components |
| `__init__.py` | Test package initialization |

### Documentation (`docs/` - 2+ files)

| File | Size | Description |
|------|------|-------------|
| `architecture.md` | 8KB | Detailed system architecture |
| `quickstart.md` | 5KB | Quick start guide |

### Examples (`examples/` - 1 file)

| File | Size | Description |
|------|------|-------------|
| `terrorist_network_gnn_demo.ipynb` | 2.0MB | Interactive Jupyter notebook demo |

### CI/CD (`.github/workflows/` - 1 file)

| File | Description |
|------|-------------|
| `python-tests.yml` | GitHub Actions workflow for automated testing |

## Key Statistics

### Code Metrics

```
Total Source Lines:    ~7,000+ lines
Total Code Size:       207KB
Number of Classes:     30+
Number of Functions:   100+
Test Coverage:         Target >80%
```

### Language Distribution

```
Python:     95%
Markdown:   3%
YAML:       1%
Other:      1%
```

### Module Dependencies

```
torch                (Core framework)
torch-geometric      (GNN operations)
networkx            (Graph algorithms)
numpy               (Numerical computing)
scipy               (Statistical functions)
matplotlib          (Visualization)
seaborn             (Statistical plots)
pandas              (Data manipulation)
tqdm                (Progress bars)
```

## Development Workflow

```
1. Clone Repository
   └→ git clone

2. Setup Environment
   └→ Virtual environment
   └→ Install dependencies

3. Development
   ├→ Edit source files in src/
   ├→ Add tests in tests/
   └→ Update documentation

4. Testing
   ├→ pytest tests/
   ├→ black src/
   └→ flake8 src/

5. Commit & Push
   └→ CI/CD pipeline runs automatically

6. Release
   └→ Update version
   └→ Create tag
   └→ GitHub release
```

## Module Import Structure

```python
# Top-level imports
from src import (
    AdvancedTemporalGNN,              # Core model
    MultiLayerTemporalGNN,            # Multi-layer processing
    TerroristNetworkGenerator,        # Data generation
    EnhancedTemporalGNNTrainer,       # Training
    EnhancedExperiment,               # Full pipeline
)

# Submodule imports
from src.advanced_tgnn import (
    HierarchicalTemporalPooling,
    EnhancedTemporalMemoryBank,
)

from src.terrorist_network_disruption import (
    EnhancedCriticalNodeDetector,
    TemporalResiliencePredictor,
)
```

## Data Flow Between Modules

```
terrorist_network_dataset.py
    ↓ (generates networks)
training.py
    ↓ (trains model)
advanced_tgnn.py
    ↓ (produces embeddings)
terrorist_network_disruption.py
    ↓ (analyzes disruption)
statistical_analysis.py
    ↓ (validates results)
main_experiment.py
    ↓ (orchestrates everything)
results/ (outputs)
```

## Size Summary

| Category | Size |
|----------|------|
| **Source Code** | 207KB |
| **Tests** | 10KB |
| **Documentation** | 80KB |
| **Examples** | 2.0MB |
| **Configuration** | 5KB |
| **Total** | ~2.3MB |

## Platform Support

| Platform | Status |
|----------|--------|
| Linux | ✅ Fully supported |
| macOS | ✅ Fully supported |
| Windows | ✅ Supported (tested on Windows 10/11) |
| Google Colab | ✅ Recommended for beginners |

## Python Version Support

| Version | Status |
|---------|--------|
| Python 3.8 | ✅ Supported |
| Python 3.9 | ✅ Supported |
| Python 3.10 | ✅ Supported |
| Python 3.11 | ✅ Supported |
| Python 3.12 | 🔄 Testing |

---

**Last Updated**: November 30, 2025  
**Project Version**: v2.0.0
