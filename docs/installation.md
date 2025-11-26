# Installation Guide

Complete installation instructions for the Terrorist Network GNN project.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Dependency Installation](#dependency-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Optional Components](#optional-components)

## System Requirements

### Minimum Requirements

- **OS**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Disk Space**: 5GB free space

### Recommended Requirements

- **GPU**: NVIDIA GPU with CUDA 11.8+ (for acceleration)
- **RAM**: 16GB or more
- **Python**: 3.10
- **Disk Space**: 10GB free space

### Software Dependencies

- Python 3.8+
- pip (latest version)
- Git
- CUDA Toolkit 11.8+ (optional, for GPU support)

## Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/terrorist-network-gnn.git
cd terrorist-network-gnn

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Method 2: With GPU Support

```bash
# Clone the repository
git clone https://github.com/yourusername/terrorist-network-gnn.git
cd terrorist-network-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install remaining dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Method 3: Google Colab

No installation needed! Just open the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/terrorist-network-gnn/blob/main/notebooks/terrorist_network_gnn_demo.ipynb)

### Method 4: Docker (Coming Soon)

```bash
# Pull Docker image
docker pull yourusername/terrorist-network-gnn:latest

# Run container
docker run -it --gpus all yourusername/terrorist-network-gnn:latest
```

## Dependency Installation

### Core Dependencies

Install core packages:

```bash
pip install torch>=2.0.0
pip install torch-geometric>=2.3.0
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install networkx>=3.1
pip install tqdm>=4.65.0
```

### Development Dependencies

For development work:

```bash
pip install -r requirements-dev.txt
```

This includes:
- pytest (testing)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)
- sphinx (documentation)

### Optional Dependencies

#### Jupyter Support

```bash
pip install jupyter ipykernel ipywidgets
```

#### Advanced Visualization

```bash
pip install plotly dash
```

## Verification

### Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check PyTorch Geometric
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

# Run package import test
python -c "from src.advanced_tgnn import AdvancedTemporalGNN; print('✓ Package installed successfully')"
```

### Run Test Suite

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_tgnn.py -v
```

### Quick Functionality Test

```bash
# Run quick experiment (5-10 minutes)
python scripts/run_experiment.py \
    --num-networks 3 \
    --num-timesteps 10 \
    --output-dir experiments/installation_test
```

## Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Reduce network size in configuration
config = NetworkConfig(
    initial_nodes=30,  # Reduced from 50
    max_nodes=50       # Reduced from 80
)
```

Or use CPU:
```bash
python scripts/run_experiment.py --device cpu
```

#### Issue 2: PyTorch Geometric Installation Fails

**Error**: `No matching distribution found for torch-geometric`

**Solution**:
```bash
# Install PyTorch first
pip install torch

# Then install PyG with specific version
pip install torch-geometric==2.3.0
```

#### Issue 3: Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

#### Issue 4: Permission Denied (Linux/Mac)

**Error**: `Permission denied` when creating directories

**Solution**:
```bash
# Give execution permission to scripts
chmod +x scripts/*.py

# Or run with python explicitly
python scripts/run_experiment.py
```

#### Issue 5: SSL Certificate Error

**Error**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution**:
```bash
# Update certificates (macOS)
/Applications/Python\ 3.x/Install\ Certificates.command

# Or install with trusted host
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Platform-Specific Issues

#### Windows

**Long Path Issues**:
```bash
# Enable long paths in Windows
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

**Visual C++ Build Tools Required**:
Download and install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

#### macOS

**SSL Certificate Issues**:
```bash
# Install certificates
/Applications/Python\ 3.x/Install\ Certificates.command
```

**Xcode Command Line Tools**:
```bash
xcode-select --install
```

#### Linux

**CUDA Driver Issues**:
```bash
# Check CUDA driver version
nvidia-smi

# Install CUDA toolkit (Ubuntu)
sudo apt-get install nvidia-cuda-toolkit
```

### Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/yourusername/terrorist-network-gnn/issues)
2. **Search discussions**: [GitHub Discussions](https://github.com/yourusername/terrorist-network-gnn/discussions)
3. **Create new issue**: Include:
   - OS and Python version
   - Complete error message
   - Steps to reproduce
   - What you've tried

## Optional Components

### Enable GPU Acceleration

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Setup Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Build Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs/
make html

# Open documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

### Setup Jupyter Kernel

```bash
# Create Jupyter kernel for this environment
python -m ipykernel install --user --name=terrorist-network-gnn --display-name="Terrorist Network GNN"

# Launch Jupyter
jupyter notebook notebooks/
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv/

# Remove package (if installed with pip install)
pip uninstall terrorist-network-gnn

# Remove repository
cd ..
rm -rf terrorist-network-gnn/
```

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](../QUICKSTART.md)
2. Try the [Tutorial Notebook](../notebooks/terrorist_network_gnn_demo.ipynb)
3. Review the [API Reference](api_reference.md)
4. Run your first experiment!

---

**Need help?** Open an issue or start a discussion on GitHub!
