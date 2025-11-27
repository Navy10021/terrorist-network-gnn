# Contributing to Terrorist Network T-GNN

First off, thank you for considering contributing to this project! 🎉

The following is a set of guidelines for contributing to the Terrorist Network T-GNN project. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Ethical Considerations](#ethical-considerations)

---

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/yourusername/terrorist-network-tgnn/issues) to avoid duplicates.

When you create a bug report, please include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected behavior**
- **Actual behavior**
- **Screenshots** if applicable
- **Environment details** (OS, Python version, PyTorch version, etc.)
- **Code snippet** that reproduces the issue

Example:

```markdown
**Bug Description**
Model training fails with CUDA out of memory error

**To Reproduce**
1. Run `python src/main_experiment.py --mode full`
2. See error after epoch 3

**Expected Behavior**
Training should complete without memory errors

**Environment**
- OS: Ubuntu 20.04
- Python: 3.9.7
- PyTorch: 2.0.1
- CUDA: 11.8
- GPU: NVIDIA RTX 3080 (10GB)
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Clear title and description**
- **Use case** - why this enhancement would be useful
- **Proposed solution** if you have one
- **Alternative solutions** you've considered
- **Additional context** or screenshots

### Pull Requests

1. **Fork the repository** and create your branch from `develop`
   ```bash
   git checkout -b feature/amazing-feature develop
   ```

2. **Make your changes**
   - Follow the [Style Guidelines](#style-guidelines)
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for code refactoring

4. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

5. **Open a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Include screenshots/examples if applicable

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/terrorist-network-tgnn.git
cd terrorist-network-tgnn
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## Style Guidelines

### Python Style

We follow [PEP 8](https://pep8.org/) with the following tools:

#### Black (Code Formatter)

```bash
# Format all code
black src/

# Check formatting without making changes
black --check src/
```

#### Flake8 (Linter)

```bash
# Lint code
flake8 src/ --max-line-length=127 --ignore=E203,W503
```

#### MyPy (Type Checker)

```bash
# Type check
mypy src/ --ignore-missing-imports
```

### Code Structure

- **Function docstrings**: Use Google style
  ```python
  def function_name(param1: int, param2: str) -> bool:
      """
      Brief description.
      
      Args:
          param1: Description of param1
          param2: Description of param2
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: When something goes wrong
      """
      pass
  ```

- **Class docstrings**: Describe the class and its attributes
  ```python
  class ClassName:
      """
      Brief description of the class.
      
      Attributes:
          attr1: Description of attr1
          attr2: Description of attr2
      """
      pass
  ```

- **Type hints**: Use type hints for all function signatures
  ```python
  def process_data(data: torch.Tensor, labels: List[int]) -> Dict[str, float]:
      pass
  ```

### Naming Conventions

- **Classes**: PascalCase (`MyClassName`)
- **Functions/Methods**: snake_case (`my_function_name`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_VALUE`)
- **Private methods**: Leading underscore (`_private_method`)

---

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`

Example test:

```python
import pytest
import torch
from src.advanced_tgnn import AdvancedTemporalGNN

def test_model_forward_pass():
    """Test that model forward pass works correctly"""
    model = AdvancedTemporalGNN(
        num_node_features=64,
        num_edge_features=32,
        hidden_dim=128
    )
    
    # Create dummy input
    node_features = torch.randn(10, 64)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 32)
    
    # Forward pass
    output = model(node_features, edge_index, edge_attr)
    
    # Assertions
    assert output.shape == (10, 128)
    assert not torch.isnan(output).any()
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_tgnn.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_tgnn.py::test_model_forward_pass -v
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Critical functions require edge case tests

---

## Ethical Considerations

### Research Ethics

When contributing to this project, please ensure:

1. **Defensive Purpose**: All contributions should support counter-terrorism defense, not offense

2. **Synthetic Data Only**: Never commit or process real terrorist network data

3. **Responsible Disclosure**: Discuss potential security implications of new features

4. **Documentation**: Document any dual-use concerns in code comments

5. **Privacy**: Do not include personally identifiable information

### Code Review Criteria

Pull requests will be reviewed for:

- ✅ Technical correctness
- ✅ Code quality and style
- ✅ Test coverage
- ✅ Documentation completeness
- ✅ Ethical implications

---

## Documentation

### Adding Documentation

- Update README.md for significant features
- Add docstrings to all public functions/classes
- Create markdown files in `docs/` for detailed documentation

### Documentation Structure

```
docs/
├── architecture.md      # System architecture
├── api_reference.md     # API documentation
├── research_questions.md # Problem formulation
├── baselines.md         # Baseline methods
└── metrics.md           # Evaluation metrics
```

---

## Release Process

1. Update version in `setup.py` and `src/__init__.py`
2. Update `CHANGELOG.md`
3. Create a new tag: `git tag -a v1.x.x -m "Version 1.x.x"`
4. Push tag: `git push origin v1.x.x`
5. Create GitHub release with release notes

---

## Questions?

If you have questions or need help, please:

1. Check existing [GitHub Issues](https://github.com/yourusername/terrorist-network-tgnn/issues)
2. Start a [GitHub Discussion](https://github.com/yourusername/terrorist-network-tgnn/discussions)
3. Contact the maintainer: iyunseob4@gmail.com

---

## Recognition

Contributors will be recognized in:
- README.md acknowledgments section
- GitHub contributors page
- Academic paper acknowledgments (for significant contributions)

Thank you for your contributions! 🙏
