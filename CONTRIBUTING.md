# Contributing to Terrorist Network GNN

Thank you for your interest in contributing to our research project! This document provides guidelines for contributing to the codebase.

## Code of Conduct

### Responsible Research

This project focuses on defensive security applications. All contributors must:

- Use the code only for legitimate research and security purposes
- Follow ethical guidelines for AI research
- Respect privacy and data protection principles
- Consider potential dual-use implications
- Comply with all applicable laws and regulations

### Community Standards

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive in communications
- Welcome newcomers and help them learn
- Focus on what is best for the community and research
- Show empathy towards other community members

## How to Contribute

### Reporting Issues

**Bug Reports**

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU/CPU)
- Error messages and stack traces

**Feature Requests**

For new features, please describe:
- The problem you're trying to solve
- Your proposed solution
- Alternative solutions considered
- Potential impact on existing functionality

### Development Process

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/terrorist-network-gnn.git
   cd terrorist-network-gnn
   git remote add upstream https://github.com/original/terrorist-network-gnn.git
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Make Your Changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

5. **Test Your Changes**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run with coverage
   pytest --cov=src tests/
   
   # Check code style
   flake8 src/
   black src/ --check
   
   # Type checking
   mypy src/
   ```

6. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: Add feature description"
   ```

   Follow conventional commit format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes (formatting)
   - `refactor:` Code refactoring
   - `test:` Adding/updating tests
   - `chore:` Maintenance tasks

7. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   
   Then create a PR on GitHub with:
   - Clear title and description
   - Reference to related issues
   - Screenshots if applicable
   - Notes on testing performed

### Code Style

**Python Code Style**

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use Black for formatting
- Use type hints where appropriate
- Write docstrings for all public functions

Example:
```python
def detect_critical_nodes(
    self,
    edge_index: torch.Tensor,
    num_nodes: int,
    embeddings: Optional[torch.Tensor] = None,
    top_k: int = 10
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Detect critical nodes using ensemble of metrics.
    
    Args:
        edge_index: Graph edge indices [2, num_edges]
        num_nodes: Total number of nodes
        embeddings: Node embeddings [num_nodes, hidden_dim]
        top_k: Number of critical nodes to return
        
    Returns:
        critical_nodes: Indices of top-k critical nodes
        importance_scores: Dict of scores for each metric
    """
    # Implementation
    pass
```

**Documentation Style**

- Use Google-style docstrings
- Include type hints in function signatures
- Provide usage examples for complex functions
- Keep documentation up-to-date with code changes

### Testing

**Writing Tests**

- Place tests in `tests/` directory
- Name test files `test_<module>.py`
- Use descriptive test names
- Test edge cases and error conditions
- Aim for >80% code coverage

Example:
```python
import pytest
import torch
from src.terrorist_network_disruption import CriticalNodeDetector

def test_critical_node_detection_basic():
    """Test basic critical node detection"""
    detector = CriticalNodeDetector()
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    num_nodes = 3
    
    critical_nodes, scores = detector.detect_critical_nodes(
        edge_index, num_nodes, top_k=2
    )
    
    assert critical_nodes.size(0) == 2
    assert len(scores) > 0

def test_critical_node_detection_empty_graph():
    """Test detection on empty graph"""
    detector = CriticalNodeDetector()
    edge_index = torch.empty((2, 0), dtype=torch.long)
    num_nodes = 5
    
    critical_nodes, scores = detector.detect_critical_nodes(
        edge_index, num_nodes, top_k=2
    )
    
    assert critical_nodes.size(0) == 2
```

### Documentation

**README Updates**

Update README.md when:
- Adding new features
- Changing installation process
- Updating requirements
- Adding new examples

**API Documentation**

- Document all public functions and classes
- Include parameter descriptions
- Provide return value information
- Add usage examples

**Research Documentation**

- Document research methodology in `docs/methodology.md`
- Update architecture documentation for structural changes
- Maintain experiment logs

## Pull Request Process

1. **Before Submitting**
   - Ensure all tests pass
   - Update documentation
   - Add entry to CHANGELOG.md
   - Rebase on latest main branch

2. **PR Review**
   - At least one maintainer approval required
   - All CI checks must pass
   - Address reviewer feedback promptly
   - Keep PR focused on single feature/fix

3. **After Approval**
   - Maintainer will merge the PR
   - Delete feature branch
   - Pull latest changes to local main

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- CUDA 11.8+ (for GPU support)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/terrorist-network-gnn.git
cd terrorist-network-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_tgnn.py

# Run with coverage report
pytest --cov=src --cov-report=html

# Run with verbose output
pytest -v

# Run only fast tests
pytest -m "not slow"
```

### Building Documentation

```bash
cd docs/
make html

# Open in browser
open _build/html/index.html
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for general questions
- Email maintainers for sensitive matters

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Academic papers (for significant contributions)
- Release notes

Thank you for contributing to advancing network security research! 🙏
