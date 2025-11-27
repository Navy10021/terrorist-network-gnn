# System Architecture

## Overview

This document describes the architecture of the Terrorist Network Disruption T-GNN system.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │      Multi-Layer Temporal Network Generator        │     │
│  │       • Physical • Digital • Financial             │     │
│  │       • Ideological • Operational                  │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                    Model Layer                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │         Advanced Temporal GNN                    │    │
│  │  ┌──────────────────────────────────────────┐    │    │
│  │  │  Hierarchical Temporal Pooling           │    │    │
│  │  └──────────────────────────────────────────┘    │    │
│  │  ┌──────────────────────────────────────────┐    │    │
│  │  │  Enhanced Memory Bank (LRU)              │    │    │
│  │  └──────────────────────────────────────────┘    │    │
│  │  ┌──────────────────────────────────────────┐    │    │
│  │  │  Multi-Head Temporal Attention           │    │    │
│  │  └──────────────────────────────────────────┘    │    │
│  │  ┌──────────────────────────────────────────┐    │    │
│  │  │  Graph Transformer Layers                │    │    │
│  │  └──────────────────────────────────────────┘    │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────┐
│                  Analysis Layer                           │
│  ┌────────────┬────────────────┬─────────────────────┐    │
│  │ Q1:        │ Q2:            │ Q3:                 │    │
│  │ Critical   │ Resilience     │ Adversarial         │    │
│  │ Nodes      │ Prediction     │ Robustness          │    │
│  └────────────┴────────────────┴─────────────────────┘    │
└──────────────────────────┬────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                 Evaluation Layer                         │
│              • Statistical Validation                    │
│              • Ablation Studies                          │
│              • Baseline Comparisons                      │
└──────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer

#### Multi-Layer Temporal Network
- **Purpose**: Represent terrorist networks with multiple relationship types
- **Layers**:
  - Physical: Face-to-face meetings, shared locations
  - Digital: Online communications, social media
  - Financial: Money transfers, funding
  - Ideological: Shared beliefs, propaganda exposure
  - Operational: Joint operations, coordinated activities

#### Network Generator
- Generates realistic synthetic networks
- Implements temporal evolution
- Supports data augmentation

### 2. Model Layer

#### Hierarchical Temporal Pooling
- **Purpose**: Capture multi-scale temporal patterns
- **Scales**:
  - Local (1-2 timesteps): Immediate patterns
  - Medium (3-5 timesteps): Short-term trends
  - Global (6+ timesteps): Long-term evolution

#### Enhanced Memory Bank
- **Purpose**: Efficient temporal memory management
- **Features**:
  - LRU eviction policy
  - Configurable capacity
  - Fast retrieval with attention

#### Multi-Head Temporal Attention
- **Purpose**: Learn important temporal relationships
- **Features**:
  - Multiple attention heads (default: 8)
  - Self-attention over time
  - Cross-attention between layers

#### Graph Transformer Layers
- **Purpose**: Process graph structure
- **Features**:
  - Edge-aware attention
  - Positional encoding
  - Layer normalization

### 3. Analysis Layer

#### Q1: Critical Node Detection
- **Input**: Multi-layer temporal network
- **Output**: Ranked list of critical nodes
- **Methods**:
  - 8 centrality metrics
  - Multi-layer aggregation
  - Temporal importance weighting

#### Q2: Resilience Prediction
- **Input**: Network state after disruption
- **Output**: Predicted recovery metrics
- **Methods**:
  - Edge formation prediction
  - Recruitment probability
  - Resilience score calculation

#### Q3: Adversarial Robustness
- **Input**: Network and disruption strategy
- **Output**: Recovery rate and time
- **Methods**:
  - 4 adaptation strategies
  - Recovery simulation
  - Effectiveness evaluation

### 4. Evaluation Layer

#### Statistical Validation
- Paired t-tests
- Wilcoxon signed-rank tests
- Effect size calculation (Cohen's d)
- Bonferroni correction

#### Ablation Studies
- Component importance ranking
- Performance contribution analysis
- Sensitivity analysis

#### Baseline Comparisons
- 12 comparison methods
- Fair evaluation protocol
- Multiple metrics

## Data Flow

### Training Phase

```
1. Generate Networks
   └→ TerroristNetworkGenerator
       └→ Multi-layer temporal networks

2. Augment Data
   └→ NetworkAugmenter
       └→ Edge drop, feature mask, noise

3. Train Model
   └→ EnhancedTemporalGNNTrainer
       ├→ Temporal autoencoder loss
       ├→ Graph reconstruction loss
       ├→ Contrastive learning
       └→ Hard negative sampling

4. Validate
   └→ Validation networks
       └→ Monitor performance
```

### Inference Phase

```
1. Input Network
   └→ Multi-layer temporal network

2. Model Forward
   └→ AdvancedTemporalGNN
       ├→ Temporal encoding
       ├→ Graph convolution
       ├→ Attention mechanisms
       └→ Pooling

3. Critical Node Detection
   └→ EnhancedCriticalNodeDetector
       ├→ Compute centralities
       ├→ Aggregate scores
       └→ Rank nodes

4. Disruption Analysis
   └→ NetworkDisruptionOptimizer
       ├→ Q1: Select critical nodes
       ├→ Q2: Predict resilience
       └→ Q3: Test adversarial robustness
```

## Key Design Decisions

### 1. Multi-Layer Representation
**Decision**: Use 5 separate layers for different relationship types

**Rationale**:
- Real terrorist networks have distinct communication channels
- Different layers have different importance for disruption
- Enables layer-specific analysis

### 2. Hierarchical Temporal Pooling
**Decision**: Use 3 temporal scales (local, medium, global)

**Rationale**:
- Captures both immediate and long-term patterns
- More expressive than single-scale pooling
- Computationally efficient

### 3. LRU Memory Bank
**Decision**: Use LRU eviction policy for memory management

**Rationale**:
- Recent information is more relevant
- Bounded memory prevents overflow
- Fast O(1) operations

### 4. Self-Supervised Learning
**Decision**: Use multiple loss functions for training

**Rationale**:
- No labeled data available for terrorist networks
- Multiple objectives improve generalization
- Enables unsupervised learning

## Performance Considerations

### Memory Optimization
- Gradient checkpointing for large networks
- Mixed precision training (FP16)
- Batch processing for scalability

### Computational Efficiency
- Sparse graph operations
- Efficient attention mechanisms
- Parallelizable components

### Scalability
- Supports networks up to 100 nodes
- Handles 20+ timesteps
- GPU acceleration

## Extension Points

### Adding New Layers
```python
# Add new relationship type
new_layer = NetworkLayer(
    name="cyber",
    edge_index=cyber_edges,
    edge_attr=cyber_features,
    node_features=node_features
)
```

### Custom Centrality Metrics
```python
# Add new centrality measure
def custom_centrality(edge_index, num_nodes):
    # Your implementation
    return scores
```

### New Disruption Strategies
```python
# Add new node selection strategy
def custom_strategy(scores, k):
    # Your selection logic
    return selected_nodes
```

## References

- PyTorch Geometric Documentation
- Graph Neural Network literature
- Temporal network analysis methods
- Counter-terrorism research
