# System Architecture

This document describes the architecture and design of the Terrorist Network GNN system.

## Table of Contents

1. [Overview](#overview)
2. [System Components](#system-components)
3. [Data Flow](#data-flow)
4. [Model Architecture](#model-architecture)
5. [Design Patterns](#design-patterns)
6. [Scalability Considerations](#scalability-considerations)

---

## Overview

The Terrorist Network GNN system is designed as a modular, extensible framework for analyzing temporal multi-layer networks using Graph Neural Networks. The architecture follows separation of concerns principles, with clear boundaries between data generation, model training, and analysis components.

### Key Design Goals

- **Modularity**: Independent, reusable components
- **Extensibility**: Easy to add new models and methods
- **Reproducibility**: Deterministic behavior with seed control
- **Performance**: Efficient processing of temporal graph data
- **Maintainability**: Clean code structure and comprehensive tests

---

## System Components

### 1. Data Layer

#### `terrorist_network_dataset.py`

**Purpose**: Synthetic network generation and data management

**Key Classes**:
- `NetworkConfig`: Configuration dataclass for network parameters
- `TerroristNetworkGenerator`: Creates realistic temporal multi-layer networks
- `DisruptionEvaluator`: Evaluates disruption effectiveness

**Responsibilities**:
- Generate synthetic networks with temporal evolution
- Manage multi-layer network structures (5 layers)
- Provide evaluation metrics

```
┌─────────────────────────────────┐
│     NetworkConfig               │
│  - initial_nodes                │
│  - max_nodes                    │
│  - recruitment_rate             │
│  - dropout_rate                 │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  TerroristNetworkGenerator      │
│  + generate_temporal_network()  │
│  + add_nodes()                  │
│  + evolve_network()             │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│   MultiLayerTemporalNetwork     │
│  - Physical Layer               │
│  - Digital Layer                │
│  - Financial Layer              │
│  - Ideological Layer            │
│  - Operational Layer            │
└─────────────────────────────────┘
```

---

### 2. Model Layer

#### `advanced_tgnn.py`

**Purpose**: Advanced temporal GNN architecture

**Key Components**:

1. **AdaptiveTimeEncoding**
   - Learns continuous time representations
   - Frequency and phase parameters
   - Sinusoidal encoding

2. **TemporalMemoryBank**
   - Stores temporal patterns
   - Attention-based retrieval
   - Dynamic memory updates

3. **MultiHeadTemporalAttention**
   - Attends to relevant timesteps
   - Multi-head mechanism
   - Query-key-value architecture

4. **GraphTransformerLayer**
   - Graph structure processing
   - Transformer-based convolution
   - Edge feature integration

5. **AdvancedTemporalGNN**
   - Main model class
   - Integrates all components
   - Produces node embeddings

```
Input Sequence
     │
     ▼
┌─────────────────────┐
│  Time Encoding      │
│  (Adaptive)         │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Memory Bank        │
│  (Attention-based)  │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Temporal Attention │
│  (Multi-head)       │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Graph Transformer  │
│  (Spatial Conv)     │
└─────────────────────┘
     │
     ▼
Node Embeddings
```

---

### 3. Analysis Layer

#### `terrorist_network_disruption.py`

**Purpose**: Network disruption analysis and critical node detection

**Key Classes**:

1. **MultiLayerTemporalGNN**
   - Processes multi-layer networks
   - Layer aggregation
   - Temporal integration

2. **CriticalNodeDetector**
   - Identifies critical nodes
   - Combines centrality and GNN-based importance
   - Ensemble scoring

3. **NetworkDisruptionOptimizer**
   - Finds optimal disruption strategies
   - Budget-constrained optimization
   - Sequential removal planning

4. **TemporalResiliencePredictor**
   - Predicts network reconstruction
   - LSTM-based forecasting
   - Resilience scoring

5. **AdversarialNetworkAttack**
   - Simulates adaptive responses
   - Network evolution modeling
   - Robustness testing

```
Network State
     │
     ▼
┌──────────────────────────┐
│  Multi-layer GNN         │
│  (Process all layers)    │
└──────────────────────────┘
     │
     ▼
┌──────────────────────────┐
│  Critical Node Detector  │
│  (Importance scoring)    │
└──────────────────────────┘
     │
     ├─────────────────────┐
     ▼                     ▼
┌──────────────┐  ┌──────────────────┐
│  Disruption  │  │  Resilience      │
│  Optimizer   │  │  Predictor       │
└──────────────┘  └──────────────────┘
     │                     │
     └──────────┬──────────┘
                ▼
     Disruption Strategy
```

---

### 4. Training Layer

#### `training.py`

**Purpose**: Self-supervised learning framework

**Key Components**:

1. **TemporalLinkPredictionLoss**
   - Predicts future edges
   - Contrastive learning
   - Negative sampling

2. **ContrastiveLoss**
   - Temporal consistency
   - Positive/negative pairs
   - Temperature scaling

3. **NodeReconstructionLoss**
   - Feature reconstruction
   - Autoencoder-style
   - MSE-based

4. **TemporalGNNTrainer**
   - Training orchestration
   - Multi-loss combination
   - Validation and checkpointing

```
Training Data
     │
     ├─────────────────┬────────────────┐
     ▼                 ▼                ▼
┌──────────┐  ┌──────────────┐  ┌──────────┐
│  Link    │  │  Contrastive │  │   Node   │
│Prediction│  │     Loss     │  │  Recon   │
└──────────┘  └──────────────┘  └──────────┘
     │                 │                │
     └────────┬────────┴────────────────┘
              ▼
      Combined Loss
              │
              ▼
     Backpropagation
              │
              ▼
    Model Parameters
```

---

### 5. Baseline Layer

#### `baselines.py`

**Purpose**: Comparison methods

**Components**:
- StaticGCN, StaticGAT, StaticGraphSAGE
- SimpleTemporalGNN
- CentralityBaseline
- BaselineEvaluator

---

## Data Flow

### End-to-End Pipeline

```
1. Network Generation
   ┌──────────────────────┐
   │  NetworkConfig       │
   └──────────────────────┘
             │
             ▼
   ┌──────────────────────┐
   │  Generator           │
   │  - Create nodes      │
   │  - Add edges         │
   │  - Evolve over time  │
   └──────────────────────┘
             │
             ▼
   Multi-Layer Temporal Network

2. Model Training (Optional)
   ┌──────────────────────┐
   │  Trainer             │
   │  - Self-supervised   │
   │  - Multiple losses   │
   │  - Validation        │
   └──────────────────────┘
             │
             ▼
   Trained T-GNN Model

3. Critical Node Detection
   ┌──────────────────────┐
   │  Detector            │
   │  - Compute scores    │
   │  - Rank nodes        │
   │  - Select top-k      │
   └──────────────────────┘
             │
             ▼
   Critical Node Set

4. Disruption Analysis
   ┌──────────────────────┐
   │  Optimizer           │
   │  - Plan strategy     │
   │  - Simulate removal  │
   │  - Evaluate impact   │
   └──────────────────────┘
             │
             ▼
   Disruption Results

5. Resilience Prediction
   ┌──────────────────────┐
   │  Predictor           │
   │  - Forecast recovery │
   │  - Estimate timeline │
   └──────────────────────┘
             │
             ▼
   Resilience Metrics
```

---

## Model Architecture

### AdvancedTemporalGNN Detailed Architecture

```
Input: [x_1, x_2, ..., x_T], [e_1, e_2, ..., e_T], timestamps

┌─────────────────────────────────────────────────┐
│              Input Processing                    │
│  - Node features: [T, N, F_n]                   │
│  - Edge features: [T, E, F_e]                   │
│  - Timestamps: [T]                              │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│          Adaptive Time Encoding                  │
│  - Learnable frequency scaling                  │
│  - Phase shift parameters                       │
│  Output: [T, d_model]                           │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│            Memory Bank Query                     │
│  - Query: current state                         │
│  - Keys: memory vectors                         │
│  - Values: stored patterns                      │
│  - Attention-based retrieval                    │
│  Output: [N, d_memory]                          │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│       Temporal Attention (L layers)              │
│  ┌───────────────────────────────────────────┐  │
│  │  Multi-head Attention                     │  │
│  │  - Q, K, V projections                    │  │
│  │  - Scaled dot-product                     │  │
│  │  - num_heads splits                       │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  Feed-forward Network                     │  │
│  │  - Linear → ReLU → Dropout → Linear       │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  Layer Normalization + Residual           │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│      Graph Transformer (L layers)                │
│  ┌───────────────────────────────────────────┐  │
│  │  Graph Attention                          │  │
│  │  - Edge-aware attention                   │  │
│  │  - Message passing                        │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  Node Update                              │  │
│  │  - Aggregation + transformation           │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│            Output Projection                     │
│  - Linear transformation                        │
│  - Layer normalization                          │
│  Output: [N, hidden_dim]                        │
└─────────────────────────────────────────────────┘
```

---

## Design Patterns

### 1. Strategy Pattern

Used in baseline implementations to allow different node importance computation strategies.

```python
class NodeImportanceStrategy(ABC):
    @abstractmethod
    def compute_importance(self, graph):
        pass

class DegreeCentralityStrategy(NodeImportanceStrategy):
    def compute_importance(self, graph):
        return nx.degree_centrality(graph)

class PageRankStrategy(NodeImportanceStrategy):
    def compute_importance(self, graph):
        return nx.pagerank(graph)
```

### 2. Builder Pattern

Used in network generation to construct complex multi-layer networks.

```python
generator = (TerroristNetworkGenerator(config)
    .with_physical_layer()
    .with_digital_layer()
    .with_financial_layer()
    .with_ideological_layer()
    .with_operational_layer()
    .build())
```

### 3. Observer Pattern

Used in training to monitor and log progress.

```python
trainer.add_callback(LossLogger())
trainer.add_callback(CheckpointSaver())
trainer.add_callback(EarlyStopping())
```

---

## Scalability Considerations

### Memory Management

- **Batch Processing**: Process timesteps in batches
- **Gradient Checkpointing**: Trade compute for memory
- **Sparse Representations**: Use COO format for edges

### Computational Efficiency

- **GPU Acceleration**: CUDA support for all operations
- **Parallel Processing**: Multi-threaded data loading
- **Model Quantization**: Reduce precision for inference

### Distributed Training

- **Data Parallelism**: Split networks across GPUs
- **Model Parallelism**: Split layers across devices
- **Gradient Accumulation**: Effective large batch sizes

---

## Extension Points

### Adding New Models

1. Inherit from `nn.Module`
2. Implement `forward()` method
3. Register in model factory
4. Add tests

### Adding New Evaluation Metrics

1. Implement metric function
2. Add to `DisruptionEvaluator`
3. Update result aggregation
4. Document in API reference

### Adding New Network Layers

1. Define layer type in `NetworkLayer`
2. Implement generation logic
3. Add to `MultiLayerTemporalNetwork`
4. Update aggregation logic

---

## Performance Metrics

### Computational Complexity

- **Network Generation**: O(T × N × L) where T=timesteps, N=nodes, L=layers
- **Model Forward Pass**: O(T × N × d²) where d=hidden_dim
- **Critical Node Detection**: O(N log N) for sorting

### Memory Requirements

- **Model Parameters**: ~5M parameters for default config
- **Network Storage**: ~100MB per network with 100 nodes × 20 timesteps
- **Training Batch**: ~2GB GPU memory for batch_size=32

---

## Security Considerations

### Data Privacy

- Only synthetic data used
- No real network information
- Anonymization built-in

### Model Security

- No adversarial training data
- Defensive use only
- Ethical guidelines enforced

---

## Future Enhancements

1. **Dynamic Architecture Search**: Auto-ML for model design
2. **Federated Learning**: Distributed privacy-preserving training
3. **Explainability**: Attention visualization and SHAP values
4. **Real-time Processing**: Streaming graph updates
5. **Multi-modal Learning**: Integrate text and image data

---

## References

- Graph Neural Network literature
- Temporal graph processing methods
- Network security research

For implementation details, see the source code documentation.
