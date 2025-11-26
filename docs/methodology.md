# Research Methodology

This document describes the research methodology, experimental design, and evaluation protocols for the Terrorist Network GNN project.

## Table of Contents

1. [Research Overview](#research-overview)
2. [Problem Formulation](#problem-formulation)
3. [Research Questions](#research-questions)
4. [Approach](#approach)
5. [Experimental Design](#experimental-design)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Statistical Analysis](#statistical-analysis)
8. [Baseline Methods](#baseline-methods)

---

## Research Overview

### Objective

Develop and evaluate advanced machine learning techniques for identifying critical vulnerabilities in terrorist networks and predicting their resilience patterns using temporal graph neural networks.

### Scope

- **Domain**: Network security and counterterrorism
- **Data**: Synthetic multi-layer temporal networks
- **Methods**: Deep learning, graph neural networks
- **Application**: Defensive intelligence and security

### Ethical Framework

This research is conducted under strict ethical guidelines:

✅ **Defensive Purpose**: Protect civilian populations
✅ **Synthetic Data Only**: No real terrorist network data
✅ **Responsible Disclosure**: Proper ethical oversight
✅ **Transparent Methods**: Open-source and peer-reviewed

---

## Problem Formulation

### Network Disruption as an Optimization Problem

Given a temporal multi-layer network **G(t) = {V(t), E(t), L}** where:
- V(t) = set of nodes (operatives) at time t
- E(t) = set of edges (connections) at time t
- L = set of network layers (physical, digital, financial, ideological, operational)

**Goal**: Find a set of nodes **S ⊂ V(t)** to remove such that disruption effectiveness is maximized:

```
S* = argmax D(G(t), S)
      S⊂V(t)
      |S|≤k
```

Where **D(G(t), S)** measures network disruption after removing nodes in S.

### Multi-Objective Optimization

The problem involves multiple competing objectives:

1. **Immediate Disruption**: Maximize short-term network damage
2. **Long-term Impact**: Minimize network reconstruction capability
3. **Resource Efficiency**: Minimize number of nodes to remove
4. **Robustness**: Ensure strategy works across network variations

---

## Research Questions

### Q1: Critical Node Detection

**Question**: Which nodes, when removed, most effectively disrupt the network?

**Hypothesis**: Temporal GNN-based node importance scores outperform traditional centrality measures by capturing:
- Dynamic network evolution
- Multi-layer interactions
- Temporal dependencies
- Structural and functional importance

**Approach**:
- Learn node embeddings using AdvancedTemporalGNN
- Combine multiple importance signals (centrality + learned representations)
- Ensemble scoring for robust detection

**Metrics**:
- Disruption score after removal
- Network fragmentation
- Largest connected component size
- Communication efficiency reduction

---

### Q2: Temporal Resilience Prediction

**Question**: How will the network reconstruct after node removal?

**Hypothesis**: Temporal patterns in network evolution can predict future resilience and reconstruction trajectories.

**Approach**:
- Train LSTM-based resilience predictor
- Input: historical network states before disruption
- Output: predicted resilience scores for future timesteps

**Metrics**:
- Prediction accuracy (MAE, RMSE)
- Resilience recovery time
- Structural similarity to pre-disruption state
- Communication restoration rate

---

### Q3: Adversarial Robustness

**Question**: How does the network adapt to disruption attempts?

**Hypothesis**: Networks exhibit adaptive responses:
- Recruitment acceleration
- Connection rewiring
- Leadership replacement
- Communication rerouting

**Approach**:
- Simulate adversarial network adaptation
- Model counter-strategies
- Evaluate disruption strategy robustness

**Metrics**:
- Adaptation rate
- Counter-strategy effectiveness
- Stability under perturbations
- Strategic robustness score

---

## Approach

### 1. Data Generation

#### Network Characteristics

Realistic terrorist networks exhibit:

**Structural Properties**:
- Scale-free degree distribution (power law)
- High clustering coefficient (local cohesion)
- Small-world property (short path lengths)
- Modular community structure

**Temporal Properties**:
- Gradual growth through recruitment
- Periodic activity patterns
- Event-driven edge formation
- Selective member dropout

**Multi-Layer Properties**:
- Layer interdependencies
- Cross-layer correlations
- Layer-specific dynamics
- Hierarchical organization

#### Generation Algorithm

```
Algorithm: GenerateTemporalNetwork
Input: config (NetworkConfig), T (num_timesteps)
Output: Multi-layer temporal network

1. Initialize core nodes with high connectivity
2. For t = 1 to T:
   a. Recruitment phase:
      - Add new nodes with probability p_recruit
      - Connect to existing nodes via preferential attachment
   b. Evolution phase:
      - Form new edges based on triadic closure
      - Remove edges with probability p_dropout
      - Update node features based on activity
   c. Multi-layer formation:
      - Generate 5 layer-specific networks
      - Add cross-layer correlations
      - Apply layer-specific dynamics
3. Return network sequence
```

---

### 2. Model Architecture

#### AdvancedTemporalGNN Design Rationale

**Component Selection**:

1. **Adaptive Time Encoding**
   - Rationale: Capture irregular temporal patterns
   - Alternative considered: Fixed sinusoidal encoding
   - Advantage: Learnable frequency and phase

2. **Memory Bank**
   - Rationale: Store long-term temporal patterns
   - Alternative considered: Standard LSTM
   - Advantage: Attention-based selective retrieval

3. **Multi-Head Temporal Attention**
   - Rationale: Attend to relevant historical timesteps
   - Alternative considered: Uniform temporal aggregation
   - Advantage: Adaptive temporal weighting

4. **Graph Transformer**
   - Rationale: Rich message passing
   - Alternative considered: Standard GCN
   - Advantage: Edge feature integration

#### Hyperparameter Selection

| Parameter | Value | Justification |
|-----------|-------|---------------|
| hidden_dim | 128 | Balance capacity and efficiency |
| num_temporal_layers | 3 | Sufficient temporal depth |
| num_graph_layers | 3 | Adequate spatial depth |
| num_attention_heads | 8 | Standard transformer setting |
| memory_size | 100 | Reasonable memory capacity |
| dropout | 0.1 | Prevent overfitting |
| learning_rate | 1e-3 | Adam default |

---

### 3. Training Protocol

#### Self-Supervised Learning

**Rationale**: No labeled data available for critical node detection

**Training Objectives**:

1. **Temporal Link Prediction** (weight: 0.4)
   ```
   L_link = -log(σ(z_u^T z_v)) - Σ log(σ(-z_u^T z_neg))
   ```
   Predicts future edges from node embeddings

2. **Contrastive Learning** (weight: 0.3)
   ```
   L_contrast = -log(exp(sim(z_t, z_t+1)/τ) / Σ exp(sim(z_t, z_neg)/τ))
   ```
   Enforces temporal consistency

3. **Node Reconstruction** (weight: 0.3)
   ```
   L_recon = ||X - X_hat||²
   ```
   Reconstructs node features

**Combined Loss**:
```
L_total = λ1·L_link + λ2·L_contrast + λ3·L_recon
```

#### Training Procedure

```
Algorithm: TrainTemporalGNN
Input: model, train_networks, val_networks, epochs
Output: Trained model

1. Initialize optimizer (Adam, lr=1e-3)
2. For epoch = 1 to epochs:
   a. Training phase:
      For each network in train_networks:
        - Forward pass
        - Compute multi-loss
        - Backward pass
        - Update parameters
   b. Validation phase:
      For each network in val_networks:
        - Forward pass (no grad)
        - Compute validation loss
   c. Learning rate scheduling:
      - Reduce on plateau
      - Factor: 0.5
      - Patience: 5 epochs
   d. Early stopping:
      - Monitor validation loss
      - Patience: 10 epochs
3. Return best model (by validation loss)
```

---

## Experimental Design

### Phase 1: Model Development

**Objective**: Develop and validate T-GNN architecture

**Experiments**:
1. Architecture ablation study
2. Hyperparameter tuning
3. Training objective comparison
4. Convergence analysis

**Dataset**:
- 10 networks
- 20 timesteps each
- 50-80 nodes

---

### Phase 2: Baseline Comparison

**Objective**: Compare against established methods

**Baselines**:
1. Static GNN methods (GCN, GAT, GraphSAGE)
2. Traditional centrality (Degree, Betweenness, PageRank)
3. Simple temporal GNN
4. Graph partitioning

**Experiments**:
- 50 networks per method
- 5-fold cross-validation
- Statistical significance testing

---

### Phase 3: Ablation Study

**Objective**: Understand component contributions

**Components to Ablate**:
- [ ] Remove temporal attention → Static aggregation
- [ ] Remove memory bank → No long-term memory
- [ ] Remove graph transformer → Standard GCN
- [ ] Remove adaptive time encoding → Fixed encoding
- [ ] Remove multi-layer → Single layer

**Metrics**: Performance drop when component removed

---

### Phase 4: Robustness Analysis

**Objective**: Test adversarial robustness

**Experiments**:
1. Network perturbations (add/remove edges)
2. Adaptive adversarial networks
3. Transfer across network types
4. Noise sensitivity

---

## Evaluation Metrics

### Network Disruption Metrics

#### 1. Fragmentation Score

```python
def fragmentation_score(G_before, G_after):
    """
    Measures network fragmentation after disruption

    Range: [0, 1]
    Higher = more fragmentation
    """
    components_before = nx.number_connected_components(G_before)
    components_after = nx.number_connected_components(G_after)

    return (components_after - components_before) / len(G_before)
```

#### 2. Largest Component Reduction

```python
def lcc_reduction(G_before, G_after):
    """
    Reduction in largest connected component size

    Range: [0, 1]
    Higher = better disruption
    """
    lcc_before = len(max(nx.connected_components(G_before), key=len))
    lcc_after = len(max(nx.connected_components(G_after), key=len))

    return (lcc_before - lcc_after) / lcc_before
```

#### 3. Communication Efficiency

```python
def efficiency_reduction(G_before, G_after):
    """
    Reduction in global communication efficiency

    Range: [0, 1]
    Higher = more disruption
    """
    eff_before = nx.global_efficiency(G_before)
    eff_after = nx.global_efficiency(G_after)

    return (eff_before - eff_after) / eff_before if eff_before > 0 else 0
```

#### 4. Composite Disruption Score

```python
def disruption_score(G_before, G_after, removed_nodes):
    """
    Weighted combination of multiple metrics
    """
    frag = fragmentation_score(G_before, G_after)
    lcc = lcc_reduction(G_before, G_after)
    eff = efficiency_reduction(G_before, G_after)
    cost = len(removed_nodes) / len(G_before)

    # Weighted average (disruption high, cost low)
    return 0.3*frag + 0.4*lcc + 0.3*eff - 0.1*cost
```

---

### Resilience Metrics

#### 1. Recovery Time

Time steps until network reaches 80% of original connectivity

#### 2. Reconstruction Rate

```python
def reconstruction_rate(G_original, G_recovered):
    """
    Similarity between original and recovered network
    """
    return graph_edit_distance(G_original, G_recovered)
```

#### 3. Structural Resilience

```python
def structural_resilience(networks_sequence, disruption_time):
    """
    Area under the curve of network size after disruption
    """
    sizes = [len(G) for G in networks_sequence[disruption_time:]]
    original_size = len(networks_sequence[disruption_time-1])

    return np.trapz(sizes) / (len(sizes) * original_size)
```

---

## Statistical Analysis

### Significance Testing

#### Paired t-test

For comparing two methods on same networks:
```
H0: μ_method1 = μ_method2
H1: μ_method1 ≠ μ_method2

α = 0.05
```

#### Wilcoxon Signed-Rank Test

Non-parametric alternative for non-normal distributions

#### Effect Size (Cohen's d)

```python
def cohens_d(group1, group2):
    """
    Effect size measure

    Small: d = 0.2
    Medium: d = 0.5
    Large: d = 0.8
    """
    pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

---

### Multiple Comparison Correction

**Bonferroni Correction**:
```
α_corrected = α / n_comparisons
```

For 10 baseline comparisons: α_corrected = 0.05 / 10 = 0.005

---

## Baseline Methods

### 1. Static GNN Baselines

**GCN (Graph Convolutional Network)**:
- Simple message passing
- No temporal modeling
- Fast but limited expressiveness

**GAT (Graph Attention Network)**:
- Attention-based aggregation
- Learn edge importance
- Better than GCN but still static

**GraphSAGE**:
- Sampling-based aggregation
- Scalable to large graphs
- Inductive learning

### 2. Centrality Baselines

**Degree Centrality**:
```python
scores = nx.degree_centrality(G)
```

**Betweenness Centrality**:
```python
scores = nx.betweenness_centrality(G)
```

**PageRank**:
```python
scores = nx.pagerank(G)
```

**Eigenvector Centrality**:
```python
scores = nx.eigenvector_centrality(G)
```

### 3. Temporal Baseline

**Simple T-GNN**:
- Basic RNN + GCN
- No attention mechanism
- No memory bank
- Simpler than AdvancedTemporalGNN

---

## Reproducibility

### Random Seed Control

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### Configuration Logging

All experiments log:
- Model hyperparameters
- Network generation parameters
- Training configuration
- Hardware specifications
- Library versions

### Code Versioning

- Git commit hash recorded
- Dependencies pinned in requirements.txt
- Docker image for environment

---

## Limitations

### 1. Synthetic Data

- May not capture all real-world complexities
- Simplified network dynamics
- No access to ground truth labels

### 2. Computational Resources

- Limited to networks with < 100 nodes
- Training time constraints
- GPU memory limitations

### 3. Evaluation Scope

- Focus on network-level metrics
- Individual node impact not fully modeled
- Socio-political factors not included

---

## Future Work

1. **Transfer Learning**: Apply to related domains
2. **Active Learning**: Iteratively improve with expert feedback
3. **Explainability**: Interpret model decisions
4. **Real-world Validation**: Collaborate with domain experts
5. **Dynamic Adaptation**: Online learning from network evolution

---

## References

### Graph Neural Networks

- Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks
- Veličković et al. (2018). Graph Attention Networks
- Hamilton et al. (2017). Inductive Representation Learning on Large Graphs

### Temporal Networks

- Xu et al. (2020). Inductive Representation Learning on Temporal Graphs
- Sankar et al. (2020). DySAT: Deep Neural Representation Learning on Dynamic Graphs
- Pareja et al. (2020). EvolveGCN: Evolving Graph Convolutional Networks

### Network Disruption

- Albert et al. (2000). Error and attack tolerance of complex networks
- Cohen et al. (2001). Breakdown of the Internet under intentional attack
- Holme et al. (2002). Attack vulnerability of complex networks

---

For implementation details, see the [Architecture Documentation](architecture.md).

For usage examples, see the [Tutorial](tutorial.md).
