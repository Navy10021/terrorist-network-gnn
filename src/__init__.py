"""
Terrorist Network Disruption using Temporal Graph Neural Networks
====================================================================

A comprehensive framework for analyzing and disrupting terrorist networks
using advanced Temporal Graph Neural Networks (T-GNN).

Main modules:
- advanced_tgnn: Core T-GNN architecture
- terrorist_network_disruption: Disruption algorithms and analysis
- terrorist_network_dataset: Network generation and evaluation
- training: Model training with self-supervised learning
- baselines: Comparison methods
- statistical_analysis: Statistical validation
- ablation_study: Component importance analysis
- main_experiment: Complete experimental pipeline

Example usage:
    >>> from terrorist_network_dataset import NetworkConfig, TerroristNetworkGenerator
    >>> from terrorist_network_disruption import MultiLayerTemporalGNN
    >>>
    >>> config = NetworkConfig(initial_nodes=50, max_nodes=80)
    >>> generator = TerroristNetworkGenerator(config)
    >>> network = generator.generate_temporal_network(num_timesteps=20)
"""

__version__ = "2.0.0"
__author__ = "Yoon-seop Lee"
__email__ = "iyunseob4@gmail.com"


# --------------------------------------------------------
# Import actual modules (project reality 기반)
# --------------------------------------------------------

# --- Ablation ---
from .ablation_study import AblationStudy

# --- Core Temporal GNN architecture ---
from .advanced_tgnn import (
    AdaptiveTimeEncoding,
    AdvancedTemporalGNN,
    EnhancedTemporalMemoryBank,
    HierarchicalTemporalPooling,
)

# --- Baselines (정확한 실제 클래스만 포함) ---
from .baselines import (
    StaticGCN,
    StaticGAT,
    StaticGraphSAGE,
    DynamicGCN,
    EvolveGCN,
    SimpleTemporalGNN,
    BaselineEvaluator,
)

# --- Main Experiment (실제로 존재하는 클래스) ---
from .main_experiment import TemporalGNNExperiment

# --- Statistical analysis ---
from .statistical_analysis import (
    StatisticalAnalyzer,
    ResultVisualizer,
)

# --- Dataset & network generation ---
from .terrorist_network_dataset import (
    DisruptionEvaluator,
    MultiLayerTemporalNetwork,
    NetworkAugmenter,
    NetworkConfig,
    TerroristNetworkGenerator,
)

# --- Disruption, adversarial analysis, resilience prediction ---
from .terrorist_network_disruption import (
    AdversarialNetworkAttack,
    EnhancedCriticalNodeDetector,
    MultiLayerTemporalGNN,
    NetworkDisruptionOptimizer,
    TemporalResiliencePredictor,
)

# --- Training ---
from .training import (
    EnhancedTemporalGNNTrainer,
    GraphReconstructionLoss,
    TemporalAutoencoderLoss,
)


# --------------------------------------------------------
# Public API
# --------------------------------------------------------

__all__ = [

    # Core architecture
    "AdvancedTemporalGNN",
    "HierarchicalTemporalPooling",
    "EnhancedTemporalMemoryBank",
    "AdaptiveTimeEncoding",

    # Disruption & robustness
    "MultiLayerTemporalGNN",
    "EnhancedCriticalNodeDetector",
    "TemporalResiliencePredictor",
    "AdversarialNetworkAttack",
    "NetworkDisruptionOptimizer",

    # Dataset & generation
    "TerroristNetworkGenerator",
    "NetworkConfig",
    "NetworkAugmenter",
    "DisruptionEvaluator",
    "MultiLayerTemporalNetwork",

    # Training
    "EnhancedTemporalGNNTrainer",
    "TemporalAutoencoderLoss",
    "GraphReconstructionLoss",

    # Baselines
    "StaticGCN",
    "StaticGAT",
    "StaticGraphSAGE",
    "DynamicGCN",
    "EvolveGCN",
    "SimpleTemporalGNN",
    "BaselineEvaluator",

    # Statistical analysis
    "StatisticalAnalyzer",
    "ResultVisualizer",

    # Ablation
    "AblationStudy",

    # Main experiment pipeline
    "TemporalGNNExperiment",
]
