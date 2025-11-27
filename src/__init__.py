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

__version__ = "1.0.0"
__author__ = "Yoon-seop Lee"
__email__ = "iyunseob4@gmail.com"

from .ablation_study import (
    AblationStudy,
    run_ablation_study,
)
from .advanced_tgnn import (
    AdaptiveTimeEncoding,
    AdvancedTemporalGNN,
    EnhancedTemporalMemoryBank,
    HierarchicalTemporalPooling,
)
from .baselines import (
    BaselineMethod,
    compute_centrality,
    run_baseline_comparison,
)
from .main_experiment import (
    EnhancedExperiment,
    ExperimentConfig,
)
from .statistical_analysis import (
    StatisticalValidator,
    compute_effect_size,
    run_statistical_tests,
)
from .terrorist_network_dataset import (
    DisruptionEvaluator,
    MultiLayerTemporalNetwork,
    NetworkAugmenter,
    NetworkConfig,
    TerroristNetworkGenerator,
)
from .terrorist_network_disruption import (
    AdversarialNetworkAttack,
    EnhancedCriticalNodeDetector,
    MultiLayerTemporalGNN,
    NetworkDisruptionOptimizer,
    TemporalResiliencePredictor,
)
from .training import (
    EnhancedTemporalGNNTrainer,
    GraphReconstructionLoss,
    TemporalAutoencoderLoss,
)

__all__ = [
    # Core architecture
    "AdvancedTemporalGNN",
    "HierarchicalTemporalPooling",
    "EnhancedTemporalMemoryBank",
    "AdaptiveTimeEncoding",
    # Disruption analysis
    "MultiLayerTemporalGNN",
    "EnhancedCriticalNodeDetector",
    "TemporalResiliencePredictor",
    "AdversarialNetworkAttack",
    "NetworkDisruptionOptimizer",
    # Dataset
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
    "BaselineMethod",
    "compute_centrality",
    "run_baseline_comparison",
    # Statistical analysis
    "StatisticalValidator",
    "compute_effect_size",
    "run_statistical_tests",
    # Ablation study
    "AblationStudy",
    "run_ablation_study",
    # Main experiment
    "EnhancedExperiment",
    "ExperimentConfig",
]
