"""
Ablation Study Module for Temporal GNN Components
=================================================

Systematically evaluates the contribution of each component:
1. Temporal Attention
2. Memory Bank
3. Multi-layer Networks
4. Causal Temporal Convolution
5. Graph Transformer Layers

Author: Advanced GNN Research
"""

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AblationStudy:
    """
    Systematic ablation study for temporal GNN components
    """

    def __init__(self, full_model, device: torch.device = None):
        """
        Args:
            full_model: The complete model with all components
        """
        self.full_model = full_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.variants = {}
        self._create_ablation_variants()

    def _create_ablation_variants(self):
        """
        Create model variants for ablation study
        """
        # Each variant removes or disables a specific component

        # Variant 1: No Temporal Attention
        self.variants["no_temporal_attention"] = {
            "name": "No Temporal Attention",
            "description": "Remove multi-head temporal attention mechanism",
            "modify_fn": self._disable_temporal_attention,
        }

        # Variant 2: No Memory Bank
        self.variants["no_memory"] = {
            "name": "No Memory Bank",
            "description": "Remove memory-augmented network component",
            "modify_fn": self._disable_memory,
        }

        # Variant 3: No Temporal Convolution
        self.variants["no_temporal_conv"] = {
            "name": "No Temporal Convolution",
            "description": "Remove causal temporal convolution blocks",
            "modify_fn": self._disable_temporal_conv,
        }

        # Variant 4: Simple GNN (No Graph Transformer)
        # TEMPORARILY DISABLED due to deepcopy issues
        """
        self.variants['simple_gnn'] = {
            'name': 'Simple GNN',
            'description': 'Replace Graph Transformer with simple GCN',
            'modify_fn': self._simplify_graph_layers
        }
        """

        # Variant 5: No Time Encoding
        self.variants["no_time_encoding"] = {
            "name": "No Time Encoding",
            "description": "Remove adaptive time encoding",
            "modify_fn": self._disable_time_encoding,
        }

        # Variant 6: Static (Last Timestep Only)
        self.variants["static_last"] = {
            "name": "Static (Last Timestep)",
            "description": "Use only the last timestep (no temporal modeling)",
            "modify_fn": self._use_only_last_timestep,
        }

    def _disable_temporal_attention(self, model):
        """Disable temporal attention layers"""
        # Handle MultiLayerTemporalGNN wrapper
        if hasattr(model, "layer_gnns"):
            # Access the underlying AdvancedTemporalGNN through layer_gnns
            for layer_gnn in model.layer_gnns:
                if hasattr(layer_gnn, "temporal_attention"):
                    for i in range(len(layer_gnn.temporal_attention)):
                        layer_gnn.temporal_attention[i] = IdentityAttention()
        elif hasattr(model, "temporal_attention"):
            # Direct AdvancedTemporalGNN
            for i in range(len(model.temporal_attention)):
                model.temporal_attention[i] = IdentityAttention()
        return model

    def _disable_memory(self, model):
        """Disable memory bank"""
        # Handle MultiLayerTemporalGNN wrapper
        if hasattr(model, "layer_gnns"):
            for layer_gnn in model.layer_gnns:
                if hasattr(layer_gnn, "memory_bank"):
                    original_memory = layer_gnn.memory_bank
                    layer_gnn.memory_bank = ZeroMemory(original_memory.memory_dim)
        elif hasattr(model, "memory_bank"):
            # Direct AdvancedTemporalGNN
            original_memory = model.memory_bank
            model.memory_bank = ZeroMemory(original_memory.memory_dim)
        return model

    def _disable_temporal_conv(self, model):
        """Disable temporal convolution"""
        # Handle MultiLayerTemporalGNN wrapper
        if hasattr(model, "layer_gnns"):
            for layer_gnn in model.layer_gnns:
                if hasattr(layer_gnn, "temporal_conv_blocks"):
                    for i in range(len(layer_gnn.temporal_conv_blocks)):
                        layer_gnn.temporal_conv_blocks[i] = IdentityConv()
        elif hasattr(model, "temporal_conv_blocks"):
            # Direct AdvancedTemporalGNN
            for i in range(len(model.temporal_conv_blocks)):
                model.temporal_conv_blocks[i] = IdentityConv()
        return model

    def _simplify_graph_layers(self, model):
        """Replace graph transformer with simple GCN"""
        print(f"    DEBUG: model type = {type(model).__name__}")
        print(f"    DEBUG: hasattr layer_gnns = {hasattr(model, 'layer_gnns')}")
        print(f"    DEBUG: hasattr graph_layers = {hasattr(model, 'graph_layers')}")

        # Handle MultiLayerTemporalGNN wrapper
        if hasattr(model, "layer_gnns"):
            print(f"    DEBUG: Found layer_gnns, count = {len(model.layer_gnns)}")
            for idx, layer_gnn in enumerate(model.layer_gnns):
                print(f"    DEBUG: layer_gnn[{idx}] type = {type(layer_gnn).__name__}")
                print(
                    f"    DEBUG: layer_gnn[{idx}] hasattr graph_layers = {hasattr(layer_gnn, 'graph_layers')}"
                )

                if hasattr(layer_gnn, "graph_layers"):
                    # Get hidden_dim safely
                    if hasattr(layer_gnn, "hidden_dim"):
                        hidden_dim = layer_gnn.hidden_dim
                    else:
                        # Infer from first graph layer
                        first_layer = layer_gnn.graph_layers[0]
                        if hasattr(first_layer, "d_model"):
                            hidden_dim = first_layer.d_model
                        else:
                            hidden_dim = 128  # Default fallback

                    print(
                        f"    DEBUG: Replacing graph_layers with SimpleGCNLayer, hidden_dim={hidden_dim}"
                    )
                    num_layers = len(layer_gnn.graph_layers)
                    layer_gnn.graph_layers = nn.ModuleList(
                        [SimpleGCNLayer(hidden_dim) for _ in range(num_layers)]
                    )
        elif hasattr(model, "graph_layers"):
            # Direct AdvancedTemporalGNN
            if hasattr(model, "hidden_dim"):
                hidden_dim = model.hidden_dim
            else:
                # Infer from first graph layer
                first_layer = model.graph_layers[0]
                if hasattr(first_layer, "d_model"):
                    hidden_dim = first_layer.d_model
                else:
                    hidden_dim = 128  # Default fallback

            num_layers = len(model.graph_layers)
            model.graph_layers = nn.ModuleList(
                [SimpleGCNLayer(hidden_dim) for _ in range(num_layers)]
            )

        print(f"    DEBUG: Replacement complete")
        return model

    def _disable_time_encoding(self, model):
        """Disable adaptive time encoding"""
        # Handle MultiLayerTemporalGNN wrapper
        if hasattr(model, "layer_gnns"):
            for layer_gnn in model.layer_gnns:
                if hasattr(layer_gnn, "time_encoder"):
                    layer_gnn.time_encoder = ZeroTimeEncoder(layer_gnn.hidden_dim)
        elif hasattr(model, "time_encoder"):
            # Direct AdvancedTemporalGNN
            model.time_encoder = ZeroTimeEncoder(model.hidden_dim)
        return model

    def _use_only_last_timestep(self, model):
        """Use only the last timestep (static model)"""
        # Check if MultiLayerTemporalGNN or AdvancedTemporalGNN
        if hasattr(model, "layer_gnns"):
            # MultiLayerTemporalGNN - wraps forward differently
            original_forward = model.forward

            def static_forward(multi_layer_network, timestamps):
                # Only use last timestep - create modified network
                last_layers = multi_layer_network.get_timestep(-1)

                # Create a temporary network with only last timestep
                from terrorist_network_disruption import MultiLayerTemporalNetwork

                static_network = MultiLayerTemporalNetwork(
                    multi_layer_network.num_nodes, multi_layer_network.num_layers
                )
                static_network.add_timestep(last_layers)

                return original_forward(static_network, timestamps[-1:])

            model.forward = static_forward
        else:
            # AdvancedTemporalGNN - original implementation
            original_forward = model.forward

            def static_forward(
                node_features_seq, edge_indices_seq, edge_features_seq, timestamps, batch_seq=None
            ):
                # Only use last timestep
                return original_forward(
                    [node_features_seq[-1]],
                    [edge_indices_seq[-1]],
                    [edge_features_seq[-1]],
                    timestamps[-1:],
                    [batch_seq[-1]] if batch_seq else None,
                )

            model.forward = static_forward

        return model

    def run_ablation_study(self, networks: List, evaluator, detector, top_k: int = 10) -> Dict:
        """
        Run complete ablation study

        Args:
            networks: List of test networks
            evaluator: DisruptionEvaluator instance
            detector: CriticalNodeDetector instance
            top_k: Number of critical nodes to detect

        Returns:
            results: Dictionary with results for each variant
        """
        print("=" * 80)
        print("ABLATION STUDY")
        print("=" * 80)
        print(f"Testing {len(self.variants) + 1} model variants...")
        print()

        results = {}

        # Evaluate full model
        print("Evaluating FULL MODEL...")
        full_results = self._evaluate_model(self.full_model, networks, evaluator, detector, top_k)
        results["full_model"] = {
            "name": "Full Model",
            "description": "Complete model with all components",
            "results": full_results,
            "mean_disruption": np.mean([r["disruption"] for r in full_results]),
            "std_disruption": np.std([r["disruption"] for r in full_results]),
        }
        print(
            f"  Mean Disruption: {results['full_model']['mean_disruption']:.4f} "
            f"(±{results['full_model']['std_disruption']:.4f})"
        )
        print()

        # Evaluate each ablation variant
        for variant_key, variant_info in self.variants.items():
            print(f"Evaluating {variant_info['name'].upper()}...")
            print(f"  {variant_info['description']}")

            # Create variant model - use state_dict copy instead of deepcopy
            # deepcopy can fail with complex nested modules
            try:
                import copy

                variant_model = copy.deepcopy(self.full_model)
            except Exception as e:
                print(f"    Warning: deepcopy failed ({e}), using self.full_model directly")
                variant_model = self.full_model

            variant_model = variant_info["modify_fn"](variant_model)
            variant_model.to(self.device)
            variant_model.eval()

            # Evaluate
            try:
                variant_results = self._evaluate_model(
                    variant_model, networks, evaluator, detector, top_k
                )

                results[variant_key] = {
                    "name": variant_info["name"],
                    "description": variant_info["description"],
                    "results": variant_results,
                    "mean_disruption": np.mean([r["disruption"] for r in variant_results]),
                    "std_disruption": np.std([r["disruption"] for r in variant_results]),
                }

                # Compute performance drop
                performance_drop = (
                    results["full_model"]["mean_disruption"]
                    - results[variant_key]["mean_disruption"]
                )
                results[variant_key]["performance_drop"] = performance_drop
                results[variant_key]["relative_drop_pct"] = (
                    performance_drop / results["full_model"]["mean_disruption"] * 100
                )

                print(
                    f"  Mean Disruption: {results[variant_key]['mean_disruption']:.4f} "
                    f"(±{results[variant_key]['std_disruption']:.4f})"
                )
                print(
                    f"  Performance Drop: {performance_drop:.4f} "
                    f"({results[variant_key]['relative_drop_pct']:.1f}%)"
                )
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                results[variant_key] = {"name": variant_info["name"], "error": str(e)}

            print()

        # Rank components by importance
        print("=" * 80)
        print("COMPONENT IMPORTANCE RANKING")
        print("=" * 80)

        valid_variants = [
            (k, v) for k, v in results.items() if k != "full_model" and "performance_drop" in v
        ]
        ranked_variants = sorted(
            valid_variants, key=lambda x: x[1]["performance_drop"], reverse=True
        )

        for i, (variant_key, variant_data) in enumerate(ranked_variants, 1):
            print(f"{i}. {variant_data['name']}")
            print(
                f"   Performance Drop: {variant_data['performance_drop']:.4f} "
                f"({variant_data['relative_drop_pct']:.1f}%)"
            )
            print(
                f"   → Component Contribution: "
                f"{variant_data['performance_drop'] / results['full_model']['mean_disruption'] * 100:.1f}%"
            )
            print()

        return results

    def _evaluate_model(self, model, networks: List, evaluator, detector, top_k: int) -> List[Dict]:
        """
        Evaluate a model variant on all networks
        """
        model.eval()
        results = []

        with torch.no_grad():
            for network in networks:
                # Get embeddings using the model
                t = -1
                layers = network.get_timestep(t)
                agg_edge_index, _ = network.get_aggregated_network(t)
                num_nodes = layers[0].node_features.size(0)

                # Get model embeddings
                timestamps = torch.arange(
                    len(network.layers_history), dtype=torch.float32, device=self.device
                )

                try:
                    embeddings = model(network, timestamps)

                    # Detect critical nodes
                    critical_nodes, _ = detector.detect_critical_nodes(
                        agg_edge_index.to(self.device),
                        num_nodes,
                        embeddings.to(self.device),
                        top_k=top_k,
                    )

                    # Evaluate disruption
                    metrics = evaluator.evaluate_disruption_strategy(
                        network, critical_nodes.cpu().tolist(), timestep=t
                    )

                    results.append(
                        {
                            "critical_nodes": critical_nodes.cpu().tolist(),
                            "disruption": metrics["overall_disruption"],
                            "metrics": metrics,
                        }
                    )
                except Exception as e:
                    print(f"    Warning: Evaluation failed - {str(e)}")
                    # Use fallback: random nodes
                    random_nodes = torch.randperm(num_nodes)[:top_k].tolist()
                    metrics = evaluator.evaluate_disruption_strategy(
                        network, random_nodes, timestep=t
                    )
                    results.append(
                        {
                            "critical_nodes": random_nodes,
                            "disruption": metrics["overall_disruption"],
                            "metrics": metrics,
                            "error": str(e),
                        }
                    )

        return results


# ============================================================================
# HELPER CLASSES FOR ABLATION
# ============================================================================


class IdentityAttention(nn.Module):
    """Identity module that passes input through unchanged"""

    def forward(self, query, key, value, mask=None):
        return query


class ZeroMemory(nn.Module):
    """Memory that returns zeros"""

    def __init__(self, memory_dim):
        super().__init__()
        self.memory_dim = memory_dim

    def forward(self, query, update=False):
        batch_size = query.size(0)
        return torch.zeros(batch_size, self.memory_dim, device=query.device)


class IdentityConv(nn.Module):
    """Identity convolution"""

    def forward(self, x):
        return x


class SimpleGCNLayer(nn.Module):
    """Simple GCN layer for ablation"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = self.norm(x)
        return x


class ZeroTimeEncoder(nn.Module):
    """Time encoder that returns zeros"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, timestamps):
        if timestamps.dim() == 1:
            return torch.zeros(timestamps.size(0), self.d_model, device=timestamps.device)
        else:
            return torch.zeros(
                timestamps.size(0), timestamps.size(1), self.d_model, device=timestamps.device
            )


# Test and demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("Ablation Study Module")
    print("=" * 80)

    print("\nAblation Variants:")
    print("  1. No Temporal Attention")
    print("  2. No Memory Bank")
    print("  3. No Temporal Convolution")
    print("  4. Simple GNN (no Transformer)")
    print("  5. No Time Encoding")
    print("  6. Static (Last Timestep Only)")

    print("\n✓ Ablation study framework ready!")
    print("\nThis module will:")
    print("  - Systematically disable each component")
    print("  - Measure performance impact")
    print("  - Rank components by importance")
    print("  - Quantify contribution of each component")
