#!/usr/bin/env python
"""
Main Experiment Runner
=====================

Command-line interface for running terrorist network disruption experiments.

Usage:
    python run_experiment.py --num-networks 10 --num-timesteps 20 --train-model

Author: Yoon-Seop Lee
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from main_experiment import EnhancedExperiment
from terrorist_network_dataset import NetworkConfig


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Run terrorist network disruption experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment configuration
    parser.add_argument('--num-networks', type=int, default=10,
                        help='Number of networks to generate')
    parser.add_argument('--num-timesteps', type=int, default=20,
                        help='Number of temporal timesteps')
    parser.add_argument('--output-dir', type=str, default='experiments/phase1',
                        help='Output directory for results')
    
    # Model configuration
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num-node-features', type=int, default=64,
                        help='Number of node features')
    parser.add_argument('--num-edge-features', type=int, default=32,
                        help='Number of edge features')
    
    # Network configuration
    parser.add_argument('--initial-nodes', type=int, default=50,
                        help='Initial number of nodes')
    parser.add_argument('--max-nodes', type=int, default=80,
                        help='Maximum number of nodes')
    parser.add_argument('--recruitment-rate', type=float, default=0.05,
                        help='New member recruitment rate')
    parser.add_argument('--dropout-rate', type=float, default=0.02,
                        help='Member dropout rate')
    
    # Experiment flags
    parser.add_argument('--train-model', action='store_true',
                        help='Train model with self-supervised learning')
    parser.add_argument('--run-baselines', action='store_true',
                        help='Run baseline method comparisons')
    parser.add_argument('--run-ablation', action='store_true',
                        help='Run ablation study')
    parser.add_argument('--no-train', dest='train_model', action='store_false',
                        help='Skip model training')
    parser.add_argument('--no-baselines', dest='run_baselines', action='store_false',
                        help='Skip baseline comparisons')
    parser.add_argument('--no-ablation', dest='run_ablation', action='store_false',
                        help='Skip ablation study')
    
    # GPU configuration
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for computation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Set defaults
    parser.set_defaults(train_model=True, run_baselines=True, run_ablation=True)
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup computation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return device


def create_network_config(args):
    """Create network configuration from arguments"""
    return NetworkConfig(
        initial_nodes=args.initial_nodes,
        max_nodes=args.max_nodes,
        recruitment_rate=args.recruitment_rate,
        dropout_rate=args.dropout_rate,
        physical_density=0.15,
        digital_density=0.35,
        financial_density=0.08,
        ideological_density=0.30,
        operational_density=0.06
    )


def create_model_config(args):
    """Create model configuration from arguments"""
    return {
        'num_node_features': args.num_node_features,
        'num_edge_features': args.num_edge_features,
        'hidden_dim': args.hidden_dim
    }


def main():
    """Main execution function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create configurations
    network_config = create_network_config(args)
    model_config = create_model_config(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print experiment configuration
    print("=" * 80)
    print("TERRORIST NETWORK DISRUPTION EXPERIMENT")
    print("=" * 80)
    print(f"\nExperiment Configuration:")
    print(f"  Networks: {args.num_networks}")
    print(f"  Timesteps: {args.num_timesteps}")
    print(f"  Output: {args.output_dir}")
    print(f"  Seed: {args.seed}")
    print(f"\nModel Configuration:")
    print(f"  Hidden Dimension: {args.hidden_dim}")
    print(f"  Node Features: {args.num_node_features}")
    print(f"  Edge Features: {args.num_edge_features}")
    print(f"\nNetwork Configuration:")
    print(f"  Initial Nodes: {args.initial_nodes}")
    print(f"  Max Nodes: {args.max_nodes}")
    print(f"  Recruitment Rate: {args.recruitment_rate}")
    print(f"  Dropout Rate: {args.dropout_rate}")
    print(f"\nExperiment Components:")
    print(f"  Train Model: {args.train_model}")
    print(f"  Run Baselines: {args.run_baselines}")
    print(f"  Run Ablation: {args.run_ablation}")
    print()
    
    # Create experiment
    experiment = EnhancedExperiment(
        config=network_config,
        model_config=model_config,
        output_dir=args.output_dir
    )
    
    # Run experiment
    try:
        experiment.run_complete_experiment(
            num_networks=args.num_networks,
            num_timesteps=args.num_timesteps,
            train_model=args.train_model,
            run_baselines=args.run_baselines,
            run_ablation=args.run_ablation
        )
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nResults saved to: {args.output_dir}")
        print("\nTo view results:")
        print(f"  python scripts/visualize_results.py --input-dir {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("EXPERIMENT FAILED")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
