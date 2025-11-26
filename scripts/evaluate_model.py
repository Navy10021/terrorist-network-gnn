#!/usr/bin/env python
"""
Model Evaluation Script
======================

Evaluate trained models on test datasets and generate comprehensive reports.

Usage:
    python evaluate_model.py --model-path checkpoints/model.pt --data-dir experiments/test_data

Author: Yoon-Seop Lee
"""

import argparse
import sys
import os
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from terrorist_network_disruption import MultiLayerTemporalGNN, CriticalNodeDetector
from terrorist_network_dataset import TerroristNetworkGenerator, NetworkConfig, DisruptionEvaluator
from statistical_analysis import StatisticalAnalyzer, ResultVisualizer


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained terrorist network GNN model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config-path', type=str, default=None,
                        help='Path to model configuration JSON')
    
    # Data configuration
    parser.add_argument('--num-test-networks', type=int, default=20,
                        help='Number of test networks to generate')
    parser.add_argument('--num-timesteps', type=int, default=20,
                        help='Number of temporal timesteps')
    
    # Evaluation settings
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of critical nodes to detect')
    parser.add_argument('--metrics', nargs='+', 
                        default=['disruption', 'fragmentation', 'resilience'],
                        help='Metrics to evaluate')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--save-visualizations', action='store_true',
                        help='Generate and save visualization plots')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed evaluation information')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    return parser.parse_args()


def load_model(model_path, config_path=None, device='cpu'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}...")
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'num_node_features': 64,
            'num_edge_features': 32,
            'hidden_dim': 128,
            'num_layers': 5
        }
    
    # Create model
    model = MultiLayerTemporalGNN(
        num_node_features=config['num_node_features'],
        num_edge_features=config['num_edge_features'],
        hidden_dim=config['hidden_dim'],
        num_layers=config.get('num_layers', 5)
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model, config


def generate_test_data(num_networks, num_timesteps, config, device):
    """Generate test networks"""
    print(f"\nGenerating {num_networks} test networks...")
    
    network_config = NetworkConfig(
        initial_nodes=50,
        max_nodes=80,
        recruitment_rate=0.05,
        dropout_rate=0.02
    )
    
    networks = []
    for i in range(num_networks):
        generator = TerroristNetworkGenerator(network_config, seed=1000+i)
        network = generator.generate_temporal_network(
            num_timesteps=num_timesteps,
            num_node_features=config['num_node_features'],
            num_edge_features=config['num_edge_features'],
            device=device
        )
        networks.append(network)
        
        if (i+1) % 5 == 0:
            print(f"  Generated {i+1}/{num_networks} networks")
    
    print(f"✓ Test data generation complete")
    return networks


def evaluate_model(model, networks, detector, evaluator, top_k=10):
    """Evaluate model on test networks"""
    print(f"\nEvaluating model on {len(networks)} networks...")
    
    results = []
    
    with torch.no_grad():
        for i, network in enumerate(networks):
            # Get network data
            t = -1
            layers = network.get_timestep(t)
            agg_edge_index, _ = network.get_aggregated_network(t)
            num_nodes = layers[0].node_features.size(0)
            
            # Get embeddings
            timestamps = torch.arange(
                len(network.layers_history),
                dtype=torch.float32,
                device=model.device if hasattr(model, 'device') else torch.device('cpu')
            )
            embeddings = model(network, timestamps)
            
            # Detect critical nodes
            critical_nodes, importance_scores = detector.detect_critical_nodes(
                agg_edge_index,
                num_nodes,
                embeddings,
                top_k=top_k
            )
            
            # Evaluate disruption
            metrics = evaluator.evaluate_disruption_strategy(
                network, critical_nodes.cpu().tolist(), timestep=t
            )
            
            results.append({
                'network_id': i,
                'critical_nodes': critical_nodes.cpu().tolist(),
                'importance_scores': {k: v.cpu().tolist() for k, v in importance_scores.items()},
                'disruption_metrics': metrics
            })
            
            if (i+1) % 5 == 0:
                print(f"  Evaluated {i+1}/{len(networks)} networks")
    
    print(f"✓ Evaluation complete")
    return results


def generate_report(results, output_dir, save_visualizations=True):
    """Generate comprehensive evaluation report"""
    print(f"\nGenerating evaluation report...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Aggregate metrics
    disruption_scores = [r['disruption_metrics']['overall_disruption'] for r in results]
    fragmentation_scores = [r['disruption_metrics']['fragmentation'] for r in results]
    
    # Calculate statistics
    stats = {
        'timestamp': datetime.now().isoformat(),
        'num_networks': len(results),
        'disruption': {
            'mean': float(np.mean(disruption_scores)),
            'std': float(np.std(disruption_scores)),
            'min': float(np.min(disruption_scores)),
            'max': float(np.max(disruption_scores)),
            'median': float(np.median(disruption_scores))
        },
        'fragmentation': {
            'mean': float(np.mean(fragmentation_scores)),
            'std': float(np.std(fragmentation_scores)),
            'min': float(np.min(fragmentation_scores)),
            'max': float(np.max(fragmentation_scores)),
            'median': float(np.median(fragmentation_scores))
        }
    }
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'statistics': stats,
            'detailed_results': results
        }, f, indent=2, default=str)
    
    print(f"✓ Results saved to {results_file}")
    
    # Generate visualizations
    if save_visualizations:
        print(f"\nGenerating visualizations...")
        
        # Distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(disruption_scores, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(disruption_scores), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(disruption_scores):.3f}')
        axes[0].set_xlabel('Disruption Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Disruption Score Distribution', fontsize=14, weight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].hist(fragmentation_scores, bins=20, color='coral', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(fragmentation_scores), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(fragmentation_scores):.3f}')
        axes[1].set_xlabel('Fragmentation Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Fragmentation Distribution', fontsize=14, weight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', 'distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Box plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [disruption_scores, fragmentation_scores]
        labels = ['Disruption', 'Fragmentation']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Evaluation Metrics Comparison', fontsize=14, weight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', 'comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved to {output_dir}/figures/")
    
    return stats


def print_summary(stats):
    """Print evaluation summary"""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nDataset:")
    print(f"  Networks Evaluated: {stats['num_networks']}")
    
    print(f"\nDisruption Performance:")
    print(f"  Mean:   {stats['disruption']['mean']:.4f}")
    print(f"  Std:    {stats['disruption']['std']:.4f}")
    print(f"  Median: {stats['disruption']['median']:.4f}")
    print(f"  Range:  [{stats['disruption']['min']:.4f}, {stats['disruption']['max']:.4f}]")
    
    print(f"\nFragmentation Performance:")
    print(f"  Mean:   {stats['fragmentation']['mean']:.4f}")
    print(f"  Std:    {stats['fragmentation']['std']:.4f}")
    print(f"  Median: {stats['fragmentation']['median']:.4f}")
    print(f"  Range:  [{stats['fragmentation']['min']:.4f}, {stats['fragmentation']['max']:.4f}]")
    
    print("\n" + "="*80)


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    
    # Load model
    model, config = load_model(args.model_path, args.config_path, device)
    
    # Generate test data
    networks = generate_test_data(
        args.num_test_networks,
        args.num_timesteps,
        config,
        device
    )
    
    # Initialize evaluators
    detector = CriticalNodeDetector()
    evaluator = DisruptionEvaluator()
    
    # Evaluate model
    results = evaluate_model(
        model, networks, detector, evaluator, top_k=args.top_k
    )
    
    # Generate report
    stats = generate_report(
        results, args.output_dir, save_visualizations=args.save_visualizations
    )
    
    # Print summary
    print_summary(stats)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results: {args.output_dir}/evaluation_results.json")
    if args.save_visualizations:
        print(f"  Figures: {args.output_dir}/figures/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
