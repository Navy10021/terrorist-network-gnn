#!/usr/bin/env python
"""
Result Visualization Script
===========================

Generate publication-quality visualizations from experimental results.

Usage:
    python visualize_results.py --input-dir experiments/phase1 --output-dir visualizations

Author: Yoon-Seop Lee
"""

import argparse
import sys
import os
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Visualize experimental results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing results JSON file')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output format for figures')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output figures')
    
    return parser.parse_args()


def load_results(input_dir):
    """Load experimental results"""
    results_file = os.path.join(input_dir, 'results_phase1.json')
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"✓ Loaded results from {results_file}")
    return results


def plot_training_curves(results, output_dir, fmt='png'):
    """Plot training history"""
    if 'training_history' not in results:
        print("  Skipping training curves (no training history)")
        return
    
    history = results['training_history']
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Total loss
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train', linewidth=2, color='steelblue')
    ax1.plot(epochs, history['val_loss'], label='Validation', linewidth=2, color='coral')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Total Loss', fontsize=11)
    ax1.set_title('Training and Validation Loss', fontsize=12, weight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Loss components
    ax2 = fig.add_subplot(gs[0, 2])
    final_losses = {
        'Link': history['link_loss'][-1],
        'Contrast': history['contrast_loss'][-1],
        'Recon': history['recon_loss'][-1]
    }
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    ax2.bar(final_losses.keys(), final_losses.values(), color=colors, alpha=0.7)
    ax2.set_ylabel('Loss Value', fontsize=10)
    ax2.set_title('Final Loss Components', fontsize=11, weight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Link prediction loss
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, history['link_loss'], color='#2ecc71', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('Link Loss', fontsize=10)
    ax3.set_title('Link Prediction', fontsize=11, weight='bold')
    ax3.grid(alpha=0.3)
    
    # Contrastive loss
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, history['contrast_loss'], color='#e74c3c', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('Contrastive Loss', fontsize=10)
    ax4.set_title('Contrastive Learning', fontsize=11, weight='bold')
    ax4.grid(alpha=0.3)
    
    # Reconstruction loss
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(epochs, history['recon_loss'], color='#3498db', linewidth=2)
    ax5.set_xlabel('Epoch', fontsize=10)
    ax5.set_ylabel('Reconstruction Loss', fontsize=10)
    ax5.set_title('Node Reconstruction', fontsize=11, weight='bold')
    ax5.grid(alpha=0.3)
    
    # Convergence plot
    ax6 = fig.add_subplot(gs[2, :])
    ax6.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2, alpha=0.7)
    ax6.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2, alpha=0.7)
    
    # Find best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_loss = min(history['val_loss'])
    ax6.axvline(best_epoch, color='red', linestyle='--', linewidth=1.5, 
                label=f'Best Epoch: {best_epoch}')
    ax6.scatter([best_epoch], [best_loss], color='red', s=100, zorder=5)
    
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('Loss', fontsize=11)
    ax6.set_title('Training Convergence Analysis', fontsize=12, weight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3)
    
    plt.suptitle('Self-Supervised Training History', fontsize=14, weight='bold', y=0.995)
    
    output_path = os.path.join(output_dir, f'training_history.{fmt}')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Training curves: {output_path}")


def plot_baseline_comparison(results, output_dir, fmt='png'):
    """Plot baseline method comparison"""
    if 'baseline_comparison' not in results:
        print("  Skipping baseline comparison (no data)")
        return
    
    baseline_data = results['baseline_comparison']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data
    methods = list(baseline_data.keys())
    data = [baseline_data[m] for m in methods]
    means = [np.mean(d) for d in data]
    
    # Sort by performance
    sorted_indices = np.argsort(means)[::-1]
    methods = [methods[i] for i in sorted_indices]
    data = [data[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    
    # Box plot
    bp = axes[0, 0].boxplot(data, labels=methods, patch_artist=True)
    for i, (patch, method) in enumerate(zip(bp['boxes'], methods)):
        color = 'gold' if method == 'our_method' else 'steelblue'
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 0].set_ylabel('Disruption Score', fontsize=11)
    axes[0, 0].set_title('Method Performance Distribution', fontsize=12, weight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45, labelsize=9)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Violin plot
    parts = axes[0, 1].violinplot(data, positions=range(len(methods)), showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.7)
    axes[0, 1].set_xticks(range(len(methods)))
    axes[0, 1].set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    axes[0, 1].set_ylabel('Disruption Score', fontsize=11)
    axes[0, 1].set_title('Method Performance Density', fontsize=12, weight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Bar plot with error bars
    stds = [np.std(d) for d in data]
    x_pos = np.arange(len(methods))
    colors = ['gold' if m == 'our_method' else 'steelblue' for m in methods]
    axes[1, 0].bar(x_pos, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_ylabel('Mean Disruption Score', fontsize=11)
    axes[1, 0].set_title('Method Performance with Std Dev', fontsize=12, weight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Heatmap of all runs
    heatmap_data = np.array([baseline_data[m] for m in methods])
    im = axes[1, 1].imshow(heatmap_data, aspect='auto', cmap='RdYlGn')
    axes[1, 1].set_yticks(range(len(methods)))
    axes[1, 1].set_yticklabels(methods, fontsize=9)
    axes[1, 1].set_xlabel('Test Network', fontsize=11)
    axes[1, 1].set_title('Performance Across Test Networks', fontsize=12, weight='bold')
    plt.colorbar(im, ax=axes[1, 1], label='Disruption Score')
    
    plt.suptitle('Baseline Method Comparison', fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'baseline_comparison.{fmt}')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Baseline comparison: {output_path}")


def plot_ablation_study(results, output_dir, fmt='png'):
    """Plot ablation study results"""
    if 'ablation_study' not in results:
        print("  Skipping ablation study (no data)")
        return
    
    ablation_data = results['ablation_study']
    
    # Extract component importance
    components = []
    performance_drops = []
    relative_drops = []
    
    for key, data in ablation_data.items():
        if key != 'full_model' and 'performance_drop' in data:
            components.append(data['name'])
            performance_drops.append(data['performance_drop'])
            relative_drops.append(data['relative_drop_pct'])
    
    # Sort by importance
    sorted_indices = np.argsort(performance_drops)[::-1]
    components = [components[i] for i in sorted_indices]
    performance_drops = [performance_drops[i] for i in sorted_indices]
    relative_drops = [relative_drops[i] for i in sorted_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Horizontal bar chart
    y_pos = np.arange(len(components))
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(components)))
    
    axes[0].barh(y_pos, performance_drops, color=colors, alpha=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(components, fontsize=10)
    axes[0].set_xlabel('Performance Drop (Disruption Score)', fontsize=11)
    axes[0].set_title('Component Importance Ranking', fontsize=12, weight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (y, drop) in enumerate(zip(y_pos, performance_drops)):
        axes[0].text(drop + 0.002, y, f'{drop:.4f}', va='center', fontsize=9)
    
    # Pie chart
    axes[1].pie(performance_drops, labels=components, autopct='%1.1f%%',
               startangle=90, colors=colors)
    axes[1].set_title('Relative Component Contribution', fontsize=12, weight='bold')
    
    plt.suptitle('Ablation Study: Component Importance', fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'ablation_study.{fmt}')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Ablation study: {output_path}")


def plot_statistical_tests(results, output_dir, fmt='png'):
    """Plot statistical significance tests"""
    if 'statistical_tests' not in results or 'baseline_comparison' not in results['statistical_tests']:
        print("  Skipping statistical tests (no data)")
        return
    
    comparison = results['statistical_tests']['baseline_comparison']
    comparisons = comparison['comparisons']
    
    methods = list(comparisons.keys())
    p_values = [comparisons[m]['t_test']['p_value'] for m in methods]
    effect_sizes = [comparisons[m]['cohens_d'] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # P-values
    colors = ['green' if p < 0.05 else 'red' for p in p_values]
    axes[0].barh(methods, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
    axes[0].axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='α=0.05')
    axes[0].set_xlabel('-log₁₀(p-value)', fontsize=11)
    axes[0].set_title('Statistical Significance', fontsize=12, weight='bold')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Effect sizes
    colors = ['darkgreen' if abs(d) > 0.8 else 'orange' if abs(d) > 0.5 else 'gray' 
              for d in effect_sizes]
    axes[1].barh(methods, effect_sizes, color=colors, alpha=0.7)
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].axvline(0.2, color='orange', linestyle='--', alpha=0.5, label='Small')
    axes[1].axvline(0.5, color='green', linestyle='--', alpha=0.5, label='Medium')
    axes[1].axvline(0.8, color='darkgreen', linestyle='--', alpha=0.5, label='Large')
    axes[1].set_xlabel("Cohen's d", fontsize=11)
    axes[1].set_title('Effect Size', fontsize=12, weight='bold')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.suptitle('Statistical Analysis', fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'statistical_tests.{fmt}')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Statistical tests: {output_path}")


def generate_summary_figure(results, output_dir, fmt='png'):
    """Generate comprehensive summary figure"""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Training history (if available)
    if 'training_history' in results:
        history = results['training_history']
        ax1 = fig.add_subplot(gs[0, :2])
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], label='Train', linewidth=2)
        ax1.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress', weight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    # Baseline comparison
    if 'baseline_comparison' in results:
        baseline_data = results['baseline_comparison']
        methods = list(baseline_data.keys())
        means = [np.mean(baseline_data[m]) for m in methods]
        
        sorted_indices = np.argsort(means)[::-1]
        methods = [methods[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        
        ax2 = fig.add_subplot(gs[0, 2])
        colors = ['gold' if m == 'our_method' else 'steelblue' for m in methods]
        ax2.barh(methods, means, color=colors, alpha=0.7)
        ax2.set_xlabel('Mean Score')
        ax2.set_title('Method Ranking', weight='bold')
        ax2.grid(axis='x', alpha=0.3)
    
    # Ablation study
    if 'ablation_study' in results:
        ablation_data = results['ablation_study']
        components = []
        drops = []
        for key, data in ablation_data.items():
            if key != 'full_model' and 'performance_drop' in data:
                components.append(data['name'])
                drops.append(data['performance_drop'])
        
        if components:
            sorted_indices = np.argsort(drops)[::-1]
            components = [components[i] for i in sorted_indices][:5]
            drops = [drops[i] for i in sorted_indices][:5]
            
            ax3 = fig.add_subplot(gs[1, :])
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(components)))
            ax3.barh(components, drops, color=colors, alpha=0.8)
            ax3.set_xlabel('Performance Drop')
            ax3.set_title('Top 5 Important Components', weight='bold')
            ax3.grid(axis='x', alpha=0.3)
    
    # Q1, Q2, Q3 results summary
    if 'q1_critical_nodes' in results:
        q1_data = results['q1_critical_nodes']
        strategies = list(q1_data.keys())
        disruptions = [np.mean([r['disruption'] for r in q1_data[s]]) for s in strategies]
        
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.bar(strategies, disruptions, color='steelblue', alpha=0.7)
        ax4.set_ylabel('Disruption')
        ax4.set_title('Q1: Critical Node Detection', weight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
    
    if 'q2_temporal_prediction' in results:
        q2_data = results['q2_temporal_prediction']
        resilience_scores = [r['resilience_score'] for r in q2_data]
        
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(resilience_scores, bins=15, color='coral', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Resilience Score')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Q2: Temporal Resilience', weight='bold')
        ax5.grid(alpha=0.3)
    
    if 'q3_adversarial_robustness' in results:
        q3_data = results['q3_adversarial_robustness']
        strategies = list(q3_data.keys())
        avg_disruptions = []
        for strategy in strategies:
            all_rounds = [r for net in q3_data[strategy] for r in net]
            avg_disruptions.append(np.mean([r['disruption'] for r in all_rounds]))
        
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.bar(strategies, avg_disruptions, color='green', alpha=0.7)
        ax6.set_ylabel('Avg Disruption')
        ax6.set_title('Q3: Adversarial Robustness', weight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Experimental Results Summary', fontsize=16, weight='bold')
    
    output_path = os.path.join(output_dir, f'summary.{fmt}')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Summary figure: {output_path}")


def main():
    """Main visualization function"""
    args = parse_args()
    
    print("="*80)
    print("RESULT VISUALIZATION")
    print("="*80)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.input_dir)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_training_curves(results, args.output_dir, args.format)
    plot_baseline_comparison(results, args.output_dir, args.format)
    plot_ablation_study(results, args.output_dir, args.format)
    plot_statistical_tests(results, args.output_dir, args.format)
    generate_summary_figure(results, args.output_dir, args.format)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {args.output_dir}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
