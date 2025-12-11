"""Plot learning curves comparing AUC vs wall-clock time.

Generates publication-quality figure showing convergence rates of different
unlearning methods.

Usage:
    python src/plot_learning_curves.py \
        --fisher_path outputs/p2/fisher_lc/learning_curve.json \
        --retrain_path outputs/p2/retrain_lc/learning_curve.json \
        --output_path outputs/p2/learning_curves.png
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_learning_curve(path: str) -> dict:
    """Load learning curve JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_learning_curves(
    fisher_data: dict,
    retrain_data: dict,
    output_path: str,
    retrain_floor_auc: float = 0.864,
    baseline_auc: float = 0.951,
    figsize: tuple = (10, 6)
):
    """Plot AUC vs wall-clock time for Fisher and retrain methods.

    Args:
        fisher_data: Fisher learning curve data
        retrain_data: Retrain learning curve data
        output_path: Output path for figure
        retrain_floor_auc: Target AUC (retrain floor)
        baseline_auc: Baseline AUC (before unlearning)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data from Fisher
    fisher_times = [r['wall_clock_seconds'] for r in fisher_data['records']]
    fisher_aucs = [r['auc'] for r in fisher_data['records']]
    fisher_phases = [r['phase'] for r in fisher_data['records']]

    # Extract data from retrain
    retrain_times = [r['wall_clock_seconds'] for r in retrain_data['records']]
    retrain_aucs = [r['auc'] for r in retrain_data['records']]

    # Plot Fisher with phase markers
    # Split by phase for different markers
    scrub_mask = [p == 'scrub' for p in fisher_phases]
    finetune_mask = [p == 'finetune' for p in fisher_phases]

    scrub_times = [t for t, m in zip(fisher_times, scrub_mask) if m]
    scrub_aucs = [a for a, m in zip(fisher_aucs, scrub_mask) if m]

    finetune_times = [t for t, m in zip(fisher_times, finetune_mask) if m]
    finetune_aucs = [a for a, m in zip(fisher_aucs, finetune_mask) if m]

    # Plot Fisher scrub phase
    ax.plot(scrub_times, scrub_aucs, 'o-', color='#2ecc71', linewidth=2,
            markersize=6, label='Fisher (scrub phase)')

    # Plot Fisher finetune phase
    if finetune_times:
        ax.plot(finetune_times, finetune_aucs, 's-', color='#27ae60', linewidth=2,
                markersize=6, label='Fisher (finetune phase)')

    # Plot retrain
    ax.plot(retrain_times, retrain_aucs, '^-', color='#3498db', linewidth=2,
            markersize=6, label='Retrain from scratch')

    # Add reference lines
    ax.axhline(y=retrain_floor_auc, color='#e74c3c', linestyle='--',
               linewidth=1.5, label=f'Retrain floor (AUC={retrain_floor_auc:.3f})')
    ax.axhline(y=baseline_auc, color='#95a5a6', linestyle=':',
               linewidth=1.5, label=f'Baseline (AUC={baseline_auc:.3f})')

    # Add target band
    target_low = retrain_floor_auc - 0.03
    target_high = retrain_floor_auc + 0.03
    ax.axhspan(target_low, target_high, alpha=0.15, color='#e74c3c',
               label='Target band (floor +/- 0.03)')

    # Annotations
    fisher_final_time = fisher_times[-1] if fisher_times else 0
    fisher_final_auc = fisher_aucs[-1] if fisher_aucs else 0
    retrain_final_time = retrain_times[-1] if retrain_times else 0
    retrain_final_auc = retrain_aucs[-1] if retrain_aucs else 0

    # Mark final points
    ax.scatter([fisher_final_time], [fisher_final_auc], s=150, color='#27ae60',
               marker='*', zorder=5, edgecolors='black', linewidths=1)
    ax.scatter([retrain_final_time], [retrain_final_auc], s=150, color='#3498db',
               marker='*', zorder=5, edgecolors='black', linewidths=1)

    # Add time annotations
    ax.annotate(f'{fisher_final_time:.0f}s', (fisher_final_time, fisher_final_auc),
                textcoords='offset points', xytext=(10, -15), fontsize=10,
                color='#27ae60', fontweight='bold')
    ax.annotate(f'{retrain_final_time:.0f}s', (retrain_final_time, retrain_final_auc),
                textcoords='offset points', xytext=(10, 10), fontsize=10,
                color='#3498db', fontweight='bold')

    # Style
    ax.set_xlabel('Wall-clock time (seconds)', fontsize=12)
    ax.set_ylabel('MIA AUC', fontsize=12)
    ax.set_title('Learning Curves: AUC vs Wall-Clock Time', fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y limits to show full range
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0, max(fisher_final_time, retrain_final_time) * 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved learning curves to {output_path}")

    # Print summary
    print("\nSummary:")
    print(f"  Fisher: {fisher_final_time:.1f}s to reach AUC={fisher_final_auc:.4f}")
    print(f"  Retrain: {retrain_final_time:.1f}s to reach AUC={retrain_final_auc:.4f}")
    if retrain_final_time > 0:
        print(f"  Speedup: {retrain_final_time / fisher_final_time:.1f}x")


def plot_combined_with_inset(
    fisher_data: dict,
    retrain_data: dict,
    output_path: str,
    retrain_floor_auc: float = 0.864,
    baseline_auc: float = 0.951,
    figsize: tuple = (12, 7)
):
    """Plot with inset showing early convergence behavior."""
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    fisher_times = [r['wall_clock_seconds'] for r in fisher_data['records']]
    fisher_aucs = [r['auc'] for r in fisher_data['records']]

    retrain_times = [r['wall_clock_seconds'] for r in retrain_data['records']]
    retrain_aucs = [r['auc'] for r in retrain_data['records']]

    # Main plot
    ax.plot(fisher_times, fisher_aucs, 'o-', color='#2ecc71', linewidth=2,
            markersize=5, label='Fisher unlearning')
    ax.plot(retrain_times, retrain_aucs, '^-', color='#3498db', linewidth=2,
            markersize=5, label='Retrain from scratch')

    # Reference lines
    ax.axhline(y=retrain_floor_auc, color='#e74c3c', linestyle='--',
               linewidth=1.5, label=f'Target (AUC={retrain_floor_auc:.3f})')
    ax.axhline(y=baseline_auc, color='#95a5a6', linestyle=':',
               linewidth=1.5, label=f'Baseline (AUC={baseline_auc:.3f})')

    ax.set_xlabel('Wall-clock time (seconds)', fontsize=12)
    ax.set_ylabel('MIA AUC', fontsize=12)
    ax.set_title('Convergence Comparison: Fisher vs Retrain', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    # Inset for early behavior (first 200s)
    axins = ax.inset_axes([0.15, 0.15, 0.35, 0.35])

    # Filter to first 200s
    early_fisher_mask = [t <= 200 for t in fisher_times]
    early_retrain_mask = [t <= 200 for t in retrain_times]

    early_fisher_times = [t for t, m in zip(fisher_times, early_fisher_mask) if m]
    early_fisher_aucs = [a for a, m in zip(fisher_aucs, early_fisher_mask) if m]

    early_retrain_times = [t for t, m in zip(retrain_times, early_retrain_mask) if m]
    early_retrain_aucs = [a for a, m in zip(retrain_aucs, early_retrain_mask) if m]

    axins.plot(early_fisher_times, early_fisher_aucs, 'o-', color='#2ecc71',
               linewidth=2, markersize=4)
    axins.plot(early_retrain_times, early_retrain_aucs, '^-', color='#3498db',
               linewidth=2, markersize=4)
    axins.axhline(y=retrain_floor_auc, color='#e74c3c', linestyle='--', linewidth=1)

    axins.set_xlim(0, 200)
    axins.set_ylim(0.8, 1.0)
    axins.set_xlabel('Time (s)', fontsize=9)
    axins.set_ylabel('AUC', fontsize=9)
    axins.set_title('First 200s', fontsize=9)
    axins.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved combined plot with inset to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot learning curves')
    parser.add_argument('--fisher_path', type=str, required=True,
                        help='Path to Fisher learning curve JSON')
    parser.add_argument('--retrain_path', type=str, required=True,
                        help='Path to retrain learning curve JSON')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for figure')
    parser.add_argument('--retrain_floor_auc', type=float, default=0.864,
                        help='Retrain floor AUC')
    parser.add_argument('--baseline_auc', type=float, default=0.951,
                        help='Baseline AUC')
    parser.add_argument('--with_inset', action='store_true',
                        help='Generate combined plot with inset')

    args = parser.parse_args()

    # Load data
    fisher_data = load_learning_curve(args.fisher_path)
    retrain_data = load_learning_curve(args.retrain_path)

    # Create output directory if needed
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Plot
    if args.with_inset:
        plot_combined_with_inset(
            fisher_data, retrain_data, args.output_path,
            args.retrain_floor_auc, args.baseline_auc
        )
    else:
        plot_learning_curves(
            fisher_data, retrain_data, args.output_path,
            args.retrain_floor_auc, args.baseline_auc
        )


if __name__ == '__main__':
    main()
