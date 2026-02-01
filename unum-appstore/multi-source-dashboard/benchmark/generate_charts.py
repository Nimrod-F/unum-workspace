#!/usr/bin/env python3
"""
Chart Generator for Multi-Source Dashboard Benchmark Results

Generates visualization charts from benchmark results JSON file.
Creates charts comparing CLASSIC, EAGER, and FUTURE_BASED execution modes.

Usage:
    python generate_charts.py --results-file results/benchmark_20260201_205450.json
    python generate_charts.py --results-dir results/
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_results(results_file: Path) -> Dict:
    """Load benchmark results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_e2e_latency_comparison(data: Dict, output_dir: Path):
    """
    Bar chart comparing E2E latency across modes with error bars.
    """
    modes = []
    means = []
    stds = []
    colors = []

    color_map = {
        'CLASSIC': '#e74c3c',      # Red
        'EAGER': '#f39c12',         # Orange
        'FUTURE_BASED': '#27ae60'   # Green
    }

    for mode in ['CLASSIC', 'EAGER', 'FUTURE_BASED']:
        if mode in data:
            summary = data[mode]['summary']
            modes.append(mode.replace('_', '\n'))
            means.append(summary['e2e_mean_ms'])
            stds.append(summary['e2e_std_ms'])
            colors.append(color_map.get(mode, '#95a5a6'))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(modes))
    bars = ax.bar(x, means, yerr=stds, capsize=10, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 200,
                f'{mean:.0f}ms\n±{std:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Execution Mode', fontsize=12, fontweight='bold')
    ax.set_ylabel('End-to-End Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Source Dashboard: E2E Latency Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'e2e_latency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  OK Created: e2e_latency_comparison.png")
    plt.close()


def plot_cold_vs_warm(data: Dict, output_dir: Path):
    """
    Grouped bar chart comparing cold vs warm start latencies.
    """
    modes = []
    cold_means = []
    warm_means = []

    for mode in ['CLASSIC', 'EAGER', 'FUTURE_BASED']:
        if mode in data:
            summary = data[mode]['summary']
            modes.append(mode.replace('_', '\n'))
            cold_means.append(summary['cold_mean_ms'])
            warm_means.append(summary['warm_mean_ms'])

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, cold_means, width, label='Cold Start',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, warm_means, width, label='Warm Start',
                   color='#3498db', alpha=0.8, edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                    f'{height:.0f}ms',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Execution Mode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Cold Start vs Warm Start Impact',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'cold_vs_warm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  OK Created: cold_vs_warm_comparison.png")
    plt.close()


def plot_improvement_chart(data: Dict, output_dir: Path):
    """
    Bar chart showing improvement percentage vs CLASSIC baseline.
    """
    if 'CLASSIC' not in data:
        print("  ⚠ Skipping improvement chart: CLASSIC baseline not found")
        return

    baseline = data['CLASSIC']['summary']['e2e_mean_ms']
    modes = []
    improvements = []
    colors = []

    for mode in ['EAGER', 'FUTURE_BASED']:
        if mode in data:
            mean = data[mode]['summary']['e2e_mean_ms']
            improvement = ((baseline - mean) / baseline) * 100
            modes.append(mode.replace('_', '\n'))
            improvements.append(improvement)
            colors.append('#27ae60' if improvement > 0 else '#e74c3c')

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(modes, improvements, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        label = f'{improvement:+.1f}%'
        va = 'bottom' if height >= 0 else 'top'
        y_offset = 1 if height >= 0 else -1
        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                label, ha='center', va=va, fontsize=12, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Improvement vs CLASSIC (%)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Improvement Over CLASSIC Baseline',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_over_classic.png', dpi=300, bbox_inches='tight')
    print(f"  OK Created: improvement_over_classic.png")
    plt.close()


def plot_pre_resolved_efficiency(data: Dict, output_dir: Path):
    """
    Bar chart showing average pre-resolved input count for FUTURE mode.
    """
    if 'FUTURE_BASED' not in data:
        print("  ⚠ Skipping pre-resolved chart: FUTURE_BASED data not found")
        return

    avg_pre_resolved = data['FUTURE_BASED']['summary']['avg_pre_resolved']
    total_inputs = 6  # Multi-source dashboard has 6 parallel inputs

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Pre-Resolved\n(Background Cached)',
                  'Blocked\n(Had to Wait)']
    values = [avg_pre_resolved, total_inputs - avg_pre_resolved]
    colors = ['#27ae60', '#e74c3c']

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        percentage = (value / total_inputs) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=total_inputs, color='gray', linestyle='--',
               linewidth=1, alpha=0.5, label=f'Total Inputs ({total_inputs})')
    ax.set_ylabel('Number of Inputs', fontsize=12, fontweight='bold')
    ax.set_title('FUTURE_BASED: Background Polling Efficiency\n(out of 6 parallel inputs)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, total_inputs + 1)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'pre_resolved_efficiency.png', dpi=300, bbox_inches='tight')
    print(f"  OK Created: pre_resolved_efficiency.png")
    plt.close()


def plot_box_whisker(data: Dict, output_dir: Path):
    """
    Box and whisker plot showing latency distribution for each mode.
    """
    modes_data = []
    mode_labels = []
    colors = []

    color_map = {
        'CLASSIC': '#e74c3c',
        'EAGER': '#f39c12',
        'FUTURE_BASED': '#27ae60'
    }

    for mode in ['CLASSIC', 'EAGER', 'FUTURE_BASED']:
        if mode in data:
            runs = data[mode]['runs']
            latencies = [run['e2e_latency_ms'] for run in runs if run['success']]
            if latencies:
                modes_data.append(latencies)
                mode_labels.append(mode.replace('_', '\n'))
                colors.append(color_map.get(mode, '#95a5a6'))

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(modes_data, tick_labels=mode_labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(color='darkred', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2, linestyle='--'))

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('End-to-End Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Distribution Across Execution Modes',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add legend
    median_patch = mpatches.Patch(color='darkred', label='Median')
    mean_patch = mpatches.Patch(color='blue', label='Mean')
    ax.legend(handles=[median_patch, mean_patch], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  OK Created: latency_distribution.png")
    plt.close()


def create_summary_table(data: Dict, output_dir: Path):
    """
    Create a text summary table and save as image.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    headers = ['Metric', 'CLASSIC', 'EAGER', 'FUTURE_BASED']
    rows = []

    metrics = [
        ('E2E Mean (ms)', 'e2e_mean_ms', ':.0f'),
        ('E2E Std Dev (ms)', 'e2e_std_ms', ':.0f'),
        ('Cold Start (ms)', 'cold_mean_ms', ':.0f'),
        ('Warm Start (ms)', 'warm_mean_ms', ':.0f'),
        ('Min Latency (ms)', 'e2e_min_ms', ':.0f'),
        ('Max Latency (ms)', 'e2e_max_ms', ':.0f'),
        ('Successful Runs', 'successful_runs', ':d'),
        ('Pre-Resolved Avg', 'avg_pre_resolved', ':.1f'),
    ]

    for metric_name, key, fmt in metrics:
        row = [metric_name]
        for mode in ['CLASSIC', 'EAGER', 'FUTURE_BASED']:
            if mode in data:
                value = data[mode]['summary'].get(key, 0)
                # Format the value based on the format spec
                if fmt == ':.0f':
                    row.append(f'{value:.0f}')
                elif fmt == ':.1f':
                    row.append(f'{value:.1f}')
                elif fmt == ':d':
                    row.append(f'{int(value):d}')
                else:
                    row.append(str(value))
            else:
                row.append('N/A')
        rows.append(row)

    # Calculate improvement
    if 'CLASSIC' in data and 'FUTURE_BASED' in data:
        classic_mean = data['CLASSIC']['summary']['e2e_mean_ms']
        future_mean = data['FUTURE_BASED']['summary']['e2e_mean_ms']
        improvement = ((classic_mean - future_mean) / classic_mean) * 100
        rows.append(['', '', '', ''])
        rows.append(['Improvement vs CLASSIC', '—',
                    f'{((classic_mean - data.get("EAGER", {}).get("summary", {}).get("e2e_mean_ms", 0)) / classic_mean) * 100:+.1f}%' if 'EAGER' in data else 'N/A',
                    f'{improvement:+.1f}%'])

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                    cellLoc='center', colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('#ffffff')

    plt.title('Multi-Source Dashboard Benchmark Summary',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    print(f"  OK Created: summary_table.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate charts from multi-source-dashboard benchmark results'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--results-file', type=str,
                      help='Path to benchmark results JSON file')
    group.add_argument('--results-dir', type=str,
                      help='Directory containing results (uses latest file)')
    parser.add_argument('--output-dir', type=str, default='charts',
                      help='Output directory for charts (default: charts/)')

    args = parser.parse_args()

    # Determine results file
    if args.results_file:
        results_file = Path(args.results_file)
    else:
        results_dir = Path(args.results_dir)
        json_files = sorted(results_dir.glob('benchmark_*.json'),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not json_files:
            print(f"Error: No benchmark_*.json files found in {results_dir}")
            return
        results_file = json_files[0]

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return

    print(f"\nGenerating charts from: {results_file}")

    # Load data
    data = load_results(results_file)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Generate all charts
    print("Generating charts:")
    plot_e2e_latency_comparison(data, output_dir)
    plot_cold_vs_warm(data, output_dir)
    plot_improvement_chart(data, output_dir)
    plot_pre_resolved_efficiency(data, output_dir)
    plot_box_whisker(data, output_dir)
    create_summary_table(data, output_dir)

    print(f"\nOK All charts generated successfully in: {output_dir}")
    print(f"\nGenerated files:")
    for chart_file in sorted(output_dir.glob('*.png')):
        print(f"  - {chart_file.name}")


if __name__ == '__main__':
    main()
