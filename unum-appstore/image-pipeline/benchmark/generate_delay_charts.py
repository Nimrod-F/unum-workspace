#!/usr/bin/env python3
"""
Generate visualization charts for the Artificial Delay Benchmark results.

Creates:
1. Bar chart comparing CLASSIC vs FUTURE_BASED latency per scenario
2. Line chart showing improvement percentage across scenarios
3. Stacked bar showing branch timing breakdown
4. Summary table with all metrics

Usage:
    python generate_delay_charts.py delay_benchmark_YYYYMMDD_HHMMSS.json
    python generate_delay_charts.py --latest
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Chart styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'classic': '#E74C3C',      # Red
    'future': '#27AE60',       # Green
    'improvement': '#3498DB',   # Blue
    'neutral': '#95A5A6',       # Gray
    'branches': ['#F39C12', '#9B59B6', '#1ABC9C', '#E74C3C']  # Orange, Purple, Teal, Red
}

BRANCHES = ['Thumbnail', 'Transform', 'Filters', 'Contour']


def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON file"""
    with open(filepath) as f:
        return json.load(f)


def find_latest_results() -> Path:
    """Find the most recent benchmark results file - prefer combined results"""
    benchmark_dir = Path(__file__).parent
    
    # First check for combined results
    combined = benchmark_dir / 'combined_benchmark_results.json'
    if combined.exists():
        return combined
    
    # Otherwise find latest individual file
    files = list(benchmark_dir.glob('delay_benchmark_*.json'))
    if not files:
        raise FileNotFoundError("No benchmark results found")
    return max(files, key=lambda f: f.stat().st_mtime)



def create_latency_comparison_chart(data: dict, output_dir: Path):
    """Create bar chart comparing CLASSIC vs FUTURE latency"""
    scenarios = [s['name'] for s in data['scenarios']]
    classic_latencies = [s['classic']['avg_latency_ms'] for s in data['scenarios']]
    future_latencies = [s['future']['avg_latency_ms'] for s in data['scenarios']]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, classic_latencies, width, label='CLASSIC', 
                   color=COLORS['classic'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, future_latencies, width, label='FUTURE_BASED',
                   color=COLORS['future'], edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Delay Scenario', fontsize=12)
    ax.set_ylabel('End-to-End Latency (ms)', fontsize=12)
    ax.set_title('Image Pipeline: CLASSIC vs FUTURE_BASED Execution\n(Lower is Better)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend()
    
    ax.set_ylim(0, max(classic_latencies) * 1.2)
    
    plt.tight_layout()
    filepath = output_dir / 'latency_comparison.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Created {filepath}")
    return filepath


def create_improvement_chart(data: dict, output_dir: Path):
    """Create chart showing latency improvement percentage"""
    scenarios = [s['name'] for s in data['scenarios']]
    improvements_pct = [s['improvement_pct'] for s in data['scenarios']]
    improvements_ms = [s['improvement_ms'] for s in data['scenarios']]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    
    # Percentage bars
    bars = ax1.bar(x, improvements_pct, color=COLORS['improvement'], 
                   edgecolor='white', linewidth=1, alpha=0.8)
    
    # Add labels
    for i, (bar, ms) in enumerate(zip(bars, improvements_ms)):
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%\n({ms:.0f}ms)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Delay Scenario', fontsize=12)
    ax1.set_ylabel('Latency Improvement (%)', fontsize=12, color=COLORS['improvement'])
    ax1.set_title('Future-Based Execution: Latency Improvement by Scenario\n(Higher is Better)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=15, ha='right')
    ax1.set_ylim(0, max(improvements_pct) * 1.3)
    
    # Add horizontal line at average
    avg_improvement = np.mean(improvements_pct)
    ax1.axhline(y=avg_improvement, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.annotate(f'Average: {avg_improvement:.1f}%', 
                 xy=(len(scenarios)-0.5, avg_improvement),
                 xytext=(10, 0), textcoords="offset points",
                 fontsize=10, color='red')
    
    plt.tight_layout()
    filepath = output_dir / 'improvement_chart.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Created {filepath}")
    return filepath


def create_delay_configuration_chart(data: dict, output_dir: Path):
    """Create chart showing delay configuration for each scenario"""
    scenarios_data = data['scenarios']
    
    fig, axes = plt.subplots(1, len(scenarios_data), figsize=(15, 5), sharey=True)
    
    if len(scenarios_data) == 1:
        axes = [axes]
    
    for idx, (ax, scenario) in enumerate(zip(axes, scenarios_data)):
        delays = scenario['delays']
        branches = list(delays.keys())
        delay_values = [delays[b] for b in branches]
        
        colors = COLORS['branches'][:len(branches)]
        bars = ax.barh(branches, delay_values, color=colors, edgecolor='white')
        
        ax.set_xlabel('Artificial Delay (ms)')
        ax.set_title(scenario['name'], fontsize=11, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, delay_values):
            ax.annotate(f'{val}ms',
                       xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                       xytext=(3, 0), textcoords="offset points",
                       ha='left', va='center', fontsize=9)
    
    fig.suptitle('Delay Configuration per Scenario', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    filepath = output_dir / 'delay_configurations.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Created {filepath}")
    return filepath


def create_timing_breakdown_chart(data: dict, output_dir: Path):
    """Create stacked bar chart showing branch timing breakdown"""
    # Get first scenario for detailed breakdown
    scenario = data['scenarios'][0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, mode in zip(axes, ['classic', 'future']):
        runs = scenario[mode]['runs']
        
        # Average branch durations across runs
        branch_avgs = {b: [] for b in BRANCHES}
        for run in runs:
            if 'branch_durations' in run:
                for b in BRANCHES:
                    if b in run['branch_durations']:
                        branch_avgs[b].append(run['branch_durations'][b])
        
        branch_avgs = {b: np.mean(v) if v else 0 for b, v in branch_avgs.items()}
        
        # Create stacked bars
        bottom = 0
        for i, branch in enumerate(BRANCHES):
            ax.bar(['Branches'], [branch_avgs[branch]], bottom=[bottom],
                  color=COLORS['branches'][i], label=branch, edgecolor='white')
            
            # Add label if significant
            if branch_avgs[branch] > 50:
                ax.text(0, bottom + branch_avgs[branch]/2, 
                       f'{branch}\n{branch_avgs[branch]:.0f}ms',
                       ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            
            bottom += branch_avgs[branch]
        
        mode_label = mode.upper() if mode == 'classic' else 'FUTURE_BASED'
        ax.set_title(f'{mode_label}\nTotal: {scenario[mode]["avg_latency_ms"]:.0f}ms', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Duration (ms)')
        ax.set_xlim(-0.5, 0.5)
    
    # Single legend for both
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05))
    
    fig.suptitle(f'Branch Timing Breakdown: {scenario["name"]}', 
                 fontsize=14, fontweight='bold', y=1.1)
    
    plt.tight_layout()
    filepath = output_dir / 'timing_breakdown.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Created {filepath}")
    return filepath


def create_summary_visualization(data: dict, output_dir: Path):
    """Create comprehensive summary visualization"""
    fig = plt.figure(figsize=(16, 12))
    
    # Grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    scenarios = data['scenarios']
    scenario_names = [s['name'] for s in scenarios]
    
    # 1. Latency comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(scenario_names))
    width = 0.35
    
    classic = [s['classic']['avg_latency_ms'] for s in scenarios]
    future = [s['future']['avg_latency_ms'] for s in scenarios]
    
    ax1.bar(x - width/2, classic, width, label='CLASSIC', color=COLORS['classic'])
    ax1.bar(x + width/2, future, width, label='FUTURE_BASED', color=COLORS['future'])
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('End-to-End Latency Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, rotation=20, ha='right', fontsize=9)
    ax1.legend()
    
    # 2. Improvement percentage (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    improvements = [s['improvement_pct'] for s in scenarios]
    colors = [COLORS['future'] if p > 0 else COLORS['classic'] for p in improvements]
    bars = ax2.bar(scenario_names, improvements, color=colors)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Latency Improvement (FUTURE vs CLASSIC)', fontweight='bold')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha='right', fontsize=9)
    
    for bar, val in zip(bars, improvements):
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # 3. Theoretical vs Actual savings (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    # Calculate theoretical max savings if not present
    theoretical = []
    for s in scenarios:
        if 'theoretical_max_savings_ms' in s:
            theoretical.append(s['theoretical_max_savings_ms'])
        else:
            # Calculate: max delay - min delay
            delays = list(s['delays'].values())
            theoretical.append(max(delays) - min(delays))
    actual = [s['improvement_ms'] for s in scenarios]
    
    x = np.arange(len(scenario_names))
    ax3.bar(x - width/2, theoretical, width, label='Theoretical Max', color=COLORS['neutral'])
    ax3.bar(x + width/2, actual, width, label='Actual', color=COLORS['improvement'])
    ax3.set_ylabel('Savings (ms)')
    ax3.set_title('Theoretical vs Actual Savings', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenario_names, rotation=20, ha='right', fontsize=9)
    ax3.legend()
    
    # 4. Delay configurations (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Stacked bars for delay config
    bottom = np.zeros(len(scenarios))
    for i, branch in enumerate(BRANCHES):
        delays = [s['delays'].get(branch, 0) for s in scenarios]
        ax4.bar(scenario_names, delays, bottom=bottom, label=branch, 
               color=COLORS['branches'][i])
        bottom += delays
    
    ax4.set_ylabel('Total Delay (ms)')
    ax4.set_title('Delay Configuration per Scenario', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=20, ha='right', fontsize=9)
    
    # 5. Summary text (bottom spanning both columns)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Calculate summary statistics
    avg_improvement_pct = np.mean([s['improvement_pct'] for s in scenarios])
    avg_improvement_ms = np.mean([s['improvement_ms'] for s in scenarios])
    max_improvement = max(scenarios, key=lambda s: s['improvement_pct'])
    min_improvement = min(scenarios, key=lambda s: s['improvement_pct'])
    
    # Handle both data formats (config may or may not exist)
    config = data.get('config', {})
    region = config.get('region', 'eu-central-1')
    test_image = config.get('test_image', 'N/A')
    
    summary_text = f"""
    BENCHMARK SUMMARY
    {'='*60}
    
    Test Configuration:
    • Region: {region}
    • Scenarios Tested: {len(scenarios)}
    
    Key Findings:
    • Average Improvement: {avg_improvement_pct:.1f}% ({avg_improvement_ms:.0f}ms)
    • Best Scenario: {max_improvement['name']} ({max_improvement['improvement_pct']:.1f}% improvement)
    • Worst Scenario: {min_improvement['name']} ({min_improvement['improvement_pct']:.1f}% improvement)
    
    Conclusion:
    Future-Based execution provides significant latency improvements when branch
    execution times vary. The larger the variance between branches, the greater
    the benefit of Future-Based execution.
    """
    
    ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Image Pipeline: Artificial Delay Benchmark Results', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath = output_dir / 'summary_visualization.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Created {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Generate charts from delay benchmark results')
    parser.add_argument('results_file', nargs='?', help='Path to benchmark results JSON')
    parser.add_argument('--latest', action='store_true', help='Use most recent results file')
    parser.add_argument('--output-dir', type=str, help='Output directory for charts')
    
    args = parser.parse_args()
    
    # Find results file
    if args.latest or not args.results_file:
        try:
            results_path = find_latest_results()
            print(f"Using latest results: {results_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    else:
        results_path = Path(args.results_file)
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return
    
    # Load data
    data = load_results(results_path)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / 'charts'
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating charts in: {output_dir}")
    print("=" * 50)
    
    # Generate charts
    create_latency_comparison_chart(data, output_dir)
    create_improvement_chart(data, output_dir)
    create_delay_configuration_chart(data, output_dir)
    
    if data['scenarios']:
        create_timing_breakdown_chart(data, output_dir)
    
    create_summary_visualization(data, output_dir)
    
    print("=" * 50)
    print(f"✓ All charts generated successfully")


if __name__ == "__main__":
    main()
