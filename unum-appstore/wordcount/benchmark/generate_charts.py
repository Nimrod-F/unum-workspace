#!/usr/bin/env python3
"""Generate benchmark comparison charts for Wordcount CLASSIC vs FUTURE_BASED modes."""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

# Create results directory if not exists
os.makedirs('results', exist_ok=True)


def load_latest_results(results_dir: Path) -> dict:
    """Load the most recent benchmark summary files."""
    data = {}
    
    for mode in ['CLASSIC', 'FUTURE_BASED']:
        # Find latest summary file for each mode
        pattern = f'benchmark_{mode}_*_summary.json'
        files = sorted(results_dir.glob(pattern), reverse=True)
        
        if files:
            with open(files[0]) as f:
                summary = json.load(f)
                
                # Determine if cold or warm based on cold_start_rate
                run_type = 'cold' if summary.get('cold_start_rate', 0) > 0.5 else 'warm'
                
                if mode not in data:
                    data[mode] = {}
                data[mode][run_type] = summary
    
    return data


def create_e2e_comparison_chart(data: dict, output_dir: Path):
    """Create E2E latency comparison bar chart for all scenarios."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    scenarios = []
    latencies = []
    std_devs = []
    colors = []
    
    for mode in ['CLASSIC', 'FUTURE_BASED']:
        for run_type in ['cold', 'warm']:
            if mode in data and run_type in data[mode]:
                summary = data[mode][run_type]
                scenarios.append(f"{mode}\n{run_type.title()}")
                latencies.append(summary.get('e2e_latency_mean_ms', 0))
                std_devs.append(summary.get('e2e_latency_std_ms', 0))
                colors.append('#e74c3c' if mode == 'CLASSIC' else '#27ae60')
    
    if not scenarios:
        print("No data to plot for E2E comparison")
        return
    
    bars = ax.bar(scenarios, latencies, yerr=std_devs, capsize=8, 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('End-to-End Latency (ms)', fontsize=12)
    ax.set_title('Wordcount: E2E Latency Comparison\nCLASSIC vs FUTURE_BASED', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, val, std in zip(bars, latencies, std_devs):
        ax.annotate(f'{val:.0f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 50),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='CLASSIC'),
        Patch(facecolor='#27ae60', edgecolor='black', label='FUTURE_BASED')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'e2e_latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'e2e_latency_comparison.png'}")


def create_cold_vs_warm_chart(data: dict, output_dir: Path):
    """Create cold vs warm impact comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modes = ['CLASSIC', 'FUTURE_BASED']
    x = np.arange(len(modes))
    width = 0.35
    
    cold_latencies = []
    warm_latencies = []
    
    for mode in modes:
        if mode in data:
            cold_latencies.append(data[mode].get('cold', {}).get('e2e_latency_mean_ms', 0))
            warm_latencies.append(data[mode].get('warm', {}).get('e2e_latency_mean_ms', 0))
        else:
            cold_latencies.append(0)
            warm_latencies.append(0)
    
    bars1 = ax.bar(x - width/2, cold_latencies, width, label='Cold Start', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, warm_latencies, width, label='Warm Start', color='#f39c12', edgecolor='black')
    
    ax.set_ylabel('E2E Latency (ms)', fontsize=12)
    ax.set_title('Wordcount: Cold Start Impact by Mode', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            if bar.get_height() > 0:
                ax.annotate(f'{bar.get_height():.0f}ms',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cold_vs_warm_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'cold_vs_warm_impact.png'}")


def create_cost_chart(data: dict, output_dir: Path):
    """Create cost comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = []
    costs = []
    colors = []
    
    for mode in ['CLASSIC', 'FUTURE_BASED']:
        for run_type in ['cold', 'warm']:
            if mode in data and run_type in data[mode]:
                summary = data[mode][run_type]
                scenarios.append(f"{mode[:7]}\n{run_type.title()}")
                costs.append(summary.get('cost_per_run', 0) * 1e5)  # Convert to 10^-5
                colors.append('#e74c3c' if mode == 'CLASSIC' else '#27ae60')
    
    if not scenarios:
        print("No data to plot for cost comparison")
        return
    
    bars = ax.bar(scenarios, costs, color=colors, edgecolor='black')
    ax.set_ylabel('Cost per Run ($ × 10⁻⁵)', fontsize=12)
    ax.set_title('Wordcount: Cost Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar, cost in zip(bars, costs):
        ax.annotate(f'${cost:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cost_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'cost_comparison.png'}")


def create_dashboard(data: dict, output_dir: Path):
    """Create comprehensive 4-panel dashboard."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Wordcount MapReduce Benchmark Dashboard\nCLASSIC vs FUTURE_BASED', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Collect data
    all_scenarios = []
    for mode in ['CLASSIC', 'FUTURE_BASED']:
        for run_type in ['cold', 'warm']:
            if mode in data and run_type in data[mode]:
                all_scenarios.append((mode, run_type, data[mode][run_type]))
    
    if not all_scenarios:
        print("No data for dashboard")
        return
    
    # 1. E2E Latency (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    labels = [f"{m[:5]}\n{t}" for m, t, _ in all_scenarios]
    e2e_vals = [s.get('e2e_latency_mean_ms', 0) for _, _, s in all_scenarios]
    colors = ['#e74c3c' if m == 'CLASSIC' else '#27ae60' for m, _, _ in all_scenarios]
    
    bars1 = ax1.bar(labels, e2e_vals, color=colors, edgecolor='black')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('E2E Latency', fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars1:
        ax1.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9)
    
    # 2. Cold Start Rate (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    cold_rates = [s.get('cold_start_rate', 0) * 100 for _, _, s in all_scenarios]
    bars2 = ax2.bar(labels, cold_rates, color=colors, edgecolor='black')
    ax2.set_ylabel('Cold Start Rate (%)')
    ax2.set_title('Cold Start Rate', fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        ax2.annotate(f'{bar.get_height():.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9)
    
    # 3. Billed Duration (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    billed = [s.get('billed_duration_mean_ms', 0) for _, _, s in all_scenarios]
    bars3 = ax3.bar(labels, billed, color=colors, edgecolor='black')
    ax3.set_ylabel('Billed Duration (ms)')
    ax3.set_title('Total Billed Duration', fontweight='bold')
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars3:
        ax3.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9)
    
    # 4. Cost (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    costs = [s.get('cost_per_run', 0) * 1000 for _, _, s in all_scenarios]  # Convert to milli-dollars
    bars4 = ax4.bar(labels, costs, color=colors, edgecolor='black')
    ax4.set_ylabel('Cost per Run (m$)')
    ax4.set_title('Cost per Run', fontweight='bold')
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars4:
        ax4.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'benchmark_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'benchmark_dashboard.png'}")


def create_workflow_comparison_chart(data: dict, output_dir: Path):
    """Create workflow-specific comparison showing data scale impact."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get data scale info from first available summary
    num_mappers = 0
    words_per_mapper = 0
    total_words = 0
    
    for mode in data:
        for run_type in data[mode]:
            summary = data[mode][run_type]
            num_mappers = summary.get('num_mappers', 0)
            words_per_mapper = summary.get('words_per_mapper', 0)
            total_words = summary.get('total_words', 0)
            if num_mappers > 0:
                break
        if num_mappers > 0:
            break
    
    fig.suptitle(f'Wordcount: {num_mappers} Mappers × {words_per_mapper} Words = {total_words:,} Total',
                 fontsize=14, fontweight='bold')
    
    # Left: Throughput comparison
    ax1 = axes[0]
    throughputs = []
    labels = []
    colors = []
    
    for mode in ['CLASSIC', 'FUTURE_BASED']:
        for run_type in ['cold', 'warm']:
            if mode in data and run_type in data[mode]:
                summary = data[mode][run_type]
                e2e = summary.get('e2e_latency_mean_ms', 0)
                if e2e > 0:
                    throughput = (total_words / (e2e / 1000))  # words per second
                    throughputs.append(throughput)
                    labels.append(f"{mode[:5]}\n{run_type}")
                    colors.append('#e74c3c' if mode == 'CLASSIC' else '#27ae60')
    
    if throughputs:
        bars = ax1.bar(labels, throughputs, color=colors, edgecolor='black')
        ax1.set_ylabel('Throughput (words/sec)')
        ax1.set_title('Processing Throughput', fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        for bar in bars:
            ax1.annotate(f'{bar.get_height():,.0f}', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # Right: Latency per word
    ax2 = axes[1]
    latencies_per_word = []
    labels2 = []
    colors2 = []
    
    for mode in ['CLASSIC', 'FUTURE_BASED']:
        for run_type in ['cold', 'warm']:
            if mode in data and run_type in data[mode]:
                summary = data[mode][run_type]
                e2e = summary.get('e2e_latency_mean_ms', 0)
                if total_words > 0 and e2e > 0:
                    lat_per_word = e2e / total_words  # ms per word
                    latencies_per_word.append(lat_per_word)
                    labels2.append(f"{mode[:5]}\n{run_type}")
                    colors2.append('#e74c3c' if mode == 'CLASSIC' else '#27ae60')
    
    if latencies_per_word:
        bars2 = ax2.bar(labels2, latencies_per_word, color=colors2, edgecolor='black')
        ax2.set_ylabel('Latency per Word (ms)')
        ax2.set_title('Processing Efficiency', fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        for bar in bars2:
            ax2.annotate(f'{bar.get_height():.4f}', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'workflow_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'workflow_comparison.png'}")


def main():
    """Generate all charts from latest benchmark results."""
    print("=" * 60)
    print("📊 Generating Wordcount Benchmark Charts")
    print("=" * 60)
    
    script_dir = Path(__file__).parent.resolve()
    results_dir = script_dir / 'results'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run benchmarks first: python run_benchmark.py --all")
        return
    
    # Load latest results
    data = load_latest_results(results_dir)
    
    if not data:
        print("No benchmark results found in results/")
        return
    
    print(f"\nFound results for: {list(data.keys())}")
    for mode in data:
        print(f"  {mode}: {list(data[mode].keys())}")
    
    print("\nGenerating charts...")
    
    # Generate charts
    create_e2e_comparison_chart(data, results_dir)
    create_cold_vs_warm_chart(data, results_dir)
    create_cost_chart(data, results_dir)
    create_dashboard(data, results_dir)
    create_workflow_comparison_chart(data, results_dir)
    
    print("\n" + "=" * 60)
    print(f"✅ All charts saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
