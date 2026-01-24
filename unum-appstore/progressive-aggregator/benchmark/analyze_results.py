#!/usr/bin/env python3
"""
Benchmark Analysis and Visualization Script

Analyzes benchmark results and generates:
- Statistical comparisons
- Visualizations (bar charts, box plots, etc.)
- LaTeX tables for academic papers
- Summary reports

Usage:
    python analyze_results.py results/
    python analyze_results.py results/ --output figures/
    python analyze_results.py results/ --latex
"""

import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import datetime


@dataclass
class ModeStats:
    """Statistics for a single mode"""
    mode: str
    runs: int
    
    # E2E Latency
    e2e_mean: float
    e2e_median: float
    e2e_std: float
    e2e_min: float
    e2e_max: float
    e2e_p95: float
    
    # Fan-In
    fanin_wait_mean: float
    fanin_wait_max: float
    poll_count_mean: float
    pre_resolved_mean: float
    
    # Cost
    cost_per_run: float
    dynamo_reads: float
    dynamo_writes: float


def load_summaries(results_dir: Path) -> Dict[str, dict]:
    """Load all summary JSON files from results directory"""
    summaries = {}
    
    for summary_file in results_dir.glob('*_summary.json'):
        with open(summary_file, 'r') as f:
            data = json.load(f)
            mode = data.get('mode', 'UNKNOWN')
            
            # Keep latest if multiple runs for same mode
            if mode not in summaries:
                summaries[mode] = data
            else:
                # Compare timestamps
                existing_ts = summaries[mode].get('timestamp', '')
                new_ts = data.get('timestamp', '')
                if new_ts > existing_ts:
                    summaries[mode] = data
    
    return summaries


def load_all_runs(results_dir: Path) -> Dict[str, List[dict]]:
    """Load all run data from results directory"""
    runs = {}
    
    for runs_file in results_dir.glob('*_runs.json'):
        with open(runs_file, 'r') as f:
            data = json.load(f)
            if data:
                mode = data[0].get('mode', 'UNKNOWN')
                if mode not in runs:
                    runs[mode] = data
                else:
                    runs[mode].extend(data)
    
    return runs


def compute_statistics(summaries: Dict[str, dict]) -> Dict[str, ModeStats]:
    """Compute detailed statistics from summaries"""
    stats = {}
    
    for mode, data in summaries.items():
        stats[mode] = ModeStats(
            mode=mode,
            runs=data.get('successful_runs', 0),
            e2e_mean=data.get('e2e_latency_mean_ms', 0),
            e2e_median=data.get('e2e_latency_median_ms', 0),
            e2e_std=data.get('e2e_latency_std_ms', 0),
            e2e_min=data.get('e2e_latency_min_ms', 0),
            e2e_max=data.get('e2e_latency_max_ms', 0),
            e2e_p95=data.get('e2e_latency_p95_ms', 0),
            fanin_wait_mean=data.get('fanin_wait_mean_ms', 0),
            fanin_wait_max=data.get('fanin_wait_max_ms', 0),
            poll_count_mean=data.get('avg_poll_count', 0),
            pre_resolved_mean=data.get('avg_pre_resolved', 0),
            cost_per_run=data.get('cost_per_run', 0),
            dynamo_reads=data.get('avg_dynamo_reads', 0),
            dynamo_writes=data.get('avg_dynamo_writes', 0),
        )
    
    return stats


def welch_t_test(data1: List[float], data2: List[float]) -> tuple:
    """
    Perform Welch's t-test for unequal variances.
    Returns (t_statistic, p_value_approx)
    """
    n1, n2 = len(data1), len(data2)
    if n1 < 2 or n2 < 2:
        return 0, 1
    
    mean1, mean2 = statistics.mean(data1), statistics.mean(data2)
    var1, var2 = statistics.variance(data1), statistics.variance(data2)
    
    # Welch's t-statistic
    se = ((var1 / n1) + (var2 / n2)) ** 0.5
    if se == 0:
        return 0, 1
    
    t = (mean1 - mean2) / se
    
    # Degrees of freedom (Welch-Satterthwaite)
    num = ((var1 / n1) + (var2 / n2)) ** 2
    denom = ((var1 / n1) ** 2 / (n1 - 1)) + ((var2 / n2) ** 2 / (n2 - 1))
    df = num / denom if denom > 0 else 1
    
    # Approximate p-value using normal distribution for large df
    # For academic rigor, use scipy.stats.t.sf() if available
    import math
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    
    return t, p_value


def cohens_d(data1: List[float], data2: List[float]) -> float:
    """Calculate Cohen's d effect size"""
    n1, n2 = len(data1), len(data2)
    if n1 < 2 or n2 < 2:
        return 0
    
    mean1, mean2 = statistics.mean(data1), statistics.mean(data2)
    var1, var2 = statistics.variance(data1), statistics.variance(data2)
    
    # Pooled standard deviation
    pooled_std = (((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) ** 0.5
    
    if pooled_std == 0:
        return 0
    
    return (mean1 - mean2) / pooled_std


def print_analysis(stats: Dict[str, ModeStats]):
    """Print formatted analysis"""
    print("\n" + "=" * 80)
    print("  BENCHMARK ANALYSIS REPORT")
    print("=" * 80)
    
    # Summary Table
    print("\n  SUMMARY TABLE")
    print("-" * 80)
    print(f"{'Mode':<15} {'Runs':>6} {'E2E Mean':>12} {'E2E Med':>12} {'E2E P95':>12} {'Cost/Run':>12}")
    print("-" * 80)
    
    for mode in ['CLASSIC', 'EAGER', 'FUTURE_BASED']:
        if mode in stats:
            s = stats[mode]
            print(f"{mode:<15} {s.runs:>6} {s.e2e_mean:>10.1f}ms {s.e2e_median:>10.1f}ms "
                  f"{s.e2e_p95:>10.1f}ms ${s.cost_per_run:>10.6f}")
    
    print("-" * 80)
    
    # Fan-In Details
    print("\n  FAN-IN METRICS")
    print("-" * 80)
    print(f"{'Mode':<15} {'Wait Mean':>12} {'Wait Max':>12} {'Poll Count':>12} {'Pre-Resolved':>12}")
    print("-" * 80)
    
    for mode in ['CLASSIC', 'EAGER', 'FUTURE_BASED']:
        if mode in stats:
            s = stats[mode]
            print(f"{mode:<15} {s.fanin_wait_mean:>10.1f}ms {s.fanin_wait_max:>10.1f}ms "
                  f"{s.poll_count_mean:>12.1f} {s.pre_resolved_mean:>10.1f}/5")
    
    print("-" * 80)
    
    # DynamoDB Operations
    print("\n  DYNAMODB OPERATIONS")
    print("-" * 80)
    print(f"{'Mode':<15} {'Reads/Run':>12} {'Writes/Run':>12}")
    print("-" * 80)
    
    for mode in ['CLASSIC', 'EAGER', 'FUTURE_BASED']:
        if mode in stats:
            s = stats[mode]
            print(f"{mode:<15} {s.dynamo_reads:>12.1f} {s.dynamo_writes:>12.1f}")
    
    print("-" * 80)
    
    # Improvement Analysis
    if 'CLASSIC' in stats and 'FUTURE_BASED' in stats:
        classic = stats['CLASSIC']
        future = stats['FUTURE_BASED']
        
        print("\n  IMPROVEMENT ANALYSIS: FUTURE_BASED vs CLASSIC")
        print("-" * 80)
        
        if classic.e2e_mean > 0:
            latency_improvement = (classic.e2e_mean - future.e2e_mean) / classic.e2e_mean * 100
            print(f"  E2E Latency Improvement: {latency_improvement:.1f}%")
            print(f"    Classic: {classic.e2e_mean:.1f}ms → Future: {future.e2e_mean:.1f}ms")
        
        if classic.fanin_wait_mean > 0:
            wait_improvement = (classic.fanin_wait_mean - future.fanin_wait_mean) / classic.fanin_wait_mean * 100
            print(f"  Fan-In Wait Improvement: {wait_improvement:.1f}%")
        
        if classic.cost_per_run > 0:
            cost_increase = (future.cost_per_run - classic.cost_per_run) / classic.cost_per_run * 100
            print(f"  Cost Change: {cost_increase:+.1f}%")
        
        print(f"  Background Polling Effectiveness: {future.pre_resolved_mean:.1f}/5 inputs pre-resolved")
        
    print("\n" + "=" * 80)


def generate_latex_table(stats: Dict[str, ModeStats]) -> str:
    """Generate LaTeX table for academic paper"""
    latex = r"""
\begin{table}[ht]
\centering
\caption{Benchmark Results: Progressive-Aggregator Workflow}
\label{tab:benchmark-results}
\begin{tabular}{lrrrrr}
\toprule
Mode & E2E Mean (ms) & E2E P95 (ms) & Fan-In Wait (ms) & Pre-Resolved & Cost/Run \\
\midrule
"""
    
    for mode in ['CLASSIC', 'EAGER', 'FUTURE\_BASED']:
        mode_key = mode.replace('\\_', '_')
        if mode_key in stats:
            s = stats[mode_key]
            latex += f"{mode} & {s.e2e_mean:.1f} & {s.e2e_p95:.1f} & "
            latex += f"{s.fanin_wait_mean:.1f} & {s.pre_resolved_mean:.1f}/5 & "
            latex += f"\\${s.cost_per_run:.6f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_chart_data(stats: Dict[str, ModeStats]) -> dict:
    """Generate data structure for charting"""
    modes = ['CLASSIC', 'EAGER', 'FUTURE_BASED']
    
    return {
        'modes': modes,
        'e2e_latency': {
            'mean': [stats.get(m, ModeStats(m,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).e2e_mean for m in modes],
            'median': [stats.get(m, ModeStats(m,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).e2e_median for m in modes],
            'p95': [stats.get(m, ModeStats(m,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).e2e_p95 for m in modes],
        },
        'fanin_wait': {
            'mean': [stats.get(m, ModeStats(m,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).fanin_wait_mean for m in modes],
            'max': [stats.get(m, ModeStats(m,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).fanin_wait_max for m in modes],
        },
        'pre_resolved': [stats.get(m, ModeStats(m,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).pre_resolved_mean for m in modes],
        'cost_per_run': [stats.get(m, ModeStats(m,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).cost_per_run for m in modes],
        'dynamo_reads': [stats.get(m, ModeStats(m,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).dynamo_reads for m in modes],
    }


def create_ascii_bar_chart(title: str, labels: List[str], values: List[float], 
                           unit: str = '', max_width: int = 40):
    """Create ASCII bar chart for terminal output"""
    print(f"\n  {title}")
    print("-" * 60)
    
    max_val = max(values) if values else 1
    
    for label, value in zip(labels, values):
        bar_width = int((value / max_val) * max_width) if max_val > 0 else 0
        bar = "█" * bar_width
        print(f"  {label:<15} {bar:<{max_width}} {value:.1f}{unit}")
    
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('results_dir', type=str, help='Directory containing benchmark results')
    parser.add_argument('--output', type=str, help='Output directory for figures')
    parser.add_argument('--latex', action='store_true', help='Generate LaTeX tables')
    parser.add_argument('--json', action='store_true', help='Output chart data as JSON')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Load data
    print(f"\nLoading results from {results_dir}...")
    summaries = load_summaries(results_dir)
    
    if not summaries:
        print("No summary files found!")
        return
    
    print(f"Found results for modes: {list(summaries.keys())}")
    
    # Compute statistics
    stats = compute_statistics(summaries)
    
    # Print analysis
    print_analysis(stats)
    
    # ASCII charts
    modes = ['CLASSIC', 'EAGER', 'FUTURE_BASED']
    available_modes = [m for m in modes if m in stats]
    
    create_ascii_bar_chart(
        "E2E Latency (Mean)",
        available_modes,
        [stats[m].e2e_mean for m in available_modes],
        "ms"
    )
    
    create_ascii_bar_chart(
        "Fan-In Wait Time (Mean)",
        available_modes,
        [stats[m].fanin_wait_mean for m in available_modes],
        "ms"
    )
    
    create_ascii_bar_chart(
        "Pre-Resolved Inputs (Background Polling)",
        available_modes,
        [stats[m].pre_resolved_mean for m in available_modes],
        "/5"
    )
    
    # LaTeX output
    if args.latex:
        print("\n  LATEX TABLE")
        print("-" * 60)
        latex = generate_latex_table(stats)
        print(latex)
        
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            latex_file = output_dir / 'benchmark_table.tex'
            with open(latex_file, 'w') as f:
                f.write(latex)
            print(f"  Saved LaTeX to {latex_file}")
    
    # JSON chart data
    if args.json:
        chart_data = generate_chart_data(stats)
        print("\n  CHART DATA (JSON)")
        print("-" * 60)
        print(json.dumps(chart_data, indent=2))
        
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            json_file = output_dir / 'chart_data.json'
            with open(json_file, 'w') as f:
                json.dump(chart_data, f, indent=2)
            print(f"  Saved JSON to {json_file}")


if __name__ == '__main__':
    main()
