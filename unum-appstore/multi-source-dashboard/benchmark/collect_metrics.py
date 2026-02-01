#!/usr/bin/env python3
"""
Metrics Collection and Analysis for Multi-Source Dashboard

This script analyzes benchmark results and extracts detailed metrics
including per-function latencies, cold start indicators, and fan-in efficiency.

Usage:
    python collect_metrics.py --results-file results/benchmark_20240115.json
    python collect_metrics.py --analyze-all results/
"""

import boto3
import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, List
import yaml


def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration."""
    config_path = Path(__file__).parent / config_file
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def analyze_benchmark_results(results_file: Path) -> Dict:
    """Analyze benchmark results and compute detailed metrics."""
    with open(results_file, 'r') as f:
        data = json.load(f)

    analysis = {
        'file': str(results_file),
        'modes': {}
    }

    for mode, mode_data in data.items():
        summary = mode_data.get('summary', {})
        runs = mode_data.get('runs', [])

        mode_analysis = {
            'summary': summary,
            'detailed_metrics': analyze_runs(runs),
            'comparison': {}
        }

        analysis['modes'][mode] = mode_analysis

    # Add cross-mode comparisons
    if 'CLASSIC' in analysis['modes'] and 'FUTURE_BASED' in analysis['modes']:
        classic = analysis['modes']['CLASSIC']['summary']
        future = analysis['modes']['FUTURE_BASED']['summary']

        improvement_pct = 0
        if classic['e2e_mean_ms'] > 0:
            improvement_pct = (
                (classic['e2e_mean_ms'] - future['e2e_mean_ms']) /
                classic['e2e_mean_ms'] * 100
            )

        analysis['comparison'] = {
            'latency_improvement_pct': improvement_pct,
            'classic_mean_ms': classic['e2e_mean_ms'],
            'future_mean_ms': future['e2e_mean_ms'],
            'time_saved_ms': classic['e2e_mean_ms'] - future['e2e_mean_ms'],
            'avg_pre_resolved_future': future.get('avg_pre_resolved', 0)
        }

    return analysis


def analyze_runs(runs: List[Dict]) -> Dict:
    """Analyze individual runs for detailed metrics."""
    successful_runs = [r for r in runs if r.get('success', False)]

    if not successful_runs:
        return {}

    # Extract latencies per function
    function_latencies = {}
    for run in successful_runs:
        for func, latency in run.get('function_latencies', {}).items():
            if func not in function_latencies:
                function_latencies[func] = []
            function_latencies[func].append(latency)

    # Compute per-function statistics
    function_stats = {}
    for func, latencies in function_latencies.items():
        function_stats[func] = {
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'min_ms': min(latencies),
            'max_ms': max(latencies)
        }

    # Cold start analysis
    total_runs = len(successful_runs)
    cold_start_counts = {}
    for run in successful_runs:
        for func in run.get('cold_starts', []):
            cold_start_counts[func] = cold_start_counts.get(func, 0) + 1

    cold_start_rates = {
        func: (count / total_runs * 100)
        for func, count in cold_start_counts.items()
    }

    # Pre-resolved analysis
    pre_resolved_values = [r.get('pre_resolved_count', 0) for r in successful_runs]
    pre_resolved_stats = {
        'mean': statistics.mean(pre_resolved_values),
        'median': statistics.median(pre_resolved_values),
        'max': max(pre_resolved_values),
        'pct_with_pre_resolved': sum(1 for v in pre_resolved_values if v > 0) / len(pre_resolved_values) * 100
    }

    return {
        'function_stats': function_stats,
        'cold_start_rates': cold_start_rates,
        'pre_resolved_stats': pre_resolved_stats
    }


def export_csv(analysis: Dict, output_file: Path):
    """Export analysis to CSV format."""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['Mode', 'Metric', 'Value'])

        # Write mode summaries
        for mode, mode_data in analysis['modes'].items():
            summary = mode_data['summary']
            writer.writerow([mode, 'E2E Mean (ms)', f"{summary['e2e_mean_ms']:.2f}"])
            writer.writerow([mode, 'E2E Std (ms)', f"{summary['e2e_std_ms']:.2f}"])
            writer.writerow([mode, 'E2E Min (ms)', f"{summary['e2e_min_ms']:.2f}"])
            writer.writerow([mode, 'E2E Max (ms)', f"{summary['e2e_max_ms']:.2f}"])
            writer.writerow([mode, 'Cold Mean (ms)', f"{summary['cold_mean_ms']:.2f}"])
            writer.writerow([mode, 'Warm Mean (ms)', f"{summary['warm_mean_ms']:.2f}"])
            writer.writerow([mode, 'Avg Pre-Resolved', f"{summary.get('avg_pre_resolved', 0):.2f}"])
            writer.writerow(['', '', ''])  # Blank row

        # Write comparison
        if 'comparison' in analysis:
            comp = analysis['comparison']
            writer.writerow(['Comparison', 'Improvement (%)', f"{comp['latency_improvement_pct']:.2f}"])
            writer.writerow(['Comparison', 'Time Saved (ms)', f"{comp['time_saved_ms']:.2f}"])


def export_chart_data(analysis: Dict, output_file: Path):
    """Export data in format suitable for charting."""
    chart_data = {
        'labels': [],
        'e2e_latency': [],
        'cold_latency': [],
        'warm_latency': [],
        'error_bars': []
    }

    for mode in ['CLASSIC', 'EAGER', 'FUTURE_BASED']:
        if mode in analysis['modes']:
            summary = analysis['modes'][mode]['summary']
            chart_data['labels'].append(mode)
            chart_data['e2e_latency'].append(summary['e2e_mean_ms'])
            chart_data['cold_latency'].append(summary['cold_mean_ms'])
            chart_data['warm_latency'].append(summary['warm_mean_ms'])
            chart_data['error_bars'].append(summary['e2e_std_ms'])

    with open(output_file, 'w') as f:
        json.dump(chart_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Collect and analyze benchmark metrics'
    )
    parser.add_argument('--results-file', type=str,
                       help='Benchmark results JSON file')
    parser.add_argument('--analyze-all', type=str,
                       help='Directory containing multiple result files')
    parser.add_argument('--output-csv', type=str,
                       help='Output CSV file')
    parser.add_argument('--output-chart', type=str,
                       help='Output chart data JSON file')

    args = parser.parse_args()

    if args.results_file:
        results_files = [Path(args.results_file)]
    elif args.analyze_all:
        results_dir = Path(args.analyze_all)
        results_files = list(results_dir.glob('benchmark_*.json'))
    else:
        print("Error: Must specify --results-file or --analyze-all")
        return

    for results_file in results_files:
        print(f"\nAnalyzing: {results_file}")
        analysis = analyze_benchmark_results(results_file)

        # Print summary
        print("\nSummary:")
        for mode, mode_data in analysis['modes'].items():
            summary = mode_data['summary']
            print(f"\n{mode}:")
            print(f"  E2E: {summary['e2e_mean_ms']:.0f}ms ± {summary['e2e_std_ms']:.0f}ms")
            print(f"  Cold: {summary['cold_mean_ms']:.0f}ms")
            print(f"  Warm: {summary['warm_mean_ms']:.0f}ms")
            print(f"  Pre-resolved: {summary.get('avg_pre_resolved', 0):.1f}")

        if 'comparison' in analysis:
            comp = analysis['comparison']
            print(f"\nImprovement: {comp['latency_improvement_pct']:.1f}%")
            print(f"Time saved: {comp['time_saved_ms']:.0f}ms")

        # Export if requested
        if args.output_csv:
            csv_path = Path(args.output_csv)
            export_csv(analysis, csv_path)
            print(f"\nCSV exported to: {csv_path}")

        if args.output_chart:
            chart_path = Path(args.output_chart)
            export_chart_data(analysis, chart_path)
            print(f"Chart data exported to: {chart_path}")


if __name__ == '__main__':
    main()
