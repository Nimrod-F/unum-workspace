#!/usr/bin/env python3
"""
Generate comparison charts for Graph Analysis benchmark results.
Uses mcp-server-chart compatible output format.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import argparse


def load_latest_results(results_dir: Path) -> Dict[str, Any]:
    """Load the most recent benchmark results"""
    result_files = sorted(results_dir.glob('benchmark_*.json'), reverse=True)
    if not result_files:
        raise FileNotFoundError(f"No benchmark results found in {results_dir}")
    
    with open(result_files[0]) as f:
        return json.load(f)


def print_chart_data_for_mcp(results: Dict[str, Any]):
    """Print data in format suitable for mcp-server-chart tools"""
    
    config = results['config']
    classic = results['classic']
    future = results['future_based']
    
    print("\n" + "="*70)
    print("CHART DATA FOR MCP-SERVER-CHART")
    print("="*70)
    
    # 1. E2E Latency Comparison (Column Chart)
    print("\n## 1. E2E Latency Comparison (Column Chart)")
    latency_data = []
    
    for run in classic['runs']:
        if run.get('error') is None:
            latency_data.append({
                'category': f"Run {run['run_id']}",
                'value': run['e2e_latency_ms'],
                'group': 'CLASSIC'
            })
    
    for run in future['runs']:
        if run.get('error') is None:
            latency_data.append({
                'category': f"Run {run['run_id']}",
                'value': run['e2e_latency_ms'],
                'group': 'FUTURE_BASED'
            })
    
    print(f"Column chart data: {json.dumps(latency_data, indent=2)}")
    
    # 2. Mean Latency Comparison (Bar Chart)
    print("\n## 2. Mean Latency Comparison")
    mean_data = [
        {'category': 'CLASSIC', 'value': classic['stats']['mean']},
        {'category': 'FUTURE_BASED', 'value': future['stats']['mean']}
    ]
    print(f"Bar chart data: {json.dumps(mean_data, indent=2)}")
    
    # 3. Memory Comparison
    print("\n## 3. Memory Comparison")
    memory_data = [
        {'category': 'Max Memory', 'value': classic['stats']['avg_max_memory'], 'group': 'CLASSIC'},
        {'category': 'Max Memory', 'value': future['stats']['avg_max_memory'], 'group': 'FUTURE_BASED'},
        {'category': 'Aggregator Memory', 'value': classic['stats']['avg_aggregator_memory'], 'group': 'CLASSIC'},
        {'category': 'Aggregator Memory', 'value': future['stats']['avg_aggregator_memory'], 'group': 'FUTURE_BASED'},
    ]
    print(f"Grouped column chart data: {json.dumps(memory_data, indent=2)}")
    
    # 4. Cost Comparison
    print("\n## 4. Cost Comparison (Pie Chart)")
    cost_data = [
        {'category': 'CLASSIC', 'value': classic['stats']['total_cost'] * 1000000},  # in micro-dollars
        {'category': 'FUTURE_BASED', 'value': future['stats']['total_cost'] * 1000000}
    ]
    print(f"Pie chart data: {json.dumps(cost_data, indent=2)}")
    
    # 5. Invoker Distribution
    print("\n## 5. Invoker Distribution")
    classic_invokers = {}
    for run in classic['runs']:
        inv = run.get('invoker_branch', 'unknown')
        classic_invokers[inv] = classic_invokers.get(inv, 0) + 1
    
    future_invokers = {}
    for run in future['runs']:
        inv = run.get('invoker_branch', 'unknown')
        future_invokers[inv] = future_invokers.get(inv, 0) + 1
    
    print(f"CLASSIC invokers: {classic_invokers}")
    print(f"FUTURE invokers: {future_invokers}")
    
    # 6. Per-Function Duration
    print("\n## 6. Per-Function Duration (from first run)")
    if classic['runs']:
        classic_funcs = classic['runs'][0].get('per_function_ms', {})
        future_funcs = future['runs'][0].get('per_function_ms', {}) if future['runs'] else {}
        
        func_data = []
        for fname in classic_funcs:
            func_data.append({
                'category': fname.replace('Function', ''),
                'value': classic_funcs.get(fname, 0),
                'group': 'CLASSIC'
            })
            func_data.append({
                'category': fname.replace('Function', ''),
                'value': future_funcs.get(fname, 0),
                'group': 'FUTURE_BASED'
            })
        print(f"Per-function data: {json.dumps(func_data, indent=2)}")
    
    # Summary stats for radar chart
    print("\n## 7. Performance Radar Data")
    # Normalize metrics for radar chart (0-100 scale)
    max_latency = max(classic['stats']['mean'], future['stats']['mean'])
    max_memory = max(classic['stats']['avg_max_memory'], future['stats']['avg_max_memory'])
    max_cost = max(classic['stats']['avg_cost'], future['stats']['avg_cost'])
    
    radar_data = [
        # CLASSIC
        {'name': 'Latency', 'value': 100 - (classic['stats']['mean'] / max_latency * 50), 'group': 'CLASSIC'},
        {'name': 'Memory Eff.', 'value': classic['stats']['avg_memory_efficiency'] * 100, 'group': 'CLASSIC'},
        {'name': 'Cost Eff.', 'value': 100 - (classic['stats']['avg_cost'] / max_cost * 50), 'group': 'CLASSIC'},
        {'name': 'Consistency', 'value': 100 - (classic['stats']['std'] / max_latency * 100), 'group': 'CLASSIC'},
        # FUTURE
        {'name': 'Latency', 'value': 100 - (future['stats']['mean'] / max_latency * 50), 'group': 'FUTURE_BASED'},
        {'name': 'Memory Eff.', 'value': future['stats']['avg_memory_efficiency'] * 100, 'group': 'FUTURE_BASED'},
        {'name': 'Cost Eff.', 'value': 100 - (future['stats']['avg_cost'] / max_cost * 50), 'group': 'FUTURE_BASED'},
        {'name': 'Consistency', 'value': 100 - (future['stats']['std'] / max_latency * 100), 'group': 'FUTURE_BASED'},
    ]
    print(f"Radar chart data: {json.dumps(radar_data, indent=2)}")
    
    return {
        'latency_comparison': latency_data,
        'mean_comparison': mean_data,
        'memory_comparison': memory_data,
        'cost_comparison': cost_data,
        'invoker_classic': classic_invokers,
        'invoker_future': future_invokers,
        'radar_data': radar_data,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='Generate charts from benchmark results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing benchmark results')
    args = parser.parse_args()
    
    results_dir = Path(__file__).parent / args.results_dir
    
    print(f"Loading results from: {results_dir}")
    results = load_latest_results(results_dir)
    
    print(f"\nBenchmark Configuration:")
    print(f"  Nodes: {results['config']['num_nodes']}")
    print(f"  Edge Probability: {results['config']['edge_probability']}")
    print(f"  Iterations: {results['config']['iterations']}")
    
    chart_data = print_chart_data_for_mcp(results)
    
    # Save chart data for easy access
    output_file = results_dir / 'chart_data.json'
    with open(output_file, 'w') as f:
        json.dump(chart_data, f, indent=2)
    print(f"\nChart data saved to: {output_file}")


if __name__ == '__main__':
    main()
