#!/usr/bin/env python3
"""
Quick benchmark runner for Graph Analysis - no redeployment needed.
Tests current deployment configuration.
"""

import boto3
import json
import time
import re
import statistics
import datetime
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

# Configuration
REGION = 'eu-central-1'
PROFILE = 'research-profile'
DYNAMODB_TABLE = 'unum-intermediate-datastore'
LOG_GROUP_PREFIX = '/aws/lambda/'

# Graph size - larger = longer computation = more visible FUTURE benefit
NUM_NODES = 200          # Increased from 50 to 200
EDGE_PROBABILITY = 0.4   # Increased from 0.3 to 0.4 (denser graph)


@dataclass
class FunctionMetrics:
    """Detailed metrics for a single Lambda function"""
    function_name: str
    duration_ms: float = 0.0
    billed_duration_ms: float = 0.0
    memory_size_mb: int = 0
    max_memory_used_mb: int = 0
    init_duration_ms: float = 0.0  # >0 means cold start
    
    @property
    def is_cold_start(self) -> bool:
        return self.init_duration_ms > 0
    
    @property
    def memory_efficiency(self) -> float:
        """Ratio of used memory to allocated memory"""
        if self.memory_size_mb > 0:
            return self.max_memory_used_mb / self.memory_size_mb
        return 0.0


@dataclass
class RunResult:
    run_id: int
    mode: str
    e2e_latency_ms: float
    invoker_branch: str
    cold_starts: int
    total_init_ms: float
    
    # Per-function detailed metrics
    per_function_ms: Dict[str, float] = field(default_factory=dict)
    function_metrics: Dict[str, FunctionMetrics] = field(default_factory=dict)
    
    # Aggregated resource metrics
    total_billed_duration_ms: float = 0.0
    total_memory_used_mb: int = 0
    max_memory_used_mb: int = 0
    aggregator_memory_mb: int = 0
    avg_memory_efficiency: float = 0.0
    
    # Cost estimation (USD)
    estimated_cost_usd: float = 0.0
    
    error: Optional[str] = None


# AWS Lambda Pricing (eu-central-1)
PRICING = {
    'lambda_gb_second': 0.0000166667,  # Per GB-second
    'lambda_request': 0.0000002,        # Per request
}


def load_function_arns():
    yaml_path = Path(__file__).parent.parent / 'function-arn.yaml'
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def force_cold_start(lambda_client, functions, eager_mode: str):
    """Force cold starts by updating function env vars"""
    print("  Forcing cold starts...")
    
    for func_name, func_arn in functions.items():
        try:
            env_vars = {
                'COLD_START_TRIGGER': str(time.time()),
                'CHECKPOINT': 'true',
                'DEBUG': 'true',
                'FAAS_PLATFORM': 'aws',
                'GC': 'false',
                'UNUM_INTERMEDIARY_DATASTORE_NAME': DYNAMODB_TABLE,
                'UNUM_INTERMEDIARY_DATASTORE_TYPE': 'dynamodb',
                'EAGER': eager_mode,
            }
            
            # Add UNUM_FUTURE_BASED for Aggregator
            if func_name == 'Aggregator':
                env_vars['UNUM_FUTURE_BASED'] = 'true' if eager_mode == 'true' else 'false'
            
            lambda_client.update_function_configuration(
                FunctionName=func_arn,
                Environment={'Variables': env_vars}
            )
        except Exception as e:
            print(f"    Warning: {func_name}: {e}")
    
    time.sleep(8)  # Wait for updates


def invoke_workflow(lambda_client, functions, num_nodes=None, edge_prob=None):
    """Invoke the workflow and return timing info"""
    nodes = num_nodes or NUM_NODES
    edges = edge_prob or EDGE_PROBABILITY
    
    payload = {
        "Data": {
            "Source": "http",
            "Value": {
                "num_nodes": nodes,
                "edge_probability": edges
            }
        }
    }
    
    start = time.time()
    response = lambda_client.invoke(
        FunctionName=functions['GraphGenerator'],
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    end = time.time()
    
    if 'FunctionError' in response:
        error = json.loads(response['Payload'].read())
        raise Exception(f"Lambda error: {error}")
    
    return start, end, (end - start) * 1000


def get_lambda_metrics(logs_client, func_arn, start_time, end_time, debug=False) -> FunctionMetrics:
    """Get Lambda REPORT metrics including memory and billing info"""
    func_name = func_arn.split(':')[-1]
    log_group = f"{LOG_GROUP_PREFIX}{func_name}"
    
    metrics = FunctionMetrics(function_name=func_name)
    
    try:
        response = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=int(start_time * 1000),
            endTime=int(end_time * 1000) + 60000,  # Extended window
            filterPattern='REPORT'
        )
        
        if debug:
            print(f"      [DEBUG] {func_name}: {len(response.get('events', []))} REPORT logs")
        
        for event in response.get('events', []):
            msg = event['message']
            if debug:
                print(f"        {msg[:120]}...")
            
            # Parse all fields from REPORT log
            # Format: REPORT RequestId: xxx Duration: 123.45 ms Billed Duration: 124 ms Memory Size: 128 MB Max Memory Used: 64 MB Init Duration: 234.56 ms
            
            duration_match = re.search(r'Duration:\s*([\d.]+)\s*ms', msg)
            billed_match = re.search(r'Billed Duration:\s*([\d.]+)\s*ms', msg)
            memory_size_match = re.search(r'Memory Size:\s*(\d+)\s*MB', msg)
            max_memory_match = re.search(r'Max Memory Used:\s*(\d+)\s*MB', msg)
            init_match = re.search(r'Init Duration:\s*([\d.]+)\s*ms', msg)
            
            if duration_match:
                metrics.duration_ms = float(duration_match.group(1))
                metrics.billed_duration_ms = float(billed_match.group(1)) if billed_match else metrics.duration_ms
                metrics.memory_size_mb = int(memory_size_match.group(1)) if memory_size_match else 128
                metrics.max_memory_used_mb = int(max_memory_match.group(1)) if max_memory_match else 0
                metrics.init_duration_ms = float(init_match.group(1)) if init_match else 0
                break
                
    except Exception as e:
        if debug:
            print(f"      [DEBUG] {func_name}: Error - {e}")
    
    return metrics


def get_invoker_branch(logs_client, functions, start_time, end_time, debug=False):
    """Find which branch invoked the Aggregator"""
    
    # Patterns that indicate this branch triggered the Aggregator
    # Different patterns for CLASSIC vs FUTURE modes
    invoke_patterns = [
        'invoking Aggregator',      # Generic
        'CLASSIC invoking',         # CLASSIC mode
        'Invoking next',            # Unum invoking next function
        'Successfully claimed',     # FUTURE mode - won the race
        'all branches ready',       # CLASSIC - last to checkpoint
    ]
    
    # Patterns that indicate this branch did NOT invoke (skipped)
    skip_patterns = [
        'already claimed',          # FUTURE mode - lost the race
        'waiting for others',       # CLASSIC mode - not last
        'Skipping fan-in',          # Explicitly skipped
    ]
    
    invokers = []
    
    for branch in ['PageRank', 'BFS', 'MST']:
        func_arn = functions.get(branch, '')
        if not func_arn:
            continue
        
        func_name = func_arn.split(':')[-1]
        log_group = f"{LOG_GROUP_PREFIX}{func_name}"
        
        try:
            # Get all logs for this function in the time window
            response = logs_client.filter_log_events(
                logGroupName=log_group,
                startTime=int(start_time * 1000),
                endTime=int(end_time * 1000) + 60000,
            )
            
            logs_text = ' '.join([e['message'] for e in response.get('events', [])])
            
            if debug:
                print(f"      [DEBUG] {branch}: {len(response.get('events', []))} log events")
            
            # Check for invoke patterns
            did_invoke = False
            did_skip = False
            
            for pattern in invoke_patterns:
                if pattern.lower() in logs_text.lower():
                    did_invoke = True
                    if debug:
                        print(f"        Found invoke pattern: '{pattern}'")
                    break
            
            for pattern in skip_patterns:
                if pattern.lower() in logs_text.lower():
                    did_skip = True
                    if debug:
                        print(f"        Found skip pattern: '{pattern}'")
                    break
            
            # If we found an invoke pattern and no skip pattern, this is our invoker
            if did_invoke and not did_skip:
                invokers.append(branch)
                
        except Exception as e:
            if debug:
                print(f"      [DEBUG] {branch}: Error - {e}")
    
    if debug and not invokers:
        print(f"      [DEBUG] No invoker found - checking Aggregator logs...")
        # Check Aggregator logs to see who called it
        agg_arn = functions.get('Aggregator', '')
        if agg_arn:
            func_name = agg_arn.split(':')[-1]
            log_group = f"{LOG_GROUP_PREFIX}{func_name}"
            try:
                response = logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=int(start_time * 1000),
                    endTime=int(end_time * 1000) + 60000,
                )
                for event in response.get('events', [])[:5]:  # First 5 events
                    print(f"        Aggregator: {event['message'][:80]}...")
            except:
                pass
    
    # Return the invoker(s)
    if len(invokers) == 1:
        return invokers[0]
    elif len(invokers) > 1:
        return '+'.join(invokers)  # Multiple invokers (shouldn't happen)
    else:
        return 'unknown'


def run_benchmark(mode: str, iterations: int = 3, debug: bool = False, num_nodes: int = None, edge_prob: float = None):
    """Run benchmark for a mode"""
    nodes = num_nodes or NUM_NODES
    edges = edge_prob or EDGE_PROBABILITY
    
    print(f"\n{'='*60}")
    print(f"  {mode} MODE - {iterations} iterations with cold starts")
    print(f"  Graph: {nodes} nodes, {edges} edge probability")
    print(f"{'='*60}")
    
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    lambda_client = session.client('lambda')
    logs_client = session.client('logs')
    
    functions = load_function_arns()
    eager_mode = 'true' if mode == 'FUTURE_BASED' else 'false'
    
    results = []
    
    for i in range(iterations):
        print(f"\n  Run {i+1}/{iterations}")
        
        # Force cold start
        force_cold_start(lambda_client, functions, eager_mode)
        
        try:
            # Invoke
            start, end, latency = invoke_workflow(lambda_client, functions, num_nodes=nodes, edge_prob=edges)
            print(f"    E2E Latency: {latency:.1f}ms")
            
            # Wait longer for CloudWatch logs to be available
            print(f"    Waiting for CloudWatch logs...")
            time.sleep(15)  # Increased from 5s to 15s
            
            # Get detailed metrics for each function
            per_func = {}
            function_metrics = {}
            total_init = 0
            cold_count = 0
            total_billed = 0
            total_memory = 0
            max_memory = 0
            aggregator_memory = 0
            memory_efficiencies = []
            
            for fname, farn in functions.items():
                metrics = get_lambda_metrics(logs_client, farn, start, end, debug=debug)
                per_func[fname] = metrics.duration_ms
                function_metrics[fname] = metrics
                
                if metrics.is_cold_start:
                    cold_count += 1
                    total_init += metrics.init_duration_ms
                
                total_billed += metrics.billed_duration_ms
                total_memory += metrics.max_memory_used_mb
                max_memory = max(max_memory, metrics.max_memory_used_mb)
                
                if 'Aggregator' in fname:
                    aggregator_memory = metrics.max_memory_used_mb
                
                if metrics.memory_efficiency > 0:
                    memory_efficiencies.append(metrics.memory_efficiency)
            
            # Calculate cost
            # Cost = (billed_duration_ms / 1000) * (memory_size_mb / 1024) * price_per_gb_second + request_cost
            total_cost = 0.0
            for fname, m in function_metrics.items():
                gb_seconds = (m.billed_duration_ms / 1000) * (m.memory_size_mb / 1024)
                total_cost += gb_seconds * PRICING['lambda_gb_second']
                total_cost += PRICING['lambda_request']
            
            # Get invoker
            invoker = get_invoker_branch(logs_client, functions, start, end, debug=debug)
            
            avg_efficiency = statistics.mean(memory_efficiencies) if memory_efficiencies else 0
            
            print(f"    Invoker: {invoker}, Cold starts: {cold_count}")
            print(f"    Memory: max={max_memory}MB, aggregator={aggregator_memory}MB, efficiency={avg_efficiency:.1%}")
            print(f"    Cost: ${total_cost:.6f}")
            
            results.append(RunResult(
                run_id=i+1,
                mode=mode,
                e2e_latency_ms=latency,
                invoker_branch=invoker,
                cold_starts=cold_count,
                total_init_ms=total_init,
                per_function_ms=per_func,
                function_metrics=function_metrics,
                total_billed_duration_ms=total_billed,
                total_memory_used_mb=total_memory,
                max_memory_used_mb=max_memory,
                aggregator_memory_mb=aggregator_memory,
                avg_memory_efficiency=avg_efficiency,
                estimated_cost_usd=total_cost
            ))
            
        except Exception as e:
            print(f"    Error: {e}")
            results.append(RunResult(
                run_id=i+1, mode=mode, e2e_latency_ms=0,
                invoker_branch='', cold_starts=0,
                total_init_ms=0, error=str(e)
            ))
        
        # Brief pause
        time.sleep(2)
    
    return results


def print_summary(classic_results, future_results):
    """Print comparison summary"""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    
    def stats(results):
        valid = [r for r in results if r.error is None]
        if not valid:
            return {}
        
        latencies = [r.e2e_latency_ms for r in valid]
        return {
            'mean': statistics.mean(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'cold_starts': sum(r.cold_starts for r in valid),
            'init_time': sum(r.total_init_ms for r in valid),
            'invokers': [r.invoker_branch for r in valid],
            # Resource metrics
            'avg_billed_duration': statistics.mean([r.total_billed_duration_ms for r in valid]),
            'avg_max_memory': statistics.mean([r.max_memory_used_mb for r in valid]) if valid else 0,
            'avg_total_memory': statistics.mean([r.total_memory_used_mb for r in valid]) if valid else 0,
            'avg_aggregator_memory': statistics.mean([r.aggregator_memory_mb for r in valid]) if valid else 0,
            'avg_memory_efficiency': statistics.mean([r.avg_memory_efficiency for r in valid]) if valid else 0,
            'avg_cost': statistics.mean([r.estimated_cost_usd for r in valid]) if valid else 0,
            'total_cost': sum(r.estimated_cost_usd for r in valid),
        }
    
    classic = stats(classic_results)
    future = stats(future_results)
    
    print(f"\n{'Metric':<35} {'CLASSIC':<20} {'FUTURE_BASED':<20}")
    print("-"*75)
    
    if classic and future:
        # Latency metrics
        print(f"{'E2E Latency (mean)':<35} {classic['mean']:.1f}ms{'':<12} {future['mean']:.1f}ms")
        print(f"{'E2E Latency (min)':<35} {classic['min']:.1f}ms{'':<12} {future['min']:.1f}ms")
        print(f"{'E2E Latency (max)':<35} {classic['max']:.1f}ms{'':<12} {future['max']:.1f}ms")
        print(f"{'E2E Latency (std)':<35} {classic['std']:.1f}ms{'':<12} {future['std']:.1f}ms")
        print(f"{'Total Cold Starts':<35} {classic['cold_starts']:<20} {future['cold_starts']}")
        print(f"{'Total Init Time':<35} {classic['init_time']:.1f}ms{'':<12} {future['init_time']:.1f}ms")
        
        # Resource metrics
        print("\n" + "-"*75)
        print("RESOURCE METRICS")
        print("-"*75)
        print(f"{'Avg Billed Duration (total)':<35} {classic['avg_billed_duration']:.1f}ms{'':<8} {future['avg_billed_duration']:.1f}ms")
        print(f"{'Avg Max Memory Used':<35} {classic['avg_max_memory']:.0f}MB{'':<12} {future['avg_max_memory']:.0f}MB")
        print(f"{'Avg Total Memory (all funcs)':<35} {classic['avg_total_memory']:.0f}MB{'':<12} {future['avg_total_memory']:.0f}MB")
        print(f"{'Avg Aggregator Memory':<35} {classic['avg_aggregator_memory']:.0f}MB{'':<12} {future['avg_aggregator_memory']:.0f}MB")
        print(f"{'Avg Memory Efficiency':<35} {classic['avg_memory_efficiency']:.1%}{'':<12} {future['avg_memory_efficiency']:.1%}")
        print(f"{'Avg Cost per Run':<35} ${classic['avg_cost']:.6f}{'':<8} ${future['avg_cost']:.6f}")
        print(f"{'Total Cost':<35} ${classic['total_cost']:.6f}{'':<8} ${future['total_cost']:.6f}")
        
        # Improvements
        print("\n" + "-"*75)
        print("IMPROVEMENTS (FUTURE vs CLASSIC)")
        print("-"*75)
        if classic['mean'] > 0:
            latency_improvement = ((classic['mean'] - future['mean']) / classic['mean']) * 100
            print(f"{'Latency Improvement:':<35} {latency_improvement:+.1f}%")
        
        if classic['avg_aggregator_memory'] > 0:
            memory_overhead = ((future['avg_aggregator_memory'] - classic['avg_aggregator_memory']) / classic['avg_aggregator_memory']) * 100
            print(f"{'Aggregator Memory Overhead:':<35} {memory_overhead:+.1f}%")
        
        if classic['avg_cost'] > 0:
            cost_diff = ((future['avg_cost'] - classic['avg_cost']) / classic['avg_cost']) * 100
            print(f"{'Cost Difference:':<35} {cost_diff:+.1f}%")
        
        # Invoker distribution
        print("\nInvoker Distribution (which branch triggered Aggregator):")
        print(f"  CLASSIC:      {classic['invokers']}")
        print(f"  FUTURE_BASED: {future['invokers']}")
    
    return classic, future


def save_results(classic_results, future_results, classic_stats, future_stats, num_nodes=None, edge_prob=None):
    """Save results to JSON"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'num_nodes': num_nodes or NUM_NODES,
            'edge_probability': edge_prob or EDGE_PROBABILITY,
            'iterations': len(classic_results)
        },
        'classic': {
            'runs': [asdict(r) for r in classic_results],
            'stats': classic_stats
        },
        'future_based': {
            'runs': [asdict(r) for r in future_results],
            'stats': future_stats
        }
    }
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / f'benchmark_{timestamp}.json'
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Graph Analysis Benchmark')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--iterations', '-n', type=int, default=3, help='Number of iterations per mode')
    parser.add_argument('--nodes', type=int, default=NUM_NODES, help=f'Number of graph nodes (default: {NUM_NODES})')
    parser.add_argument('--edge-prob', type=float, default=EDGE_PROBABILITY, help=f'Edge probability (default: {EDGE_PROBABILITY})')
    args = parser.parse_args()
    
    print("\n" + "#"*70)
    print("#  GRAPH ANALYSIS BENCHMARK: CLASSIC vs FUTURE-BASED")
    print(f"#  Graph: {args.nodes} nodes, {args.edge_prob} edge probability")
    print(f"#  {args.iterations} iterations per mode, forced cold starts")
    if args.debug:
        print("#  DEBUG MODE ENABLED")
    print("#"*70)
    
    # Run CLASSIC first
    classic_results = run_benchmark('CLASSIC', iterations=args.iterations, debug=args.debug, 
                                    num_nodes=args.nodes, edge_prob=args.edge_prob)
    
    # Run FUTURE_BASED
    future_results = run_benchmark('FUTURE_BASED', iterations=args.iterations, debug=args.debug,
                                   num_nodes=args.nodes, edge_prob=args.edge_prob)
    
    # Print summary
    classic_stats, future_stats = print_summary(classic_results, future_results)
    
    # Save results
    filepath = save_results(classic_results, future_results, classic_stats, future_stats,
                           num_nodes=args.nodes, edge_prob=args.edge_prob)
    
    return classic_stats, future_stats, filepath


if __name__ == '__main__':
    main()
