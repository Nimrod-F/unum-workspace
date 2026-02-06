"""
Streaming Benchmark Script

Compares Partial Parameter Streaming vs Normal execution.

Usage:
    python benchmark.py --mode streaming --runs 5
    python benchmark.py --mode normal --runs 5
    python benchmark.py --compare
"""

import argparse
import json
import time
import boto3
import os
import statistics
from datetime import datetime


# AWS Configuration
REGION = os.environ.get('AWS_REGION', 'eu-central-1')
PROFILE = os.environ.get('AWS_PROFILE', 'research-profile')
STACK_NAME = 'unum-streaming-benchmark'


def get_lambda_client():
    """Get Lambda client with the correct profile"""
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    return session.client('lambda')


def get_function_arn(function_name):
    """Get function ARN from stack outputs or construct it"""
    return f"arn:aws:lambda:{REGION}:{get_account_id()}:function:{STACK_NAME}-{function_name}"


def get_account_id():
    """Get AWS account ID"""
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    sts = session.client('sts')
    return sts.get_caller_identity()['Account']


def invoke_workflow(lambda_client, payload=None):
    """
    Invoke the workflow and measure end-to-end latency.
    
    Returns timing information about the execution.
    """
    if payload is None:
        payload = {
            "data": [100 + i * 0.5 for i in range(100)]  # Sample data
        }
    
    # Wrap in unum format
    unum_payload = {
        "Data": {
            "Source": "http",
            "Value": payload
        },
        "Session": f"benchmark-{datetime.now().isoformat()}"
    }
    
    producer_arn = f"{STACK_NAME}-Producer"
    
    start_time = time.time()
    
    # Invoke Producer (which triggers the rest of the pipeline)
    response = lambda_client.invoke(
        FunctionName=producer_arn,
        InvocationType='RequestResponse',
        Payload=json.dumps(unum_payload)
    )
    
    # Parse response
    response_payload = json.loads(response['Payload'].read())
    
    end_time = time.time()
    e2e_latency = (end_time - start_time) * 1000  # Convert to ms
    
    return {
        "e2e_latency_ms": round(e2e_latency, 2),
        "start_time": start_time,
        "end_time": end_time,
        "response": response_payload
    }


def run_benchmark(mode, runs=5, warmup=1):
    """
    Run benchmark for the specified mode.
    
    Args:
        mode: 'streaming' or 'normal'
        runs: Number of benchmark runs
        warmup: Number of warmup runs (not counted)
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {mode.upper()} MODE")
    print(f"{'='*60}")
    print(f"Runs: {runs}, Warmup: {warmup}")
    
    lambda_client = get_lambda_client()
    
    # Warmup runs
    print(f"\nPerforming {warmup} warmup run(s)...")
    for i in range(warmup):
        try:
            result = invoke_workflow(lambda_client)
            print(f"  Warmup {i+1}: {result['e2e_latency_ms']:.2f}ms")
        except Exception as e:
            print(f"  Warmup {i+1} failed: {e}")
    
    # Wait for any background processing
    time.sleep(2)
    
    # Benchmark runs
    print(f"\nPerforming {runs} benchmark run(s)...")
    results = []
    
    for i in range(runs):
        try:
            result = invoke_workflow(lambda_client)
            results.append(result)
            print(f"  Run {i+1}: {result['e2e_latency_ms']:.2f}ms")
        except Exception as e:
            print(f"  Run {i+1} failed: {e}")
        
        # Small delay between runs
        time.sleep(1)
    
    # Calculate statistics
    if results:
        latencies = [r['e2e_latency_ms'] for r in results]
        
        stats = {
            "mode": mode,
            "runs": len(results),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "mean_ms": round(statistics.mean(latencies), 2),
            "median_ms": round(statistics.median(latencies), 2),
            "stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
            "raw_latencies": latencies
        }
        
        print(f"\n--- Results ({mode}) ---")
        print(f"  Min:    {stats['min_ms']:.2f}ms")
        print(f"  Max:    {stats['max_ms']:.2f}ms")
        print(f"  Mean:   {stats['mean_ms']:.2f}ms")
        print(f"  Median: {stats['median_ms']:.2f}ms")
        print(f"  StdDev: {stats['stdev_ms']:.2f}ms")
        
        # Save results
        filename = f"benchmark_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to: {filename}")
        
        return stats
    
    return None


def compare_results():
    """
    Compare streaming vs normal results from saved files.
    """
    import glob
    
    print(f"\n{'='*60}")
    print("COMPARISON: STREAMING vs NORMAL")
    print(f"{'='*60}")
    
    # Find most recent result files
    streaming_files = sorted(glob.glob("benchmark_streaming_*.json"), reverse=True)
    normal_files = sorted(glob.glob("benchmark_normal_*.json"), reverse=True)
    
    if not streaming_files:
        print("No streaming benchmark results found. Run with --mode streaming first.")
        return
    
    if not normal_files:
        print("No normal benchmark results found. Run with --mode normal first.")
        return
    
    # Load most recent results
    with open(streaming_files[0], 'r') as f:
        streaming = json.load(f)
    
    with open(normal_files[0], 'r') as f:
        normal = json.load(f)
    
    print(f"\nStreaming results: {streaming_files[0]}")
    print(f"Normal results: {normal_files[0]}")
    
    print(f"\n{'Metric':<20} {'Streaming':>12} {'Normal':>12} {'Improvement':>12}")
    print("-" * 58)
    
    for metric in ['min_ms', 'max_ms', 'mean_ms', 'median_ms']:
        s_val = streaming[metric]
        n_val = normal[metric]
        improvement = ((n_val - s_val) / n_val) * 100
        
        metric_name = metric.replace('_ms', '').capitalize()
        print(f"{metric_name:<20} {s_val:>10.2f}ms {n_val:>10.2f}ms {improvement:>10.1f}%")
    
    # Overall verdict
    mean_improvement = ((normal['mean_ms'] - streaming['mean_ms']) / normal['mean_ms']) * 100
    
    print(f"\n{'='*58}")
    if mean_improvement > 0:
        print(f"STREAMING is {mean_improvement:.1f}% FASTER than NORMAL execution")
    else:
        print(f"NORMAL is {-mean_improvement:.1f}% FASTER than STREAMING execution")
    print(f"{'='*58}")


def main():
    parser = argparse.ArgumentParser(description='Streaming Benchmark')
    parser.add_argument('--mode', choices=['streaming', 'normal'], 
                        help='Benchmark mode')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=1,
                        help='Number of warmup runs')
    parser.add_argument('--compare', action='store_true',
                        help='Compare saved results')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results()
    elif args.mode:
        run_benchmark(args.mode, args.runs, args.warmup)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
