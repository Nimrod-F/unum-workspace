#!/usr/bin/env python3
"""
Quick benchmark runner for Image Pipeline - no redeployment needed.
Tests current deployment configuration with minimal setup.

Usage:
    python quick_benchmark.py
    python quick_benchmark.py --iterations 5
    python quick_benchmark.py --mode FUTURE_BASED
"""

import boto3
import json
import time
import re
import statistics
import datetime
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Configuration
REGION = 'eu-central-1'
PROFILE = 'research-profile'
DYNAMODB_TABLE = 'unum-intermediate-datastore'
LOG_GROUP_PREFIX = '/aws/lambda/'

# Test image
TEST_BUCKET = 'unum-benchmark-images'
TEST_KEY = 'test-images/sample-1920x1080.jpg'

# Branches
FAN_IN_BRANCHES = ['Thumbnail', 'Transform', 'Filters', 'Contour']


@dataclass
class FunctionMetrics:
    """Metrics for a single Lambda function"""
    function_name: str
    duration_ms: float = 0.0
    billed_duration_ms: float = 0.0
    memory_size_mb: int = 0
    max_memory_used_mb: int = 0
    init_duration_ms: float = 0.0
    
    @property
    def is_cold_start(self) -> bool:
        return self.init_duration_ms > 0


@dataclass
class RunResult:
    run_id: int
    mode: str
    e2e_latency_ms: float
    invoker_branch: str
    cold_starts: int
    per_function_ms: Dict[str, float] = field(default_factory=dict)
    branch_variance_ms: float = 0.0
    total_billed_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    error: Optional[str] = None


# Pricing
PRICING = {
    'lambda_gb_second': 0.0000166667,
    'lambda_request': 0.0000002,
}


def load_function_arns():
    yaml_path = Path(__file__).parent.parent / 'function-arn.yaml'
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def configure_mode(lambda_client, functions, mode: str):
    """Configure Publisher for the specified mode"""
    print(f"  Configuring {mode} mode...")
    
    future_value = 'true' if mode == 'FUTURE_BASED' else 'false'
    
    try:
        lambda_client.update_function_configuration(
            FunctionName=functions['Publisher'],
            Environment={
                'Variables': {
                    'CHECKPOINT': 'true',
                    'DEBUG': 'true',
                    'FAAS_PLATFORM': 'aws',
                    'GC': 'false',
                    'UNUM_INTERMEDIARY_DATASTORE_NAME': DYNAMODB_TABLE,
                    'UNUM_INTERMEDIARY_DATASTORE_TYPE': 'dynamodb',
                    'EAGER': 'true',
                    'UNUM_FUTURE_BASED': future_value,
                }
            }
        )
        print(f"  ✓ Publisher configured for {mode}")
    except Exception as e:
        print(f"  ⚠ Warning: {e}")
    
    time.sleep(5)


def force_cold_start(lambda_client, functions, mode: str):
    """Force cold starts by updating function env vars"""
    print("  Forcing cold starts...")
    
    future_value = 'true' if mode == 'FUTURE_BASED' else 'false'
    
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
                'EAGER': 'true',
            }
            
            if func_name == 'Publisher':
                env_vars['UNUM_FUTURE_BASED'] = future_value
            
            lambda_client.update_function_configuration(
                FunctionName=func_arn,
                Environment={'Variables': env_vars}
            )
        except Exception as e:
            pass
    
    time.sleep(8)


def invoke_workflow(lambda_client, functions):
    """Invoke the workflow and return timing info"""
    payload = {
        "Data": {
            "Source": "http",
            "Value": {
                "bucket": TEST_BUCKET,
                "key": TEST_KEY
            }
        }
    }
    
    start = time.time()
    response = lambda_client.invoke(
        FunctionName=functions['ImageLoader'],
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    end = time.time()
    
    if 'FunctionError' in response:
        error = json.loads(response['Payload'].read())
        raise Exception(f"Lambda error: {error}")
    
    return start, end, (end - start) * 1000


def get_lambda_metrics(logs_client, func_arn, start_time, end_time) -> FunctionMetrics:
    """Get Lambda REPORT metrics"""
    func_name = func_arn.split(':')[-1]
    log_group = f"{LOG_GROUP_PREFIX}{func_name}"
    
    metrics = FunctionMetrics(function_name=func_name)
    
    try:
        response = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=int(start_time * 1000),
            endTime=int(end_time * 1000) + 60000,
            filterPattern='REPORT'
        )
        
        for event in response.get('events', []):
            msg = event['message']
            
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
                
    except Exception:
        pass
    
    return metrics


def get_invoker_branch(logs_client, functions, start_time, end_time) -> str:
    """Find which branch invoked the Publisher"""
    invoke_patterns = ['invoking Publisher', 'invoking next', 'Successfully claimed', 'all branches ready']
    skip_patterns = ['already claimed', 'waiting for others', 'Skipping']
    
    invokers = []
    
    for branch in FAN_IN_BRANCHES:
        func_arn = functions.get(branch, '')
        if not func_arn:
            continue
        
        func_name = func_arn.split(':')[-1]
        log_group = f"{LOG_GROUP_PREFIX}{func_name}"
        
        try:
            response = logs_client.filter_log_events(
                logGroupName=log_group,
                startTime=int(start_time * 1000),
                endTime=int(end_time * 1000) + 60000,
            )
            
            logs_text = ' '.join([e['message'] for e in response.get('events', [])])
            
            did_invoke = any(p.lower() in logs_text.lower() for p in invoke_patterns)
            did_skip = any(p.lower() in logs_text.lower() for p in skip_patterns)
            
            if did_invoke and not did_skip:
                invokers.append(branch)
                
        except Exception:
            pass
    
    if len(invokers) == 1:
        return invokers[0]
    elif len(invokers) > 1:
        return '+'.join(invokers)
    return 'unknown'


def run_benchmark(mode: str, iterations: int = 3, force_cold: bool = True):
    """Run benchmark for a mode"""
    print(f"\n{'='*60}")
    print(f"  {mode} MODE - {iterations} iterations")
    print(f"{'='*60}")
    
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    lambda_client = session.client('lambda')
    logs_client = session.client('logs')
    
    functions = load_function_arns()
    
    # Configure mode
    configure_mode(lambda_client, functions, mode)
    
    results = []
    
    for i in range(iterations):
        print(f"\n  Run {i+1}/{iterations}")
        
        if force_cold:
            force_cold_start(lambda_client, functions, mode)
        
        try:
            start, end, latency = invoke_workflow(lambda_client, functions)
            print(f"    E2E Latency: {latency:.1f}ms")
            
            # Wait for logs
            print(f"    Waiting for CloudWatch logs...")
            time.sleep(20)  # Increased wait time for CloudWatch propagation
            
            # Get metrics
            per_func = {}
            cold_count = 0
            total_billed = 0
            branch_durations = []
            
            all_functions = ['ImageLoader', 'Thumbnail', 'Transform', 'Filters', 'Contour', 'Publisher']
            
            for fname in all_functions:
                farn = functions.get(fname, '')
                if farn:
                    metrics = get_lambda_metrics(logs_client, farn, start, end)
                    per_func[fname] = metrics.duration_ms
                    print(f"      {fname}: {metrics.duration_ms:.1f}ms (cold: {metrics.is_cold_start})")
                    if metrics.is_cold_start:
                        cold_count += 1
                    total_billed += metrics.billed_duration_ms
                    
                    if fname in FAN_IN_BRANCHES:
                        if metrics.duration_ms > 0:
                            branch_durations.append(metrics.duration_ms)
            
            # Branch variance
            variance = max(branch_durations) - min(branch_durations) if branch_durations else 0
            
            # Invoker
            invoker = get_invoker_branch(logs_client, functions, start, end)
            
            # Cost
            cost = (total_billed / 1000) * (512 / 1024) * PRICING['lambda_gb_second']
            cost += len(all_functions) * PRICING['lambda_request']
            
            print(f"    Invoker: {invoker}, Cold starts: {cold_count}")
            print(f"    Branch variance: {variance:.1f}ms")
            
            results.append(RunResult(
                run_id=i+1,
                mode=mode,
                e2e_latency_ms=latency,
                invoker_branch=invoker,
                cold_starts=cold_count,
                per_function_ms=per_func,
                branch_variance_ms=variance,
                total_billed_ms=total_billed,
                estimated_cost_usd=cost
            ))
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results.append(RunResult(
                run_id=i+1, mode=mode, e2e_latency_ms=0,
                invoker_branch='', cold_starts=0, error=str(e)
            ))
        
        # Brief pause
        if i < iterations - 1:
            time.sleep(3)
    
    return results


def print_summary(classic_results: List[RunResult], future_results: List[RunResult]):
    """Print benchmark summary"""
    print("\n" + "=" * 70)
    print("  IMAGE PIPELINE BENCHMARK SUMMARY")
    print("  CLASSIC vs FUTURE_BASED")
    print("=" * 70)
    
    # Calculate averages
    classic_ok = [r for r in classic_results if not r.error]
    future_ok = [r for r in future_results if not r.error]
    
    classic_avg = statistics.mean([r.e2e_latency_ms for r in classic_ok]) if classic_ok else 0
    future_avg = statistics.mean([r.e2e_latency_ms for r in future_ok]) if future_ok else 0
    
    print(f"\n  {'Metric':<30} {'CLASSIC':>15} {'FUTURE_BASED':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'E2E Latency (avg ms)':<30} {classic_avg:>15.1f} {future_avg:>15.1f}")
    
    if classic_avg > 0 and future_avg > 0:
        improvement = classic_avg - future_avg
        improvement_pct = (improvement / classic_avg) * 100
        print(f"\n  Improvement: {improvement:.1f}ms ({improvement_pct:.1f}%)")
    
    # Invoker distribution
    print(f"\n  Invoker Distribution:")
    for mode, results in [('CLASSIC', classic_ok), ('FUTURE_BASED', future_ok)]:
        invokers = {}
        for r in results:
            invokers[r.invoker_branch] = invokers.get(r.invoker_branch, 0) + 1
        print(f"    {mode}: {invokers}")
    
    # Branch durations
    print(f"\n  Average Per-Function Duration (ms):")
    functions = ['ImageLoader', 'Thumbnail', 'Transform', 'Filters', 'Contour', 'Publisher']
    
    for func in functions:
        classic_dur = statistics.mean([r.per_function_ms.get(func, 0) for r in classic_ok]) if classic_ok else 0
        future_dur = statistics.mean([r.per_function_ms.get(func, 0) for r in future_ok]) if future_ok else 0
        print(f"    {func:<20} {classic_dur:>10.1f} {future_dur:>10.1f}")
    
    print("\n" + "-" * 70)
    print("  EXPECTED BEHAVIOR:")
    print("-" * 70)
    print("  CLASSIC:      Contour (slowest) triggers Publisher")
    print("  FUTURE_BASED: Thumbnail (fastest) triggers Publisher")
    
    variance = statistics.mean([r.branch_variance_ms for r in classic_ok]) if classic_ok else 0
    print(f"\n  Branch variance: {variance:.0f}ms (expected latency saving in FUTURE mode)")
    print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Image Pipeline Benchmark')
    parser.add_argument('--iterations', type=int, default=3, help='Iterations per mode')
    parser.add_argument('--mode', choices=['CLASSIC', 'FUTURE_BASED'], help='Single mode to test')
    parser.add_argument('--no-cold', action='store_true', help='Skip cold start forcing')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  IMAGE PIPELINE QUICK BENCHMARK")
    print("=" * 70)
    print(f"  Test image: s3://{TEST_BUCKET}/{TEST_KEY}")
    print(f"  Iterations: {args.iterations}")
    
    if args.mode:
        results = run_benchmark(args.mode, args.iterations, not args.no_cold)
        avg = statistics.mean([r.e2e_latency_ms for r in results if not r.error])
        print(f"\n  {args.mode} Average: {avg:.1f}ms")
    else:
        classic_results = run_benchmark('CLASSIC', args.iterations, not args.no_cold)
        future_results = run_benchmark('FUTURE_BASED', args.iterations, not args.no_cold)
        print_summary(classic_results, future_results)
    
    # Save results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"quick_benchmark_{timestamp}.json"
    
    all_results = {
        'timestamp': timestamp,
        'classic': [{'e2e_ms': r.e2e_latency_ms, 'invoker': r.invoker_branch} 
                   for r in (classic_results if not args.mode else [])],
        'future': [{'e2e_ms': r.e2e_latency_ms, 'invoker': r.invoker_branch}
                  for r in (future_results if not args.mode else results if args.mode == 'FUTURE_BASED' else [])]
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n  ✓ Results saved to {results_file}")


if __name__ == "__main__":
    main()
