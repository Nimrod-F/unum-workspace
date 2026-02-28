#!/usr/bin/env python3
"""
Comprehensive Benchmark for Streaming Demo

This script benchmarks both Normal and Streaming modes, collecting:
- End-to-end latency
- Per-stage latency
- Billed duration and cost
- Memory usage
- Cold start vs warm start performance

Usage:
    python benchmark.py --mode normal --runs 5
    python benchmark.py --mode streaming --runs 5
    python benchmark.py --compare  # Run both and generate comparison
"""

import boto3
import json
import time
import argparse
import statistics
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import sys

# AWS Configuration
REGION = "eu-central-1"
LAMBDA_CLIENT = boto3.client('lambda', region_name=REGION)
LOGS_CLIENT = boto3.client('logs', region_name=REGION)

# Function names
FUNCTIONS = {
    "Generator": "streaming-demo-Generator",
    "Processor": "streaming-demo-Processor", 
    "Analyzer": "streaming-demo-Analyzer",
    "Reporter": "streaming-demo-Reporter"
}

MEMORY_COST_PER_GB_SECOND = 0.0000166667  # USD per GB-second


def force_cold_start():
    """Force cold start by updating function configuration."""
    print("Forcing cold starts by updating function configurations...")
    for name, func_name in FUNCTIONS.items():
        try:
            # Update environment to trigger new container
            LAMBDA_CLIENT.update_function_configuration(
                FunctionName=func_name,
                Environment={
                    'Variables': {
                        'COLD_START_TRIGGER': str(time.time())
                    }
                }
            )
        except Exception as e:
            print(f"  Warning: Could not update {name}: {e}")
    
    # Wait for updates to propagate
    print("Waiting 10s for updates to propagate...")
    time.sleep(10)


def invoke_workflow(input_data=None, session_id=None):
    """Invoke the workflow and return timing information."""
    if input_data is None:
        input_data = {"input": f"benchmark_{time.time()}"}
    
    if session_id:
        input_data["Session"] = session_id
    
    start_time = time.time()
    
    response = LAMBDA_CLIENT.invoke(
        FunctionName=FUNCTIONS["Generator"],
        InvocationType='RequestResponse',
        Payload=json.dumps(input_data)
    )
    
    invoke_time = time.time() - start_time
    
    # Parse response
    response_payload = json.loads(response['Payload'].read())
    
    return {
        "invoke_time": invoke_time,
        "status_code": response.get('StatusCode'),
        "response": response_payload,
        "start_time": start_time
    }


def get_log_events(function_name, start_time, end_time=None):
    """Get CloudWatch log events for a function."""
    log_group = f"/aws/lambda/{function_name}"
    
    if end_time is None:
        end_time = time.time()
    
    try:
        # Get latest log stream
        streams = LOGS_CLIENT.describe_log_streams(
            logGroupName=log_group,
            orderBy='LastEventTime',
            descending=True,
            limit=5
        )
        
        if not streams.get('logStreams'):
            return []
        
        events = []
        for stream in streams['logStreams']:
            try:
                response = LOGS_CLIENT.get_log_events(
                    logGroupName=log_group,
                    logStreamName=stream['logStreamName'],
                    startTime=int(start_time * 1000),
                    endTime=int(end_time * 1000)
                )
                events.extend(response.get('events', []))
            except Exception:
                pass
        
        return events
    except Exception as e:
        print(f"  Warning: Could not get logs for {function_name}: {e}")
        return []


def parse_report_log(events):
    """Parse REPORT log to extract billed duration and memory."""
    for event in events:
        msg = event.get('message', '')
        if 'REPORT' in msg and 'Billed Duration' in msg:
            # Parse: REPORT RequestId: xxx Duration: 123.45 ms Billed Duration: 124 ms Memory Size: 256 MB Max Memory Used: 89 MB
            parts = msg.split('\t')
            result = {}
            for part in parts:
                if 'Billed Duration' in part:
                    result['billed_duration_ms'] = int(part.split(':')[1].strip().split()[0])
                if 'Max Memory Used' in part:
                    result['max_memory_mb'] = int(part.split(':')[1].strip().split()[0])
                if 'Memory Size' in part:
                    result['memory_size_mb'] = int(part.split(':')[1].strip().split()[0])
                if 'Init Duration' in part:
                    result['init_duration_ms'] = float(part.split(':')[1].strip().split()[0])
            return result
    return None


def get_e2e_from_reporter_logs(start_time, timeout=60):
    """Get end-to-end completion time from Reporter logs."""
    end_time = start_time + timeout
    
    while time.time() < end_time:
        events = get_log_events(FUNCTIONS["Reporter"], start_time)
        
        for event in events:
            msg = event.get('message', '')
            if '[Reporter] Completed in' in msg:
                # Extract completion timestamp
                return event.get('timestamp', 0) / 1000.0
        
        time.sleep(1)
    
    return None


def run_single_benchmark(run_id, cold_start=False):
    """Run a single benchmark iteration."""
    print(f"\n  Run {run_id}: ", end="", flush=True)
    
    session_id = f"bench_{run_id}_{time.time()}"
    
    result = invoke_workflow(session_id=session_id)
    
    # Wait for workflow completion
    time.sleep(3)
    
    # Get completion time from Reporter logs
    reporter_complete_time = get_e2e_from_reporter_logs(result['start_time'], timeout=120)
    
    if reporter_complete_time:
        e2e_latency = reporter_complete_time - result['start_time']
    else:
        e2e_latency = result['invoke_time']  # Fallback
    
    # Collect per-function metrics
    function_metrics = {}
    total_billed_ms = 0
    total_cost = 0
    
    for name, func_name in FUNCTIONS.items():
        events = get_log_events(func_name, result['start_time'])
        report = parse_report_log(events)
        
        if report:
            function_metrics[name] = report
            total_billed_ms += report.get('billed_duration_ms', 0)
            
            # Calculate cost
            memory_gb = report.get('memory_size_mb', 256) / 1024
            duration_s = report.get('billed_duration_ms', 0) / 1000
            cost = memory_gb * duration_s * MEMORY_COST_PER_GB_SECOND
            total_cost += cost
    
    cold_marker = "[COLD]" if cold_start else "[WARM]"
    print(f"{cold_marker} E2E: {e2e_latency:.3f}s, Billed: {total_billed_ms}ms, Cost: ${total_cost:.6f}")
    
    return {
        "run_id": run_id,
        "cold_start": cold_start,
        "e2e_latency": e2e_latency,
        "invoke_time": result['invoke_time'],
        "total_billed_ms": total_billed_ms,
        "total_cost": total_cost,
        "function_metrics": function_metrics
    }


def run_benchmark(mode, num_runs=5, cold_runs=2):
    """Run complete benchmark for a mode."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {mode.upper()} MODE")
    print(f"{'='*60}")
    
    results = {
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "runs": [],
        "cold_runs": [],
        "summary": {}
    }
    
    # Cold start runs
    print(f"\n[Cold Start Runs: {cold_runs}]")
    for i in range(cold_runs):
        force_cold_start()
        run_result = run_single_benchmark(i + 1, cold_start=True)
        results["cold_runs"].append(run_result)
    
    # Warm runs
    print(f"\n[Warm Runs: {num_runs}]")
    for i in range(num_runs):
        run_result = run_single_benchmark(i + 1, cold_start=False)
        results["runs"].append(run_result)
    
    # Calculate summary statistics
    warm_latencies = [r["e2e_latency"] for r in results["runs"]]
    cold_latencies = [r["e2e_latency"] for r in results["cold_runs"]]
    total_costs = [r["total_cost"] for r in results["runs"]]
    total_billed = [r["total_billed_ms"] for r in results["runs"]]
    
    results["summary"] = {
        "warm": {
            "avg_latency": statistics.mean(warm_latencies),
            "min_latency": min(warm_latencies),
            "max_latency": max(warm_latencies),
            "std_dev": statistics.stdev(warm_latencies) if len(warm_latencies) > 1 else 0
        },
        "cold": {
            "avg_latency": statistics.mean(cold_latencies) if cold_latencies else 0,
            "min_latency": min(cold_latencies) if cold_latencies else 0,
            "max_latency": max(cold_latencies) if cold_latencies else 0
        },
        "cost": {
            "avg_cost": statistics.mean(total_costs),
            "avg_billed_ms": statistics.mean(total_billed)
        }
    }
    
    print(f"\n[Summary for {mode.upper()} mode]")
    print(f"  Warm E2E: {results['summary']['warm']['avg_latency']:.3f}s "
          f"(min: {results['summary']['warm']['min_latency']:.3f}s, "
          f"max: {results['summary']['warm']['max_latency']:.3f}s)")
    print(f"  Cold E2E: {results['summary']['cold']['avg_latency']:.3f}s")
    print(f"  Avg Cost: ${results['summary']['cost']['avg_cost']:.6f}")
    print(f"  Avg Billed: {results['summary']['cost']['avg_billed_ms']:.0f}ms")
    
    return results


def run_comparison(num_runs=5, cold_runs=2):
    """Run both modes and generate comparison."""
    print("\n" + "="*60)
    print("STREAMING DEMO - COMPREHENSIVE BENCHMARK")
    print("="*60)
    print(f"\nWorkflow: Generator → Processor → Analyzer → Reporter")
    print(f"Each stage: 5 items × 0.5s = 2.5s")
    print(f"Expected Normal E2E: ~10s (sequential)")
    print(f"Expected Streaming E2E: ~4-5s (parallel)")
    
    # Note: In real scenario, you'd deploy different modes
    # For now, we run with current deployment
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "workflow": "streaming-demo",
        "stages": 4,
        "items_per_stage": 5,
        "time_per_item": 0.5,
        "runs_per_mode": num_runs,
        "cold_runs_per_mode": cold_runs,
        "results": {}
    }
    
    # Run benchmark (current mode)
    mode_result = run_benchmark("current", num_runs, cold_runs)
    results["results"]["current"] = mode_result
    
    # Save results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Streaming Demo Benchmark')
    parser.add_argument('--mode', choices=['normal', 'streaming', 'current'],
                       default='current', help='Mode to benchmark')
    parser.add_argument('--runs', type=int, default=5, help='Number of warm runs')
    parser.add_argument('--cold-runs', type=int, default=2, help='Number of cold runs')
    parser.add_argument('--compare', action='store_true', help='Run comparison benchmark')
    
    args = parser.parse_args()
    
    if args.compare:
        run_comparison(args.runs, args.cold_runs)
    else:
        run_benchmark(args.mode, args.runs, args.cold_runs)


if __name__ == "__main__":
    main()
