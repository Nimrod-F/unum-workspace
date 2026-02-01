#!/usr/bin/env python3
"""
Multi-Source Dashboard Benchmark Runner

This script runs comprehensive benchmarks for the multi-source dashboard
application to demonstrate the benefits of Future-Based execution.

Usage:
    python run_benchmark.py --mode FUTURE_BASED --iterations 10
    python run_benchmark.py --mode all --iterations 20 --output-file results.json
"""

import boto3
import json
import time
import argparse
import os
import yaml
import statistics
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict


# ============================================================
# Configuration
# ============================================================

def load_config(config_file: str = "config.yaml") -> dict:
    """Load benchmark configuration from YAML file."""
    config_path = Path(__file__).parent / config_file
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


CONFIG = load_config()

REGION = CONFIG['aws']['region']
STACK_NAME = CONFIG['aws']['stack_name']
ENTRY_FUNCTION = CONFIG['functions']['entry']
TERMINAL_FUNCTION = CONFIG['functions']['terminal']
LOG_GROUP_PREFIX = CONFIG['cloudwatch']['log_group_prefix']


# ============================================================
# Data Classes
# ============================================================

@dataclass
class BenchmarkRun:
    """Single benchmark run data."""
    run_id: int
    mode: str
    run_type: str  # 'cold' or 'warm'
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    success: bool = False
    error: Optional[str] = None

    # Per-function metrics
    function_latencies: Dict[str, float] = field(default_factory=dict)
    cold_starts: List[str] = field(default_factory=list)

    # Fan-in specific metrics
    pre_resolved_count: int = 0
    aggregator_latency_ms: float = 0.0

    # Memory metrics
    max_memory_mb: int = 0
    aggregator_memory_mb: int = 0


@dataclass
class BenchmarkSummary:
    """Summary statistics for a mode."""
    mode: str
    total_runs: int
    successful_runs: int
    failed_runs: int

    # Latency statistics
    e2e_mean_ms: float = 0.0
    e2e_median_ms: float = 0.0
    e2e_std_ms: float = 0.0
    e2e_min_ms: float = 0.0
    e2e_max_ms: float = 0.0
    e2e_p95_ms: float = 0.0

    # Cold vs Warm
    cold_mean_ms: float = 0.0
    warm_mean_ms: float = 0.0

    # Future-specific
    avg_pre_resolved: float = 0.0

    # Memory
    avg_aggregator_memory_mb: float = 0.0


# ============================================================
# Benchmark Runner
# ============================================================

class DashboardBenchmarkRunner:
    """Runs benchmarks for multi-source dashboard."""

    def __init__(self, region: str = REGION):
        self.region = region
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.logs_client = boto3.client('logs', region_name=region)
        self.cf_client = boto3.client('cloudformation', region_name=region)

        self.functions: Dict[str, str] = {}
        self.discover_functions()

    def discover_functions(self):
        """Discover Lambda function ARNs from CloudFormation stack."""
        print(f"Discovering functions from stack: {STACK_NAME}")
        try:
            response = self.cf_client.describe_stack_resources(StackName=STACK_NAME)
            for resource in response['StackResources']:
                if resource['ResourceType'] == 'AWS::Lambda::Function':
                    logical_id = resource['LogicalResourceId']
                    physical_id = resource['PhysicalResourceId']
                    # Remove 'Function' suffix to get clean name
                    func_name = logical_id.replace('Function', '')
                    self.functions[func_name] = physical_id
            print(f"  Discovered {len(self.functions)} functions")
        except Exception as e:
            print(f"  Error discovering functions: {e}")
            raise

    def set_mode(self, mode: str):
        """Configure all functions for specified execution mode."""
        print(f"\nSetting mode to {mode}...")

        mode_config = {
            'CLASSIC': {'EAGER': 'false', 'UNUM_FUTURE_BASED': 'false'},
            'EAGER': {'EAGER': 'true', 'UNUM_FUTURE_BASED': 'false'},
            'FUTURE_BASED': {'EAGER': 'true', 'UNUM_FUTURE_BASED': 'true'}
        }

        config = mode_config[mode]

        for func_name, func_arn in self.functions.items():
            try:
                response = self.lambda_client.get_function_configuration(
                    FunctionName=func_arn
                )
                env = response.get('Environment', {}).get('Variables', {})
                env.update(config)

                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={'Variables': env}
                )
            except Exception as e:
                print(f"  Warning: Failed to update {func_name}: {e}")

        time.sleep(5)  # Wait for configuration updates
        self._wait_for_functions_ready()

    def _wait_for_functions_ready(self, timeout: int = 60):
        """Wait for all functions to be in Active state."""
        start = time.time()
        for func_name, func_arn in self.functions.items():
            while (time.time() - start) < timeout:
                try:
                    response = self.lambda_client.get_function(FunctionName=func_arn)
                    state = response['Configuration']['State']
                    status = response['Configuration'].get('LastUpdateStatus', 'Successful')
                    if state == 'Active' and status == 'Successful':
                        break
                except:
                    pass
                time.sleep(1)

    def force_cold_start(self):
        """Force cold start by updating environment."""
        timestamp = str(int(time.time()))
        for func_name, func_arn in self.functions.items():
            try:
                response = self.lambda_client.get_function_configuration(
                    FunctionName=func_arn
                )
                env = response.get('Environment', {}).get('Variables', {})
                env['FORCE_COLD'] = timestamp

                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={'Variables': env}
                )
            except Exception as e:
                pass

        time.sleep(5)
        self._wait_for_functions_ready()

    def invoke_dashboard(self) -> tuple:
        """Invoke the dashboard workflow."""
        session_id = f"bench-{int(time.time() * 1000)}"

        payload = {
            "Data": {
                "Source": "http",
                "Value": {
                    "request_id": session_id,
                    "dashboard_type": "executive",
                    "time_range": "24h"
                }
            },
            "Session": session_id
        }

        entry_arn = self.functions.get(ENTRY_FUNCTION)
        if not entry_arn:
            raise ValueError(f"Entry function {ENTRY_FUNCTION} not found")

        start_time = time.time()
        response = self.lambda_client.invoke(
            FunctionName=entry_arn,
            InvocationType='Event',
            Payload=json.dumps(payload)
        )

        return session_id, start_time

    def wait_for_completion(self, start_time: float, timeout: int = 120) -> tuple:
        """Wait for workflow completion."""
        terminal_arn = self.functions.get(TERMINAL_FUNCTION)
        if not terminal_arn:
            return False, time.time()

        log_group = f"{LOG_GROUP_PREFIX}{terminal_arn}"
        start_ms = int(start_time * 1000)

        completion_patterns = [
            '"aggregation_complete"',
            '"merge_complete"'
        ]

        deadline = time.time() + timeout
        while time.time() < deadline:
            for pattern in completion_patterns:
                try:
                    response = self.logs_client.filter_log_events(
                        logGroupName=log_group,
                        startTime=start_ms,
                        filterPattern=pattern
                    )

                    if response.get('events'):
                        for event in response['events']:
                            if event.get('timestamp', 0) >= start_ms:
                                return True, time.time()
                except Exception:
                    pass

            time.sleep(0.5)

        return False, time.time()

    def collect_metrics(self, session_id: str, start_time: float,
                       end_time: float) -> BenchmarkRun:
        """Collect metrics from CloudWatch logs."""
        run = BenchmarkRun(
            run_id=0,
            mode='',
            run_type='',
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            e2e_latency_ms=(end_time - start_time) * 1000
        )

        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + 60000  # +60s buffer

        for func_name, func_arn in self.functions.items():
            log_group = f"{LOG_GROUP_PREFIX}{func_arn}"

            try:
                # Get function completion logs
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=start_ms,
                    endTime=end_ms,
                    filterPattern='latency_ms'
                )

                for event in response.get('events', []):
                    try:
                        msg = json.loads(event['message'])
                        if 'latency_ms' in msg:
                            run.function_latencies[func_name] = msg['latency_ms']
                    except:
                        pass

                # Check for cold starts
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=start_ms,
                    endTime=end_ms,
                    filterPattern='Init Duration'
                )

                if response.get('events'):
                    run.cold_starts.append(func_name)

                # Get aggregator-specific metrics
                if TERMINAL_FUNCTION in func_name:
                    response = self.logs_client.filter_log_events(
                        logGroupName=log_group,
                        startTime=start_ms,
                        endTime=end_ms,
                        filterPattern='pre_resolved'
                    )

                    for event in response.get('events', []):
                        try:
                            msg = json.loads(event['message'])
                            if 'pre_resolved' in msg:
                                run.pre_resolved_count = msg['pre_resolved']
                        except:
                            pass
            except Exception as e:
                pass

        return run

    def run_benchmark(self, mode: str, iterations: int,
                     cold_iterations: int = 2) -> BenchmarkSummary:
        """Run complete benchmark for a mode."""
        print(f"\n{'='*60}")
        print(f"Benchmark: {mode} mode")
        print(f"{'='*60}")

        self.set_mode(mode)

        runs: List[BenchmarkRun] = []
        warm_iterations = iterations - cold_iterations

        # Cold start runs
        print(f"\nCold start runs ({cold_iterations}):")
        for i in range(cold_iterations):
            print(f"  Run {i+1}/{cold_iterations}...", end=' ', flush=True)
            self.force_cold_start()
            time.sleep(2)

            try:
                session_id, start_time = self.invoke_dashboard()
                success, end_time = self.wait_for_completion(start_time)

                if success:
                    time.sleep(CONFIG['cloudwatch']['metrics_delay_sec'])
                    run = self.collect_metrics(session_id, start_time, end_time)
                    run.run_id = i + 1
                    run.mode = mode
                    run.run_type = 'cold'
                    run.success = True
                    runs.append(run)
                    print(f"✓ {run.e2e_latency_ms:.0f}ms (pre-resolved: {run.pre_resolved_count})")
                else:
                    print("✗ Timeout")
            except Exception as e:
                print(f"✗ Error: {e}")

        # Warm start runs
        print(f"\nWarm start runs ({warm_iterations}):")
        time.sleep(5)

        for i in range(warm_iterations):
            print(f"  Run {i+1}/{warm_iterations}...", end=' ', flush=True)

            try:
                session_id, start_time = self.invoke_dashboard()
                success, end_time = self.wait_for_completion(start_time)

                if success:
                    time.sleep(CONFIG['cloudwatch']['metrics_delay_sec'])
                    run = self.collect_metrics(session_id, start_time, end_time)
                    run.run_id = cold_iterations + i + 1
                    run.mode = mode
                    run.run_type = 'warm'
                    run.success = True
                    runs.append(run)
                    print(f"✓ {run.e2e_latency_ms:.0f}ms (pre-resolved: {run.pre_resolved_count})")
                else:
                    print("✗ Timeout")

                time.sleep(CONFIG['benchmark']['delay_between_runs_sec'])
            except Exception as e:
                print(f"✗ Error: {e}")

        # Compute summary
        summary = self._compute_summary(mode, runs)

        return summary, runs

    def _compute_summary(self, mode: str, runs: List[BenchmarkRun]) -> BenchmarkSummary:
        """Compute summary statistics."""
        successful = [r for r in runs if r.success]

        summary = BenchmarkSummary(
            mode=mode,
            total_runs=len(runs),
            successful_runs=len(successful),
            failed_runs=len(runs) - len(successful)
        )

        if successful:
            latencies = [r.e2e_latency_ms for r in successful]
            summary.e2e_mean_ms = statistics.mean(latencies)
            summary.e2e_median_ms = statistics.median(latencies)
            summary.e2e_std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0
            summary.e2e_min_ms = min(latencies)
            summary.e2e_max_ms = max(latencies)

            cold_runs = [r for r in successful if r.run_type == 'cold']
            warm_runs = [r for r in successful if r.run_type == 'warm']

            if cold_runs:
                summary.cold_mean_ms = statistics.mean([r.e2e_latency_ms for r in cold_runs])
            if warm_runs:
                summary.warm_mean_ms = statistics.mean([r.e2e_latency_ms for r in warm_runs])

            if len(latencies) >= 20:
                sorted_lat = sorted(latencies)
                summary.e2e_p95_ms = sorted_lat[int(len(sorted_lat) * 0.95)]

            summary.avg_pre_resolved = statistics.mean([r.pre_resolved_count for r in successful])

        return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Source Dashboard Benchmark Runner'
    )
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'CLASSIC', 'EAGER', 'FUTURE_BASED'],
                       help='Execution mode to benchmark')
    parser.add_argument('--iterations', type=int,
                       default=CONFIG['benchmark']['iterations'],
                       help='Total iterations per mode')
    parser.add_argument('--cold-iterations', type=int,
                       default=CONFIG['benchmark']['cold_start_runs'],
                       help='Number of cold start runs')
    parser.add_argument('--output-file', type=str,
                       help='Output file for results (JSON)')

    args = parser.parse_args()

    runner = DashboardBenchmarkRunner()

    modes = ['CLASSIC', 'EAGER', 'FUTURE_BASED'] if args.mode == 'all' else [args.mode]

    all_results = {}

    for mode in modes:
        summary, runs = runner.run_benchmark(
            mode=mode,
            iterations=args.iterations,
            cold_iterations=args.cold_iterations
        )

        all_results[mode] = {
            'summary': asdict(summary),
            'runs': [asdict(r) for r in runs]
        }

        # Print summary
        print(f"\n{mode} Summary:")
        print(f"  Mean E2E: {summary.e2e_mean_ms:.0f}ms ± {summary.e2e_std_ms:.0f}ms")
        print(f"  Cold: {summary.cold_mean_ms:.0f}ms, Warm: {summary.warm_mean_ms:.0f}ms")
        print(f"  Pre-resolved avg: {summary.avg_pre_resolved:.1f}")

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        results_dir = Path(CONFIG['output']['results_dir'])
        results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_path = results_dir / f"benchmark_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
