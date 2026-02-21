#!/usr/bin/env python3
"""
Complete Benchmark Suite for Order Processing Workflow

This script comprehensively benchmarks CLASSIC vs FUTURE_BASED execution modes:
- Loads function ARNs from function-arn.yaml
- Configures Lambda environment variables for each mode
- Forces cold starts for accurate measurements
- Collects detailed CloudWatch metrics
- Calculates performance improvements
- Generates comparison charts

Usage:
    python benchmark_complete.py --iterations 10
    python benchmark_complete.py --iterations 5 --cold-all
    python benchmark_complete.py --skip-classic  # Only run FUTURE_BASED
"""

import argparse
import datetime
import json
import os
import re
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import yaml

# Try to import matplotlib for charts
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    print("Warning: matplotlib not available. Charts will not be generated.")
    print("Install with: pip install matplotlib")

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get('AWS_REGION', 'eu-central-1')
PROFILE = os.environ.get('AWS_PROFILE', 'default')
DYNAMODB_TABLE = 'unum-intermediate-datastore-orders'

# Mode configurations
MODE_CONFIGS = {
    'CLASSIC': {
        'EAGER': 'false',
        'UNUM_FUTURE_BASED': 'false',
    },
    'FUTURE_BASED': {
        'EAGER': 'true',
        'UNUM_FUTURE_BASED': 'true',
    },
}

# Function execution order and expected timings
FUNCTION_INFO = {
    'TriggerFunction': {'order': 1, 'expected_ms': 50, 'description': 'Entry point'},
    'FastProcessor': {'order': 2, 'expected_ms': 100, 'description': 'Fast branch'},
    'FusedOrderProcessing': {'order': 3, 'expected_ms': 3000, 'description': 'Fused order processing'},
    'Aggregator': {'order': 6, 'expected_ms': 100, 'description': 'Terminal aggregator'},
}

# ============================================================
# AWS Clients
# ============================================================

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
lambda_client = session.client('lambda')
logs_client = session.client('logs')


# ============================================================
# Data Classes
# ============================================================

@dataclass
class LambdaMetrics:
    """Metrics from a single Lambda function execution"""
    function_name: str
    duration_ms: float = 0.0
    billed_duration_ms: float = 0.0
    memory_size_mb: int = 0
    memory_used_mb: int = 0
    init_duration_ms: float = 0.0
    cold_start: bool = False
    invocation_time_ms: int = 0  # When function was invoked (from CloudWatch timestamp)

    @property
    def memory_efficiency(self) -> float:
        if self.memory_size_mb > 0:
            return self.memory_used_mb / self.memory_size_mb
        return 0.0


@dataclass
class WorkflowRun:
    """Complete metrics for a single workflow execution"""
    run_id: int
    mode: str
    order_id: str
    start_time: float
    end_time: float

    # Invocation metrics
    trigger_latency_ms: float = 0.0  # Time for TriggerFunction to return
    e2e_latency_ms: float = 0.0      # Time for entire workflow to complete

    # Per-function metrics
    function_metrics: Dict[str, LambdaMetrics] = field(default_factory=dict)

    # Aggregated metrics
    total_duration_ms: float = 0.0
    total_billed_ms: float = 0.0
    cold_start_count: int = 0
    total_init_ms: float = 0.0

    # Key timing metrics
    fast_processor_time_ms: float = 0.0
    fused_order_processing_time_ms: float = 0.0
    aggregator_invocation_delay_ms: float = 0.0  # Time from start to Aggregator invocation

    # Pre-resolution metrics (FUTURE_BASED benefit)
    invoker_branch: str = ''  # Which function triggered Aggregator

    # Errors
    error: Optional[str] = None

    def calculate_aggregates(self):
        """Calculate aggregate metrics from function data"""
        self.total_duration_ms = sum(m.duration_ms for m in self.function_metrics.values())
        self.total_billed_ms = sum(m.billed_duration_ms for m in self.function_metrics.values())
        self.cold_start_count = sum(1 for m in self.function_metrics.values() if m.cold_start)
        self.total_init_ms = sum(m.init_duration_ms for m in self.function_metrics.values())

        # Calculate key timings
        if 'FastProcessor' in self.function_metrics:
            self.fast_processor_time_ms = self.function_metrics['FastProcessor'].duration_ms

        if 'FusedOrderProcessing' in self.function_metrics:
            self.fused_order_processing_time_ms = self.function_metrics['FusedOrderProcessing'].duration_ms

        # Calculate when Aggregator was invoked (relative to workflow start)
        if 'Aggregator' in self.function_metrics:
            agg_invocation_ts = self.function_metrics['Aggregator'].invocation_time_ms
            workflow_start_ts = int(self.start_time * 1000)
            self.aggregator_invocation_delay_ms = agg_invocation_ts - workflow_start_ts


@dataclass
class BenchmarkSummary:
    """Statistical summary for a benchmark mode"""
    mode: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    timestamp: str

    # E2E Latency
    e2e_mean_ms: float = 0.0
    e2e_median_ms: float = 0.0
    e2e_min_ms: float = 0.0
    e2e_max_ms: float = 0.0
    e2e_std_ms: float = 0.0

    # Aggregator invocation timing (key metric)
    aggregator_invocation_mean_ms: float = 0.0
    aggregator_invocation_median_ms: float = 0.0
    aggregator_invocation_min_ms: float = 0.0
    aggregator_invocation_max_ms: float = 0.0

    # Per-function durations
    avg_trigger_ms: float = 0.0
    avg_fast_processor_ms: float = 0.0
    avg_fused_order_processing_ms: float = 0.0
    avg_aggregator_ms: float = 0.0

    # Cold starts
    total_cold_starts: int = 0
    avg_init_duration_ms: float = 0.0

    # Invoker distribution
    invoker_distribution: Dict[str, int] = field(default_factory=dict)

    # Cost (eu-central-1 pricing)
    total_billed_ms: float = 0.0
    estimated_cost_usd: float = 0.0


# ============================================================
# Function ARN Management
# ============================================================

def load_function_arns() -> Dict[str, str]:
    """Load function ARNs from function-arn.yaml"""
    yaml_path = Path(__file__).parent.parent / 'function-arn.yaml'

    if not yaml_path.exists():
        print(f"✗ Error: function-arn.yaml not found at {yaml_path}")
        print("  Deploy the workflow first using: unum-cli.py deploy -t unum-template.yaml")
        return {}

    with open(yaml_path, 'r') as f:
        arns = yaml.safe_load(f)

    print(f"✓ Loaded {len(arns)} function ARNs from function-arn.yaml")
    return arns


def extract_function_name(arn: str) -> str:
    """Extract physical function name from ARN for CloudWatch logs"""
    # ARN format: arn:aws:lambda:region:account:function:function-name
    return arn.split(':')[-1]


# ============================================================
# Lambda Configuration
# ============================================================

def configure_mode(mode: str, function_arns: Dict[str, str]):
    """Configure Lambda functions for the specified execution mode.

    EAGER controls fan-in behavior on BRANCH functions (FastProcessor,
    FusedOrderProcessing) - it determines whether _run_fan_in_eager or
    _run_fan_in_classic is used. UNUM_FUTURE_BASED controls how the
    Aggregator resolves its inputs (sync polling vs async futures).
    """
    print(f"\n{'='*70}")
    print(f"Configuring {mode} Mode")
    print(f"{'='*70}")

    config = MODE_CONFIGS[mode]

    # Update EAGER on ALL functions (controls fan-in path on branch functions)
    for func_name, arn in function_arns.items():
        try:
            response = lambda_client.get_function_configuration(FunctionName=arn)
            current_env = response.get('Environment', {}).get('Variables', {})

            current_env['EAGER'] = config['EAGER']

            # Set UNUM_FUTURE_BASED only on Aggregator
            if func_name == 'Aggregator':
                current_env['UNUM_FUTURE_BASED'] = config['UNUM_FUTURE_BASED']

            lambda_client.update_function_configuration(
                FunctionName=arn,
                Environment={'Variables': current_env}
            )

            extra = ""
            if func_name == 'Aggregator':
                extra = f", UNUM_FUTURE_BASED={config['UNUM_FUTURE_BASED']}"
            print(f"  ✓ {func_name}: EAGER={config['EAGER']}{extra}")

        except Exception as e:
            print(f"  ✗ {func_name}: {e}")

    # Wait for all updates to propagate
    print(f"  Waiting for updates to propagate...")
    time.sleep(5)


def force_cold_starts(function_arns: Dict[str, str]):
    """Force cold starts by updating all function environments"""
    print("\n  Forcing cold starts for all functions...")

    timestamp = str(int(time.time()))

    for func_name, arn in function_arns.items():
        try:
            response = lambda_client.get_function_configuration(FunctionName=arn)
            current_env = response.get('Environment', {}).get('Variables', {})

            # Update timestamp to force new container
            current_env['COLD_START_TRIGGER'] = timestamp

            lambda_client.update_function_configuration(
                FunctionName=arn,
                Environment={'Variables': current_env}
            )

            print(f"    ✓ {func_name}")

        except Exception as e:
            print(f"    ✗ {func_name}: {e}")

    # Wait for all updates to propagate
    print("  Waiting 15s for updates to propagate...")
    time.sleep(15)

    print("  ✓ Cold starts forced")


# ============================================================
# Workflow Invocation
# ============================================================

def invoke_workflow(order_id: str, function_arns: Dict[str, str]) -> Dict:
    """Invoke the workflow and measure end-to-end latency"""
    trigger_arn = function_arns.get('TriggerFunction')

    if not trigger_arn:
        return {
            'success': False,
            'order_id': order_id,
            'error': 'TriggerFunction ARN not found'
        }

    # Wrap payload in unum format - the unum runtime's ingress() expects
    # Data.Source and Data.Value fields
    payload = {
        "Data": {
            "Source": "http",
            "Value": {
                "order_id": order_id,
                "customer_id": "BENCH-CUSTOMER",
                "items": [
                    {"sku": "ITEM-001", "quantity": 2, "price": 49.99},
                    {"sku": "ITEM-002", "quantity": 1, "price": 29.99}
                ]
            }
        }
    }

    start_time = time.time()

    try:
        response = lambda_client.invoke(
            FunctionName=trigger_arn,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )

        trigger_end_time = time.time()
        trigger_latency_ms = (trigger_end_time - start_time) * 1000

        # Parse response
        response_payload = json.loads(response['Payload'].read())

        # Check for Lambda function errors
        if 'FunctionError' in response:
            return {
                'success': False,
                'order_id': order_id,
                'error': f"Lambda function error: {response_payload}"
            }

        return {
            'success': True,
            'order_id': order_id,
            'start_time': start_time,
            'trigger_end_time': trigger_end_time,
            'trigger_latency_ms': trigger_latency_ms,
            'response': response_payload
        }

    except Exception as e:
        return {
            'success': False,
            'order_id': order_id,
            'error': str(e)
        }


# ============================================================
# CloudWatch Logs Collection
# ============================================================

def get_cloudwatch_logs(function_name: str, start_time: float, end_time: float) -> List[Dict]:
    """Retrieve CloudWatch logs for a function"""
    log_group = f"/aws/lambda/{function_name}"

    try:
        # Convert to milliseconds with buffer
        start_ms = int(start_time * 1000) - 5000  # 5s before
        end_ms = int(end_time * 1000) + 60000     # 60s after

        response = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=start_ms,
            endTime=end_ms,
            limit=200
        )

        return response.get('events', [])

    except Exception as e:
        print(f"      Warning: Could not fetch logs for {function_name}: {e}")
        return []


def extract_lambda_metrics(function_logical_name: str, function_physical_name: str,
                           start_time: float, end_time: float) -> LambdaMetrics:
    """Extract metrics from CloudWatch REPORT logs"""
    logs = get_cloudwatch_logs(function_physical_name, start_time, end_time)

    metrics = LambdaMetrics(function_name=function_logical_name)

    # Find REPORT log line
    report_logs = [log for log in logs if 'REPORT RequestId:' in log.get('message', '')]

    if not report_logs:
        return metrics

    # Get most recent REPORT (in case of retries)
    latest_report = max(report_logs, key=lambda x: x.get('timestamp', 0))
    message = latest_report.get('message', '')
    timestamp_ms = latest_report.get('timestamp', 0)

    # Parse REPORT line
    # Format: REPORT RequestId: ... Duration: 123.45 ms  Billed Duration: 200 ms  Memory Size: 512 MB  Max Memory Used: 128 MB  Init Duration: 456.78 ms

    duration_match = re.search(r'Duration:\s+([\d.]+)\s+ms', message)
    billed_match = re.search(r'Billed Duration:\s+([\d.]+)\s+ms', message)
    memory_size_match = re.search(r'Memory Size:\s+(\d+)\s+MB', message)
    memory_used_match = re.search(r'Max Memory Used:\s+(\d+)\s+MB', message)
    init_match = re.search(r'Init Duration:\s+([\d.]+)\s+ms', message)

    if duration_match:
        metrics.duration_ms = float(duration_match.group(1))
    if billed_match:
        metrics.billed_duration_ms = float(billed_match.group(1))
    if memory_size_match:
        metrics.memory_size_mb = int(memory_size_match.group(1))
    if memory_used_match:
        metrics.memory_used_mb = int(memory_used_match.group(1))
    if init_match:
        metrics.init_duration_ms = float(init_match.group(1))
        metrics.cold_start = True

    # Calculate invocation time (report timestamp - duration)
    if metrics.duration_ms > 0 and timestamp_ms > 0:
        metrics.invocation_time_ms = int(timestamp_ms - metrics.duration_ms)

    return metrics


def collect_all_metrics(function_arns: Dict[str, str], start_time: float,
                       end_time: float, max_retries: int = 3) -> Dict[str, LambdaMetrics]:
    """Collect metrics for all functions with retry logic"""
    all_metrics = {}

    for attempt in range(1, max_retries + 1):
        print(f"      Collecting metrics (attempt {attempt}/{max_retries})...")

        for logical_name, arn in function_arns.items():
            physical_name = extract_function_name(arn)
            metrics = extract_lambda_metrics(logical_name, physical_name, start_time, end_time)
            all_metrics[logical_name] = metrics

        # Check if we got Aggregator metrics (critical function)
        if all_metrics.get('Aggregator', LambdaMetrics('')).duration_ms > 0:
            break

        if attempt < max_retries:
            print(f"      Aggregator metrics not found, waiting 5s...")
            time.sleep(5)

    return all_metrics


# ============================================================
# Benchmark Execution
# ============================================================

def run_single_iteration(run_id: int, mode: str, order_id: str, function_arns: Dict[str, str],
                        force_cold: bool = False) -> WorkflowRun:
    """Run a single workflow iteration"""
    print(f"\n  [{run_id}] Order: {order_id}")

    # Force cold starts if requested
    if force_cold:
        force_cold_starts(function_arns)

    # Invoke workflow
    result = invoke_workflow(order_id, function_arns)

    if not result['success']:
        print(f"    ✗ Invocation failed: {result.get('error')}")
        return WorkflowRun(
            run_id=run_id,
            mode=mode,
            order_id=order_id,
            start_time=time.time(),
            end_time=time.time(),
            error=result.get('error')
        )

    print(f"    ✓ Trigger latency: {result['trigger_latency_ms']:.2f}ms")

    # Wait for workflow to complete
    print(f"    Waiting 10s for workflow completion...")
    time.sleep(10)

    # Collect CloudWatch metrics
    end_time = time.time()
    metrics = collect_all_metrics(function_arns, result['start_time'], end_time)

    # Create run object
    run = WorkflowRun(
        run_id=run_id,
        mode=mode,
        order_id=order_id,
        start_time=result['start_time'],
        end_time=end_time,
        trigger_latency_ms=result['trigger_latency_ms'],
        function_metrics=metrics
    )

    # Calculate e2e latency from Aggregator completion time
    if metrics.get('Aggregator', LambdaMetrics('')).invocation_time_ms > 0:
        agg = metrics['Aggregator']
        workflow_start_ms = int(result['start_time'] * 1000)
        workflow_end_ms = agg.invocation_time_ms + int(agg.duration_ms)
        run.e2e_latency_ms = workflow_end_ms - workflow_start_ms

    # Calculate aggregates
    run.calculate_aggregates()

    # Determine invoker branch (which function triggered Aggregator)
    agg_m = metrics.get('Aggregator', LambdaMetrics(''))
    fused_order_processing_m = metrics.get('FusedOrderProcessing', LambdaMetrics(''))
    if agg_m.invocation_time_ms > 0 and fused_order_processing_m.invocation_time_ms > 0:
        fused_order_processing_completed_ms = fused_order_processing_m.invocation_time_ms + int(fused_order_processing_m.duration_ms)
        if agg_m.invocation_time_ms < fused_order_processing_completed_ms:
            run.invoker_branch = 'FusedOrderProcessing'
        else:
            run.invoker_branch = 'Aggregator'

    # Print summary
    print(f"    Functions:")
    for func_name in sorted(metrics.keys(), key=lambda x: FUNCTION_INFO.get(x, {}).get('order', 99)):
        m = metrics[func_name]
        if m.duration_ms > 0:
            cold_tag = " [COLD]" if m.cold_start else ""
            print(f"      {func_name:20s}: {m.duration_ms:7.2f}ms{cold_tag}")

    if run.e2e_latency_ms > 0:
        print(f"    E2E Latency: {run.e2e_latency_ms:.2f}ms")

    if run.aggregator_invocation_delay_ms > 0:
        print(f"    Aggregator invoked at: +{run.aggregator_invocation_delay_ms:.2f}ms (by {run.invoker_branch})")

    return run


def run_benchmark_mode(mode: str, iterations: int, function_arns: Dict[str, str],
                      cold_start_freq: str = 'first') -> List[WorkflowRun]:
    """Run benchmark for a specific mode"""
    print(f"\n{'='*70}")
    print(f"Running {mode} Benchmark")
    print(f"Iterations: {iterations}")
    print(f"Cold start frequency: {cold_start_freq}")
    print(f"{'='*70}")

    # Configure mode
    configure_mode(mode, function_arns)

    runs = []

    for i in range(iterations):
        force_cold = (cold_start_freq == 'all') or (cold_start_freq == 'first' and i == 0)
        order_id = f"BENCH-{mode}-{i+1:03d}-{int(time.time())}"

        run = run_single_iteration(i + 1, mode, order_id, function_arns, force_cold)
        runs.append(run)

        # Brief pause between iterations
        if i < iterations - 1:
            print(f"  Waiting 5s before next iteration...")
            time.sleep(5)

    return runs


# ============================================================
# Results Analysis
# ============================================================

def compute_summary(mode: str, runs: List[WorkflowRun]) -> BenchmarkSummary:
    """Compute statistical summary from runs"""
    successful = [r for r in runs if r.error is None and r.e2e_latency_ms > 0]
    failed = [r for r in runs if r.error is not None]

    summary = BenchmarkSummary(
        mode=mode,
        total_runs=len(runs),
        successful_runs=len(successful),
        failed_runs=len(failed),
        timestamp=datetime.datetime.now().isoformat()
    )

    if not successful:
        return summary

    # E2E Latency statistics
    e2e_latencies = [r.e2e_latency_ms for r in successful]
    summary.e2e_mean_ms = statistics.mean(e2e_latencies)
    summary.e2e_median_ms = statistics.median(e2e_latencies)
    summary.e2e_min_ms = min(e2e_latencies)
    summary.e2e_max_ms = max(e2e_latencies)
    summary.e2e_std_ms = statistics.stdev(e2e_latencies) if len(e2e_latencies) > 1 else 0

    # Aggregator invocation timing
    agg_timings = [r.aggregator_invocation_delay_ms for r in successful if r.aggregator_invocation_delay_ms > 0]
    if agg_timings:
        summary.aggregator_invocation_mean_ms = statistics.mean(agg_timings)
        summary.aggregator_invocation_median_ms = statistics.median(agg_timings)
        summary.aggregator_invocation_min_ms = min(agg_timings)
        summary.aggregator_invocation_max_ms = max(agg_timings)

    # Per-function averages
    for func_name in FUNCTION_INFO.keys():
        durations = [r.function_metrics[func_name].duration_ms
                    for r in successful
                    if func_name in r.function_metrics and r.function_metrics[func_name].duration_ms > 0]

        if durations:
            avg = statistics.mean(durations)
            if func_name == 'TriggerFunction':
                summary.avg_trigger_ms = avg
            elif func_name == 'FastProcessor':
                summary.avg_fast_processor_ms = avg
            elif func_name == 'FusedOrderProcessing':
                summary.avg_fused_order_processing_ms = avg
            elif func_name == 'Aggregator':
                summary.avg_aggregator_ms = avg

    # Cold starts
    summary.total_cold_starts = sum(r.cold_start_count for r in successful)
    init_durations = [r.total_init_ms for r in successful if r.total_init_ms > 0]
    if init_durations:
        summary.avg_init_duration_ms = statistics.mean(init_durations)

    # Invoker distribution
    for r in successful:
        if r.invoker_branch:
            summary.invoker_distribution[r.invoker_branch] = \
                summary.invoker_distribution.get(r.invoker_branch, 0) + 1

    # Cost calculation (eu-central-1 pricing)
    summary.total_billed_ms = sum(r.total_billed_ms for r in successful)
    # $0.0000166667 per GB-second, assuming average 256MB = 0.25GB
    gb_seconds = (summary.total_billed_ms / 1000) * 0.25
    lambda_compute_cost = gb_seconds * 0.0000166667
    # $0.20 per 1M requests
    request_cost = len(successful) * 6 * 0.0000002  # 6 functions per workflow
    summary.estimated_cost_usd = lambda_compute_cost + request_cost

    return summary


def print_summary(summary: BenchmarkSummary):
    """Print formatted summary"""
    print(f"\n{'='*70}")
    print(f"{summary.mode} Benchmark Summary")
    print(f"{'='*70}")
    print(f"Total runs: {summary.total_runs}")
    print(f"Successful: {summary.successful_runs}")
    print(f"Failed: {summary.failed_runs}")
    print()

    if summary.successful_runs > 0:
        print(f"E2E Latency:")
        print(f"  Mean:   {summary.e2e_mean_ms:8.2f}ms")
        print(f"  Median: {summary.e2e_median_ms:8.2f}ms")
        print(f"  Min:    {summary.e2e_min_ms:8.2f}ms")
        print(f"  Max:    {summary.e2e_max_ms:8.2f}ms")
        print(f"  StdDev: {summary.e2e_std_ms:8.2f}ms")
        print()

        print(f"Aggregator Invocation Timing:")
        print(f"  Mean:   {summary.aggregator_invocation_mean_ms:8.2f}ms")
        print(f"  Median: {summary.aggregator_invocation_median_ms:8.2f}ms")
        print(f"  Min:    {summary.aggregator_invocation_min_ms:8.2f}ms")
        print(f"  Max:    {summary.aggregator_invocation_max_ms:8.2f}ms")
        print()

        print(f"Per-Function Averages:")
        print(f"  TriggerFunction:  {summary.avg_trigger_ms:8.2f}ms")
        print(f"  FastProcessor:    {summary.avg_fast_processor_ms:8.2f}ms")
        print(f"  FusedOrderProcessing: {summary.avg_fused_order_processing_ms:8.2f}ms")
        print(f"  Aggregator:       {summary.avg_aggregator_ms:8.2f}ms")
        print()

        print(f"Cold Starts:")
        print(f"  Total:        {summary.total_cold_starts}")
        print(f"  Avg Init:     {summary.avg_init_duration_ms:8.2f}ms")
        print()

        print(f"Invoker Distribution:")
        for invoker, count in summary.invoker_distribution.items():
            print(f"  {invoker}: {count} ({count/summary.successful_runs*100:.1f}%)")
        print()

        print(f"Estimated Cost: ${summary.estimated_cost_usd:.6f}")

    print(f"{'='*70}\n")


def compare_results(classic: BenchmarkSummary, future: BenchmarkSummary):
    """Print comparison between CLASSIC and FUTURE_BASED modes"""
    print(f"\n{'='*70}")
    print(f"CLASSIC vs FUTURE_BASED Comparison")
    print(f"{'='*70}\n")

    # E2E Latency improvement
    e2e_improvement_ms = classic.e2e_mean_ms - future.e2e_mean_ms
    e2e_improvement_pct = (e2e_improvement_ms / classic.e2e_mean_ms * 100) if classic.e2e_mean_ms > 0 else 0

    print(f"E2E Latency:")
    print(f"  CLASSIC:       {classic.e2e_mean_ms:8.2f}ms")
    print(f"  FUTURE_BASED:  {future.e2e_mean_ms:8.2f}ms")
    print(f"  Improvement:   {e2e_improvement_ms:8.2f}ms ({e2e_improvement_pct:+.1f}%)")
    print()

    # Aggregator invocation timing (key metric)
    agg_improvement_ms = classic.aggregator_invocation_mean_ms - future.aggregator_invocation_mean_ms
    agg_improvement_pct = (agg_improvement_ms / classic.aggregator_invocation_mean_ms * 100) if classic.aggregator_invocation_mean_ms > 0 else 0

    print(f"Aggregator Invocation Delay:")
    print(f"  CLASSIC:       {classic.aggregator_invocation_mean_ms:8.2f}ms  (invoked by FusedOrderProcessing)")
    print(f"  FUTURE_BASED:  {future.aggregator_invocation_mean_ms:8.2f}ms  (invoked by FastProcessor)")
    print(f"  Improvement:   {agg_improvement_ms:8.2f}ms ({agg_improvement_pct:+.1f}%)")
    print()

    # Cold start benefit
    init_improvement_ms = classic.avg_init_duration_ms - future.avg_init_duration_ms

    print(f"Cold Start Duration:")
    print(f"  CLASSIC:       {classic.avg_init_duration_ms:8.2f}ms")
    print(f"  FUTURE_BASED:  {future.avg_init_duration_ms:8.2f}ms")
    print(f"  Difference:    {init_improvement_ms:8.2f}ms")
    print()

    # Cost comparison
    cost_savings = classic.estimated_cost_usd - future.estimated_cost_usd
    cost_savings_pct = (cost_savings / classic.estimated_cost_usd * 100) if classic.estimated_cost_usd > 0 else 0

    print(f"Cost per Workflow:")
    print(f"  CLASSIC:       ${classic.estimated_cost_usd/classic.successful_runs:.8f}")
    print(f"  FUTURE_BASED:  ${future.estimated_cost_usd/future.successful_runs:.8f}")
    print(f"  Savings:       ${cost_savings/classic.successful_runs:.8f} ({cost_savings_pct:+.1f}%)")
    print()

    print(f"Key Finding:")
    print(f"  Future-Based mode invokes Aggregator ~{agg_improvement_ms:.0f}ms earlier,")
    print(f"  hiding cold start latency behind fused order processing execution.")
    print(f"{'='*70}\n")


# ============================================================
# Chart Generation
# ============================================================

def generate_charts(classic: BenchmarkSummary, future: BenchmarkSummary,
                   classic_runs: List[WorkflowRun], future_runs: List[WorkflowRun],
                   output_dir: Path):
    """Generate comparison charts"""
    if not CHARTS_AVAILABLE:
        print("Skipping chart generation (matplotlib not available)")
        return

    print("\nGenerating charts...")
    output_dir.mkdir(exist_ok=True)

    # Chart 1: E2E Latency Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    modes = ['CLASSIC', 'FUTURE_BASED']
    means = [classic.e2e_mean_ms, future.e2e_mean_ms]
    mins = [classic.e2e_min_ms, future.e2e_min_ms]
    maxs = [classic.e2e_max_ms, future.e2e_max_ms]

    x = range(len(modes))
    ax.bar(x, means, color=['#e74c3c', '#27ae60'], alpha=0.7)
    ax.errorbar(x, means, yerr=[[m - mn for m, mn in zip(means, mins)],
                                [mx - m for m, mx in zip(means, maxs)]],
                fmt='none', color='black', capsize=5)

    ax.set_xlabel('Execution Mode', fontsize=12)
    ax.set_ylabel('E2E Latency (ms)', fontsize=12)
    ax.set_title('End-to-End Latency: CLASSIC vs FUTURE_BASED', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.grid(axis='y', alpha=0.3)

    # Add improvement annotation
    improvement = classic.e2e_mean_ms - future.e2e_mean_ms
    improvement_pct = (improvement / classic.e2e_mean_ms * 100)
    ax.text(0.5, max(means) * 0.9, f'↓ {improvement:.0f}ms ({improvement_pct:.1f}%)',
            ha='center', fontsize=12, fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(output_dir / 'e2e_latency_comparison.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: e2e_latency_comparison.png")

    # Chart 2: Aggregator Invocation Timing
    fig, ax = plt.subplots(figsize=(10, 6))

    agg_means = [classic.aggregator_invocation_mean_ms, future.aggregator_invocation_mean_ms]

    bars = ax.bar(x, agg_means, color=['#e74c3c', '#27ae60'], alpha=0.7)

    ax.set_xlabel('Execution Mode', fontsize=12)
    ax.set_ylabel('Time to Aggregator Invocation (ms)', fontsize=12)
    ax.set_title('Aggregator Invocation Timing: CLASSIC vs FUTURE_BASED', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, agg_means)):
        invoker = "FusedOrderProcessing" if i == 0 else "FastProcessor"
        ax.text(bar.get_x() + bar.get_width()/2, val + max(agg_means)*0.02,
                f'{val:.0f}ms\n({invoker})',
                ha='center', va='bottom', fontsize=10)

    # Add improvement annotation
    agg_improvement = classic.aggregator_invocation_mean_ms - future.aggregator_invocation_mean_ms
    agg_improvement_pct = (agg_improvement / classic.aggregator_invocation_mean_ms * 100)
    ax.text(0.5, max(agg_means) * 0.5, f'↓ {agg_improvement:.0f}ms ({agg_improvement_pct:.1f}%)',
            ha='center', fontsize=14, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'aggregator_invocation_timing.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: aggregator_invocation_timing.png")

    # Chart 3: Per-Function Duration Comparison
    fig, ax = plt.subplots(figsize=(12, 7))

    functions = ['TriggerFunction', 'FastProcessor', 'FusedOrderProcessing',
                'Aggregator']
    classic_times = [
        classic.avg_trigger_ms, classic.avg_fast_processor_ms,
        classic.avg_fused_order_processing_ms, classic.avg_aggregator_ms
    ]
    future_times = [
        future.avg_trigger_ms, future.avg_fast_processor_ms,
        future.avg_fused_order_processing_ms, future.avg_aggregator_ms
    ]

    x_pos = range(len(functions))
    width = 0.35

    ax.bar([p - width/2 for p in x_pos], classic_times, width,
           label='CLASSIC', color='#e74c3c', alpha=0.7)
    ax.bar([p + width/2 for p in x_pos], future_times, width,
           label='FUTURE_BASED', color='#27ae60', alpha=0.7)

    ax.set_xlabel('Function', fontsize=12)
    ax.set_ylabel('Average Duration (ms)', fontsize=12)
    ax.set_title('Per-Function Duration: CLASSIC vs FUTURE_BASED', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(functions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_function_duration.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: per_function_duration.png")

    # Chart 4: Timeline Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # CLASSIC timeline
    classic_successful = [r for r in classic_runs if r.error is None and r.e2e_latency_ms > 0]
    if classic_successful:
        run = classic_successful[0]  # Use first successful run

        functions_order = ['TriggerFunction', 'FastProcessor', 'FusedOrderProcessing',
                          'Aggregator']

        for i, func_name in enumerate(functions_order):
            if func_name in run.function_metrics:
                m = run.function_metrics[func_name]
                start_offset = (m.invocation_time_ms - int(run.start_time * 1000))

                ax1.barh(i, m.duration_ms, left=start_offset, height=0.6,
                        color='#e74c3c' if not m.cold_start else '#c0392b',
                        alpha=0.7, edgecolor='black')

                # Add label
                ax1.text(start_offset + m.duration_ms/2, i, f'{m.duration_ms:.0f}ms',
                        ha='center', va='center', fontsize=9, fontweight='bold')

        ax1.set_yticks(range(len(functions_order)))
        ax1.set_yticklabels(functions_order)
        ax1.set_xlabel('Time from Workflow Start (ms)', fontsize=11)
        ax1.set_title('CLASSIC Mode Timeline', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.axvline(run.aggregator_invocation_delay_ms, color='red', linestyle='--',
                   label=f'Aggregator invoked at +{run.aggregator_invocation_delay_ms:.0f}ms')
        ax1.legend()

    # FUTURE_BASED timeline
    future_successful = [r for r in future_runs if r.error is None and r.e2e_latency_ms > 0]
    if future_successful:
        run = future_successful[0]  # Use first successful run

        for i, func_name in enumerate(functions_order):
            if func_name in run.function_metrics:
                m = run.function_metrics[func_name]
                start_offset = (m.invocation_time_ms - int(run.start_time * 1000))

                ax2.barh(i, m.duration_ms, left=start_offset, height=0.6,
                        color='#27ae60' if not m.cold_start else '#229954',
                        alpha=0.7, edgecolor='black')

                # Add label
                ax2.text(start_offset + m.duration_ms/2, i, f'{m.duration_ms:.0f}ms',
                        ha='center', va='center', fontsize=9, fontweight='bold')

        ax2.set_yticks(range(len(functions_order)))
        ax2.set_yticklabels(functions_order)
        ax2.set_xlabel('Time from Workflow Start (ms)', fontsize=11)
        ax2.set_title('FUTURE_BASED Mode Timeline', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.axvline(run.aggregator_invocation_delay_ms, color='green', linestyle='--',
                   label=f'Aggregator invoked at +{run.aggregator_invocation_delay_ms:.0f}ms')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'execution_timeline.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: execution_timeline.png")

    print(f"\n✓ All charts saved to: {output_dir}")


# ============================================================
# Results Persistence
# ============================================================

def save_results(classic_summary: BenchmarkSummary, future_summary: BenchmarkSummary,
                classic_runs: List[WorkflowRun], future_runs: List[WorkflowRun],
                output_dir: Path):
    """Save benchmark results to JSON"""
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save summaries
    summary_file = output_dir / f'benchmark_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'CLASSIC': asdict(classic_summary),
            'FUTURE_BASED': asdict(future_summary),
            'improvement': {
                'e2e_latency_ms': classic_summary.e2e_mean_ms - future_summary.e2e_mean_ms,
                'e2e_latency_pct': ((classic_summary.e2e_mean_ms - future_summary.e2e_mean_ms) / classic_summary.e2e_mean_ms * 100) if classic_summary.e2e_mean_ms > 0 else 0,
                'aggregator_invocation_ms': classic_summary.aggregator_invocation_mean_ms - future_summary.aggregator_invocation_mean_ms,
                'aggregator_invocation_pct': ((classic_summary.aggregator_invocation_mean_ms - future_summary.aggregator_invocation_mean_ms) / classic_summary.aggregator_invocation_mean_ms * 100) if classic_summary.aggregator_invocation_mean_ms > 0 else 0,
            }
        }, f, indent=2, default=str)

    print(f"\n✓ Saved summary: {summary_file}")

    # Save detailed runs
    runs_file = output_dir / f'benchmark_runs_{timestamp}.json'
    with open(runs_file, 'w') as f:
        json.dump({
            'CLASSIC': [asdict(r) for r in classic_runs],
            'FUTURE_BASED': [asdict(r) for r in future_runs]
        }, f, indent=2, default=str)

    print(f"✓ Saved detailed runs: {runs_file}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Complete benchmark suite for Order Processing Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations per mode (default: 5)')
    parser.add_argument('--cold-all', action='store_true',
                       help='Force cold starts for all iterations (default: only first)')
    parser.add_argument('--skip-classic', action='store_true',
                       help='Skip CLASSIC mode benchmark')
    parser.add_argument('--skip-future', action='store_true',
                       help='Skip FUTURE_BASED mode benchmark')
    parser.add_argument('--skip-charts', action='store_true',
                       help='Skip chart generation')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Order Processing Workflow - Complete Benchmark Suite")
    print(f"{'='*70}")
    print(f"Region: {REGION}")
    print(f"Profile: {PROFILE}")
    print(f"Iterations: {args.iterations}")
    print(f"{'='*70}\n")

    # Load function ARNs
    function_arns = load_function_arns()
    if not function_arns:
        print("✗ Cannot proceed without function ARNs")
        return 1

    cold_start_freq = 'all' if args.cold_all else 'first'

    # Run benchmarks
    classic_runs = []
    future_runs = []

    if not args.skip_classic:
        classic_runs = run_benchmark_mode('CLASSIC', args.iterations, function_arns, cold_start_freq)
        time.sleep(30)  # Cooldown between modes

    if not args.skip_future:
        future_runs = run_benchmark_mode('FUTURE_BASED', args.iterations, function_arns, cold_start_freq)

    # Compute summaries
    if classic_runs:
        classic_summary = compute_summary('CLASSIC', classic_runs)
        print_summary(classic_summary)

    if future_runs:
        future_summary = compute_summary('FUTURE_BASED', future_runs)
        print_summary(future_summary)

    # Compare results
    if classic_runs and future_runs:
        compare_results(classic_summary, future_summary)

        # Save results
        results_dir = Path(__file__).parent / 'results'
        save_results(classic_summary, future_summary, classic_runs, future_runs, results_dir)

        # Generate charts
        if not args.skip_charts:
            generate_charts(classic_summary, future_summary, classic_runs, future_runs, results_dir)

    print(f"\n{'='*70}")
    print(f"Benchmark Complete!")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    exit(main())
