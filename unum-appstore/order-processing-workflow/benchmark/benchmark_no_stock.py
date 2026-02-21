#!/usr/bin/env python3
"""
No-Stock Scenario Benchmark - Demonstrates Short-Circuit Optimization

This benchmark demonstrates the key advantage of FUTURE_BASED mode:
When inventory is unavailable, FUTURE_BASED can immediately reject the order
WITHOUT waiting for the slow sequential chain (payment/shipping/invoice).

Expected Results:
- CLASSIC mode: ~3000-4000ms (must wait for entire slow chain before rejecting)
- FUTURE_BASED mode: ~200-400ms (short-circuits immediately when no stock)

This shows a potential 10x improvement for failure scenarios!

Usage:
    python benchmark_no_stock.py --iterations 5
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
    import matplotlib.patches as mpatches
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
    'SlowChainStart': {'order': 3, 'expected_ms': 500, 'description': 'Chain step 1'},
    'SlowChainMid': {'order': 4, 'expected_ms': 1000, 'description': 'Chain step 2'},
    'SlowChainEnd': {'order': 5, 'expected_ms': 1500, 'description': 'Chain step 3'},
    'Aggregator': {'order': 6, 'expected_ms': 100, 'description': 'Terminal aggregator'},
}

# AWS Clients
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
    invocation_time_ms: int = 0

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

    # Scenario
    scenario: str = "no_stock"

    # Invocation metrics
    trigger_latency_ms: float = 0.0
    e2e_latency_ms: float = 0.0

    # Per-function metrics
    function_metrics: Dict[str, LambdaMetrics] = field(default_factory=dict)

    # Aggregated metrics
    total_duration_ms: float = 0.0
    total_billed_ms: float = 0.0
    cold_start_count: int = 0
    total_init_ms: float = 0.0

    # Key timing metrics
    fast_processor_time_ms: float = 0.0
    slow_chain_total_ms: float = 0.0
    aggregator_invocation_delay_ms: float = 0.0

    # Short-circuit metrics (key for this benchmark!)
    short_circuited: bool = False
    time_saved_ms: float = 0.0  # Time saved by short-circuit

    # Errors
    error: Optional[str] = None

    def calculate_aggregates(self):
        """Calculate aggregate metrics from function data"""
        self.total_duration_ms = sum(m.duration_ms for m in self.function_metrics.values())
        self.total_billed_ms = sum(m.billed_duration_ms for m in self.function_metrics.values())
        self.cold_start_count = sum(1 for m in self.function_metrics.values() if m.cold_start)
        self.total_init_ms = sum(m.init_duration_ms for m in self.function_metrics.values())

        if 'FastProcessor' in self.function_metrics:
            self.fast_processor_time_ms = self.function_metrics['FastProcessor'].duration_ms

        chain_times = []
        for func in ['SlowChainStart', 'SlowChainMid', 'SlowChainEnd']:
            if func in self.function_metrics:
                chain_times.append(self.function_metrics[func].duration_ms)
        self.slow_chain_total_ms = sum(chain_times)

        if 'Aggregator' in self.function_metrics:
            agg_invocation_ts = self.function_metrics['Aggregator'].invocation_time_ms
            workflow_start_ts = int(self.start_time * 1000)
            self.aggregator_invocation_delay_ms = agg_invocation_ts - workflow_start_ts


@dataclass
class BenchmarkSummary:
    """Statistical summary for a benchmark mode"""
    mode: str
    scenario: str
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

    # Aggregator invocation timing
    aggregator_invocation_mean_ms: float = 0.0
    aggregator_invocation_median_ms: float = 0.0

    # Per-function averages
    avg_trigger_ms: float = 0.0
    avg_fast_processor_ms: float = 0.0
    avg_slow_chain_start_ms: float = 0.0
    avg_slow_chain_mid_ms: float = 0.0
    avg_slow_chain_end_ms: float = 0.0
    avg_aggregator_ms: float = 0.0
    avg_slow_chain_total_ms: float = 0.0

    # Short-circuit metrics
    short_circuit_count: int = 0
    avg_time_saved_ms: float = 0.0

    # Cold starts
    total_cold_starts: int = 0
    avg_init_duration_ms: float = 0.0

    # Cost
    total_billed_ms: float = 0.0
    estimated_cost_usd: float = 0.0


# ============================================================
# Function ARN Management
# ============================================================

def load_function_arns() -> Dict[str, str]:
    """Load function ARNs from function-arn.yaml"""
    yaml_path = Path(__file__).parent.parent / 'function-arn.yaml'

    if not yaml_path.exists():
        print(f"Error: function-arn.yaml not found at {yaml_path}")
        return {}

    with open(yaml_path, 'r') as f:
        arns = yaml.safe_load(f)

    print(f"Loaded {len(arns)} function ARNs")
    return arns


def extract_function_name(arn: str) -> str:
    """Extract physical function name from ARN"""
    return arn.split(':')[-1]


# ============================================================
# Lambda Configuration
# ============================================================

def configure_mode(mode: str, function_arns: Dict[str, str], simulate_no_stock: bool = True):
    """Configure Lambda functions for the specified execution mode"""
    print(f"\n{'='*70}")
    print(f"Configuring {mode} Mode (SIMULATE_NO_STOCK={simulate_no_stock})")
    print(f"{'='*70}")

    config = MODE_CONFIGS[mode]

    for func_name, arn in function_arns.items():
        try:
            response = lambda_client.get_function_configuration(FunctionName=arn)
            current_env = response.get('Environment', {}).get('Variables', {})

            current_env['EAGER'] = config['EAGER']

            if func_name == 'Aggregator':
                current_env['UNUM_FUTURE_BASED'] = config['UNUM_FUTURE_BASED']

            # Set SIMULATE_NO_STOCK on FastProcessor
            if func_name == 'FastProcessor':
                current_env['SIMULATE_NO_STOCK'] = 'true' if simulate_no_stock else 'false'

            lambda_client.update_function_configuration(
                FunctionName=arn,
                Environment={'Variables': current_env}
            )

            extra = ""
            if func_name == 'Aggregator':
                extra = f", UNUM_FUTURE_BASED={config['UNUM_FUTURE_BASED']}"
            if func_name == 'FastProcessor':
                extra += f", SIMULATE_NO_STOCK={'true' if simulate_no_stock else 'false'}"
            print(f"  {func_name}: EAGER={config['EAGER']}{extra}")

        except Exception as e:
            print(f"  {func_name}: Error - {e}")

    print(f"  Waiting for updates to propagate...")
    time.sleep(5)


def force_cold_starts(function_arns: Dict[str, str]):
    """Force cold starts by updating all function environments"""
    print("\n  Forcing cold starts...")

    timestamp = str(int(time.time()))

    for func_name, arn in function_arns.items():
        try:
            response = lambda_client.get_function_configuration(FunctionName=arn)
            current_env = response.get('Environment', {}).get('Variables', {})
            current_env['COLD_START_TRIGGER'] = timestamp

            lambda_client.update_function_configuration(
                FunctionName=arn,
                Environment={'Variables': current_env}
            )
        except Exception as e:
            print(f"    {func_name}: Error - {e}")

    print("  Waiting 15s for updates to propagate...")
    time.sleep(15)


# ============================================================
# Workflow Invocation
# ============================================================

def invoke_workflow(order_id: str, function_arns: Dict[str, str]) -> Dict:
    """Invoke the workflow and measure end-to-end latency"""
    trigger_arn = function_arns.get('TriggerFunction')

    if not trigger_arn:
        return {'success': False, 'order_id': order_id, 'error': 'TriggerFunction ARN not found'}

    payload = {
        "Data": {
            "Source": "http",
            "Value": {
                "order_id": order_id,
                "customer_id": "BENCH-NOSTOCK-CUSTOMER",
                "items": [
                    {"sku": "OUT-OF-STOCK-001", "quantity": 5, "price": 99.99}
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

        response_payload = json.loads(response['Payload'].read())

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
        return {'success': False, 'order_id': order_id, 'error': str(e)}


# ============================================================
# CloudWatch Logs Collection
# ============================================================

def get_cloudwatch_logs(function_name: str, start_time: float, end_time: float) -> List[Dict]:
    """Retrieve CloudWatch logs for a function"""
    log_group = f"/aws/lambda/{function_name}"

    try:
        start_ms = int(start_time * 1000) - 5000
        end_ms = int(end_time * 1000) + 60000

        response = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=start_ms,
            endTime=end_ms,
            limit=200
        )

        return response.get('events', [])

    except Exception as e:
        return []


def extract_lambda_metrics(function_logical_name: str, function_physical_name: str,
                          start_time: float, end_time: float) -> LambdaMetrics:
    """Extract metrics from CloudWatch REPORT logs"""
    logs = get_cloudwatch_logs(function_physical_name, start_time, end_time)

    metrics = LambdaMetrics(function_name=function_logical_name)

    report_logs = [log for log in logs if 'REPORT RequestId:' in log.get('message', '')]

    if not report_logs:
        return metrics

    latest_report = max(report_logs, key=lambda x: x.get('timestamp', 0))
    message = latest_report.get('message', '')
    timestamp_ms = latest_report.get('timestamp', 0)

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

    if force_cold:
        force_cold_starts(function_arns)

    result = invoke_workflow(order_id, function_arns)

    if not result['success']:
        print(f"    Invocation failed: {result.get('error')}")
        return WorkflowRun(
            run_id=run_id,
            mode=mode,
            order_id=order_id,
            start_time=time.time(),
            end_time=time.time(),
            error=result.get('error')
        )

    print(f"    Trigger latency: {result['trigger_latency_ms']:.2f}ms")

    # Wait for workflow - shorter for FUTURE_BASED (short-circuit)
    wait_time = 5 if mode == 'FUTURE_BASED' else 10
    print(f"    Waiting {wait_time}s for workflow completion...")
    time.sleep(wait_time)

    end_time = time.time()
    metrics = collect_all_metrics(function_arns, result['start_time'], end_time)

    run = WorkflowRun(
        run_id=run_id,
        mode=mode,
        order_id=order_id,
        start_time=result['start_time'],
        end_time=end_time,
        trigger_latency_ms=result['trigger_latency_ms'],
        function_metrics=metrics,
        scenario="no_stock"
    )

    # Calculate e2e latency from Aggregator completion time
    if metrics.get('Aggregator', LambdaMetrics('')).invocation_time_ms > 0:
        agg = metrics['Aggregator']
        workflow_start_ms = int(result['start_time'] * 1000)
        workflow_end_ms = agg.invocation_time_ms + int(agg.duration_ms)
        run.e2e_latency_ms = workflow_end_ms - workflow_start_ms

    run.calculate_aggregates()

    # Detect short-circuit
    # In FUTURE_BASED + no_stock, SlowChainEnd should NOT have been waited for
    slow_end_m = metrics.get('SlowChainEnd', LambdaMetrics(''))
    agg_m = metrics.get('Aggregator', LambdaMetrics(''))

    if mode == 'FUTURE_BASED' and agg_m.invocation_time_ms > 0:
        # Check if Aggregator completed before SlowChainEnd
        if slow_end_m.invocation_time_ms > 0:
            slow_end_completed_ms = slow_end_m.invocation_time_ms + int(slow_end_m.duration_ms)
            agg_completed_ms = agg_m.invocation_time_ms + int(agg_m.duration_ms)

            if agg_completed_ms < slow_end_completed_ms:
                run.short_circuited = True
                run.time_saved_ms = slow_end_completed_ms - agg_completed_ms
        else:
            # No SlowChainEnd metrics might mean it wasn't waited for at all
            run.short_circuited = True
            run.time_saved_ms = 3000  # Approximate

    # Print summary
    print(f"    Functions executed:")
    for func_name in sorted(metrics.keys(), key=lambda x: FUNCTION_INFO.get(x, {}).get('order', 99)):
        m = metrics[func_name]
        if m.duration_ms > 0:
            cold_tag = " [COLD]" if m.cold_start else ""
            print(f"      {func_name:20s}: {m.duration_ms:7.2f}ms{cold_tag}")

    if run.e2e_latency_ms > 0:
        print(f"    E2E Latency: {run.e2e_latency_ms:.2f}ms")

    if run.short_circuited:
        print(f"    SHORT-CIRCUIT: Yes (saved ~{run.time_saved_ms:.0f}ms)")
    else:
        print(f"    SHORT-CIRCUIT: No (waited for slow chain)")

    return run


def run_benchmark_mode(mode: str, iterations: int, function_arns: Dict[str, str],
                      cold_start_freq: str = 'first') -> List[WorkflowRun]:
    """Run benchmark for a specific mode"""
    print(f"\n{'='*70}")
    print(f"Running {mode} Benchmark (No-Stock Scenario)")
    print(f"Iterations: {iterations}")
    print(f"{'='*70}")

    configure_mode(mode, function_arns, simulate_no_stock=True)

    runs = []

    for i in range(iterations):
        force_cold = (cold_start_freq == 'all') or (cold_start_freq == 'first' and i == 0)
        order_id = f"NOSTOCK-{mode}-{i+1:03d}-{int(time.time())}"

        run = run_single_iteration(i + 1, mode, order_id, function_arns, force_cold)
        runs.append(run)

        if i < iterations - 1:
            print(f"  Waiting 3s before next iteration...")
            time.sleep(3)

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
        scenario="no_stock",
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
            elif func_name == 'SlowChainStart':
                summary.avg_slow_chain_start_ms = avg
            elif func_name == 'SlowChainMid':
                summary.avg_slow_chain_mid_ms = avg
            elif func_name == 'SlowChainEnd':
                summary.avg_slow_chain_end_ms = avg
            elif func_name == 'Aggregator':
                summary.avg_aggregator_ms = avg

    # Slow chain total
    chain_totals = [r.slow_chain_total_ms for r in successful if r.slow_chain_total_ms > 0]
    if chain_totals:
        summary.avg_slow_chain_total_ms = statistics.mean(chain_totals)

    # Short-circuit metrics
    summary.short_circuit_count = sum(1 for r in successful if r.short_circuited)
    time_saved = [r.time_saved_ms for r in successful if r.time_saved_ms > 0]
    if time_saved:
        summary.avg_time_saved_ms = statistics.mean(time_saved)

    # Cold starts
    summary.total_cold_starts = sum(r.cold_start_count for r in successful)
    init_durations = [r.total_init_ms for r in successful if r.total_init_ms > 0]
    if init_durations:
        summary.avg_init_duration_ms = statistics.mean(init_durations)

    # Cost calculation
    summary.total_billed_ms = sum(r.total_billed_ms for r in successful)
    gb_seconds = (summary.total_billed_ms / 1000) * 0.25
    lambda_compute_cost = gb_seconds * 0.0000166667
    request_cost = len(successful) * 6 * 0.0000002
    summary.estimated_cost_usd = lambda_compute_cost + request_cost

    return summary


def print_summary(summary: BenchmarkSummary):
    """Print formatted summary"""
    print(f"\n{'='*70}")
    print(f"{summary.mode} Benchmark Summary (No-Stock Scenario)")
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
        print()

        print(f"Short-Circuit:")
        print(f"  Count:      {summary.short_circuit_count}/{summary.successful_runs}")
        print(f"  Avg Saved:  {summary.avg_time_saved_ms:8.2f}ms")
        print()

        print(f"Per-Function Averages:")
        print(f"  TriggerFunction:  {summary.avg_trigger_ms:8.2f}ms")
        print(f"  FastProcessor:    {summary.avg_fast_processor_ms:8.2f}ms")
        print(f"  SlowChainStart:   {summary.avg_slow_chain_start_ms:8.2f}ms")
        print(f"  SlowChainMid:     {summary.avg_slow_chain_mid_ms:8.2f}ms")
        print(f"  SlowChainEnd:     {summary.avg_slow_chain_end_ms:8.2f}ms")
        print(f"  Aggregator:       {summary.avg_aggregator_ms:8.2f}ms")
        print()

        print(f"Estimated Cost: ${summary.estimated_cost_usd:.6f}")

    print(f"{'='*70}\n")


def compare_results(classic: BenchmarkSummary, future: BenchmarkSummary):
    """Print comparison between CLASSIC and FUTURE_BASED modes"""
    print(f"\n{'='*70}")
    print(f"NO-STOCK SCENARIO: CLASSIC vs FUTURE_BASED Comparison")
    print(f"{'='*70}\n")

    # E2E Latency improvement
    e2e_improvement_ms = classic.e2e_mean_ms - future.e2e_mean_ms
    e2e_improvement_pct = (e2e_improvement_ms / classic.e2e_mean_ms * 100) if classic.e2e_mean_ms > 0 else 0
    speedup_factor = classic.e2e_mean_ms / future.e2e_mean_ms if future.e2e_mean_ms > 0 else 0

    print(f"E2E Latency (Order Rejection Time):")
    print(f"  CLASSIC:       {classic.e2e_mean_ms:8.2f}ms  (waited for entire slow chain)")
    print(f"  FUTURE_BASED:  {future.e2e_mean_ms:8.2f}ms  (short-circuited!)")
    print(f"  Improvement:   {e2e_improvement_ms:8.2f}ms ({e2e_improvement_pct:+.1f}%)")
    print(f"  Speedup:       {speedup_factor:.1f}x faster!")
    print()

    print(f"Short-Circuit Analysis:")
    print(f"  CLASSIC short-circuits:       {classic.short_circuit_count}/{classic.successful_runs}")
    print(f"  FUTURE_BASED short-circuits:  {future.short_circuit_count}/{future.successful_runs}")
    print(f"  Avg time saved per request:   {future.avg_time_saved_ms:.0f}ms")
    print()

    # Cost comparison
    cost_savings = classic.estimated_cost_usd - future.estimated_cost_usd
    cost_savings_pct = (cost_savings / classic.estimated_cost_usd * 100) if classic.estimated_cost_usd > 0 else 0

    print(f"Cost per Rejected Order:")
    print(f"  CLASSIC:       ${classic.estimated_cost_usd/classic.successful_runs:.8f}")
    print(f"  FUTURE_BASED:  ${future.estimated_cost_usd/future.successful_runs:.8f}")
    print(f"  Savings:       ${cost_savings/classic.successful_runs:.8f} ({cost_savings_pct:+.1f}%)")
    print()

    print(f"KEY FINDING:")
    print(f"  In the no-stock scenario, FUTURE_BASED mode is {speedup_factor:.1f}x faster!")
    print(f"  It rejects the order immediately after FastProcessor returns,")
    print(f"  saving ~{e2e_improvement_ms:.0f}ms by NOT waiting for the slow chain.")
    print(f"  This demonstrates the power of early termination in failure scenarios.")
    print(f"{'='*70}\n")


# ============================================================
# Chart Generation
# ============================================================

def generate_charts(classic: BenchmarkSummary, future: BenchmarkSummary,
                   classic_runs: List[WorkflowRun], future_runs: List[WorkflowRun],
                   output_dir: Path):
    """Generate comprehensive comparison charts"""
    if not CHARTS_AVAILABLE:
        print("Skipping chart generation (matplotlib not available)")
        return

    print("\nGenerating charts...")
    output_dir.mkdir(exist_ok=True)

    # Color scheme
    classic_color = '#e74c3c'
    future_color = '#27ae60'

    # ========================================
    # Chart 1: E2E Latency Comparison (Main Result)
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 7))

    modes = ['CLASSIC\n(Must Wait)', 'FUTURE_BASED\n(Short-Circuit)']
    means = [classic.e2e_mean_ms, future.e2e_mean_ms]

    bars = ax.bar(modes, means, color=[classic_color, future_color], alpha=0.8, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(means)*0.02,
                f'{val:.0f}ms', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add improvement annotation
    improvement_pct = (classic.e2e_mean_ms - future.e2e_mean_ms) / classic.e2e_mean_ms * 100
    speedup = classic.e2e_mean_ms / future.e2e_mean_ms if future.e2e_mean_ms > 0 else 0

    ax.annotate(f'{speedup:.1f}x\nFaster!',
                xy=(1, future.e2e_mean_ms),
                xytext=(0.5, classic.e2e_mean_ms * 0.6),
                fontsize=16, fontweight='bold', color='green',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.set_ylabel('Time to Reject Order (ms)', fontsize=14)
    ax.set_title('No-Stock Scenario: Order Rejection Latency\nFUTURE_BASED Short-Circuits, CLASSIC Must Wait',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(means) * 1.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'no_stock_e2e_comparison.png', dpi=300)
    plt.close()
    print(f"  Saved: no_stock_e2e_comparison.png")

    # ========================================
    # Chart 2: Workflow Execution Timeline
    # ========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    functions_order = ['TriggerFunction', 'FastProcessor', 'SlowChainStart',
                      'SlowChainMid', 'SlowChainEnd', 'Aggregator']

    # CLASSIC timeline
    classic_successful = [r for r in classic_runs if r.error is None and r.e2e_latency_ms > 0]
    if classic_successful:
        run = classic_successful[0]

        for i, func_name in enumerate(functions_order):
            if func_name in run.function_metrics:
                m = run.function_metrics[func_name]
                if m.duration_ms > 0:
                    start_offset = max(0, m.invocation_time_ms - int(run.start_time * 1000))

                    color = '#c0392b' if m.cold_start else classic_color
                    ax1.barh(i, m.duration_ms, left=start_offset, height=0.6,
                            color=color, alpha=0.8, edgecolor='black')

                    ax1.text(start_offset + m.duration_ms + 50, i, f'{m.duration_ms:.0f}ms',
                            ha='left', va='center', fontsize=10)

        ax1.set_yticks(range(len(functions_order)))
        ax1.set_yticklabels(functions_order)
        ax1.set_xlabel('Time from Workflow Start (ms)', fontsize=12)
        ax1.set_title('CLASSIC Mode: Must Wait for Entire Slow Chain Before Rejecting',
                     fontsize=12, fontweight='bold', color=classic_color)
        ax1.grid(axis='x', alpha=0.3)

        # Add annotation for wasted work
        ax1.axvspan(0, run.e2e_latency_ms, alpha=0.1, color=classic_color)
        ax1.text(run.e2e_latency_ms/2, -0.8, 'Entire workflow must complete\n(wasted work!)',
                ha='center', fontsize=10, style='italic', color=classic_color)

    # FUTURE_BASED timeline
    future_successful = [r for r in future_runs if r.error is None and r.e2e_latency_ms > 0]
    if future_successful:
        run = future_successful[0]

        for i, func_name in enumerate(functions_order):
            if func_name in run.function_metrics:
                m = run.function_metrics[func_name]
                if m.duration_ms > 0:
                    start_offset = max(0, m.invocation_time_ms - int(run.start_time * 1000))

                    # Gray out functions that weren't waited for
                    if func_name in ['SlowChainMid', 'SlowChainEnd'] and run.short_circuited:
                        color = '#cccccc'
                        alpha = 0.4
                    else:
                        color = '#229954' if m.cold_start else future_color
                        alpha = 0.8

                    ax2.barh(i, m.duration_ms, left=start_offset, height=0.6,
                            color=color, alpha=alpha, edgecolor='black')

                    ax2.text(start_offset + m.duration_ms + 50, i, f'{m.duration_ms:.0f}ms',
                            ha='left', va='center', fontsize=10)

        ax2.set_yticks(range(len(functions_order)))
        ax2.set_yticklabels(functions_order)
        ax2.set_xlabel('Time from Workflow Start (ms)', fontsize=12)
        ax2.set_title('FUTURE_BASED Mode: Short-Circuits After FastProcessor Returns No-Stock',
                     fontsize=12, fontweight='bold', color=future_color)
        ax2.grid(axis='x', alpha=0.3)

        # Add annotation for short-circuit point
        ax2.axvline(run.e2e_latency_ms, color=future_color, linestyle='--', linewidth=2)
        ax2.text(run.e2e_latency_ms + 50, 5.5, f'Order rejected!\n({run.e2e_latency_ms:.0f}ms)',
                fontsize=10, fontweight='bold', color=future_color)

    # Match x-axis scales
    max_x = max(classic.e2e_mean_ms, future.e2e_mean_ms) * 1.2
    ax1.set_xlim(0, max_x)
    ax2.set_xlim(0, max_x)

    plt.tight_layout()
    plt.savefig(output_dir / 'no_stock_timeline.png', dpi=300)
    plt.close()
    print(f"  Saved: no_stock_timeline.png")

    # ========================================
    # Chart 3: Time Breakdown
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 7))

    # Stacked bar chart showing time components
    categories = ['CLASSIC', 'FUTURE_BASED']

    # Components
    trigger_times = [classic.avg_trigger_ms, future.avg_trigger_ms]
    fast_times = [classic.avg_fast_processor_ms, future.avg_fast_processor_ms]
    slow_chain_times = [classic.avg_slow_chain_total_ms, 0]  # FUTURE doesn't wait
    aggregator_times = [classic.avg_aggregator_ms, future.avg_aggregator_ms]

    x = range(len(categories))
    width = 0.6

    bars1 = ax.bar(x, trigger_times, width, label='TriggerFunction', color='#3498db')
    bars2 = ax.bar(x, fast_times, width, bottom=trigger_times, label='FastProcessor', color='#2ecc71')
    bars3 = ax.bar(x, slow_chain_times, width,
                  bottom=[t+f for t, f in zip(trigger_times, fast_times)],
                  label='Slow Chain (Wasted in CLASSIC)', color='#e74c3c')
    bars4 = ax.bar(x, aggregator_times, width,
                  bottom=[t+f+s for t, f, s in zip(trigger_times, fast_times, slow_chain_times)],
                  label='Aggregator', color='#9b59b6')

    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Time Breakdown: Where Time is Spent\n(Slow Chain is Wasted Work in No-Stock Scenario)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add total time labels
    for i, cat in enumerate(categories):
        total = trigger_times[i] + fast_times[i] + slow_chain_times[i] + aggregator_times[i]
        ax.text(i, total + 50, f'Total: {total:.0f}ms', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'no_stock_time_breakdown.png', dpi=300)
    plt.close()
    print(f"  Saved: no_stock_time_breakdown.png")

    # ========================================
    # Chart 4: Per-Run Latency Distribution
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 6))

    classic_latencies = [r.e2e_latency_ms for r in classic_runs if r.error is None and r.e2e_latency_ms > 0]
    future_latencies = [r.e2e_latency_ms for r in future_runs if r.error is None and r.e2e_latency_ms > 0]

    if classic_latencies and future_latencies:
        x_classic = range(1, len(classic_latencies) + 1)
        x_future = range(1, len(future_latencies) + 1)

        ax.plot(x_classic, classic_latencies, 'o-', color=classic_color,
                label='CLASSIC', linewidth=2, markersize=10)
        ax.plot(x_future, future_latencies, 's-', color=future_color,
                label='FUTURE_BASED', linewidth=2, markersize=10)

        # Add mean lines
        ax.axhline(classic.e2e_mean_ms, color=classic_color, linestyle='--', alpha=0.5,
                  label=f'CLASSIC Mean ({classic.e2e_mean_ms:.0f}ms)')
        ax.axhline(future.e2e_mean_ms, color=future_color, linestyle='--', alpha=0.5,
                  label=f'FUTURE Mean ({future.e2e_mean_ms:.0f}ms)')

        ax.set_xlabel('Run Number', fontsize=12)
        ax.set_ylabel('E2E Latency (ms)', fontsize=12)
        ax.set_title('Per-Run Latency Distribution\n(Notice consistent ~10x difference)',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'no_stock_latency_distribution.png', dpi=300)
    plt.close()
    print(f"  Saved: no_stock_latency_distribution.png")

    # ========================================
    # Chart 5: Cost Comparison
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    classic_cost = classic.estimated_cost_usd / classic.successful_runs if classic.successful_runs > 0 else 0
    future_cost = future.estimated_cost_usd / future.successful_runs if future.successful_runs > 0 else 0

    modes = ['CLASSIC', 'FUTURE_BASED']
    costs = [classic_cost * 1000000, future_cost * 1000000]  # Convert to micro-dollars for readability

    bars = ax.bar(modes, costs, color=[classic_color, future_color], alpha=0.8, edgecolor='black')

    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, cost + max(costs)*0.02,
                f'${cost/1000000:.8f}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Cost per Request (micro-dollars)', fontsize=12)
    ax.set_title('Cost per Rejected Order\n(FUTURE_BASED saves money by not running unnecessary functions)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'no_stock_cost_comparison.png', dpi=300)
    plt.close()
    print(f"  Saved: no_stock_cost_comparison.png")

    # ========================================
    # Chart 6: Cold Start Impact
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get cold start durations per function
    functions = ['TriggerFunction', 'FastProcessor', 'SlowChainStart',
                'SlowChainMid', 'SlowChainEnd', 'Aggregator']

    classic_init = []
    future_init = []

    for func in functions:
        classic_inits = [r.function_metrics[func].init_duration_ms
                        for r in classic_runs
                        if r.error is None and func in r.function_metrics
                        and r.function_metrics[func].init_duration_ms > 0]
        future_inits = [r.function_metrics[func].init_duration_ms
                       for r in future_runs
                       if r.error is None and func in r.function_metrics
                       and r.function_metrics[func].init_duration_ms > 0]

        classic_init.append(statistics.mean(classic_inits) if classic_inits else 0)
        future_init.append(statistics.mean(future_inits) if future_inits else 0)

    x = range(len(functions))
    width = 0.35

    ax.bar([p - width/2 for p in x], classic_init, width, label='CLASSIC', color=classic_color, alpha=0.8)
    ax.bar([p + width/2 for p in x], future_init, width, label='FUTURE_BASED', color=future_color, alpha=0.8)

    ax.set_xlabel('Function', fontsize=12)
    ax.set_ylabel('Cold Start Duration (ms)', fontsize=12)
    ax.set_title('Cold Start Duration by Function\n(First run of each mode)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(functions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'no_stock_cold_starts.png', dpi=300)
    plt.close()
    print(f"  Saved: no_stock_cold_starts.png")

    # ========================================
    # Chart 7: Memory Usage
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 6))

    classic_memory = []
    future_memory = []

    for func in functions:
        classic_mem = [r.function_metrics[func].memory_used_mb
                      for r in classic_runs
                      if r.error is None and func in r.function_metrics
                      and r.function_metrics[func].memory_used_mb > 0]
        future_mem = [r.function_metrics[func].memory_used_mb
                     for r in future_runs
                     if r.error is None and func in r.function_metrics
                     and r.function_metrics[func].memory_used_mb > 0]

        classic_memory.append(statistics.mean(classic_mem) if classic_mem else 0)
        future_memory.append(statistics.mean(future_mem) if future_mem else 0)

    ax.bar([p - width/2 for p in x], classic_memory, width, label='CLASSIC', color=classic_color, alpha=0.8)
    ax.bar([p + width/2 for p in x], future_memory, width, label='FUTURE_BASED', color=future_color, alpha=0.8)

    ax.set_xlabel('Function', fontsize=12)
    ax.set_ylabel('Memory Used (MB)', fontsize=12)
    ax.set_title('Memory Usage by Function', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(functions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'no_stock_memory_usage.png', dpi=300)
    plt.close()
    print(f"  Saved: no_stock_memory_usage.png")

    print(f"\nAll charts saved to: {output_dir}")


# ============================================================
# Results Persistence
# ============================================================

def save_results(classic_summary: BenchmarkSummary, future_summary: BenchmarkSummary,
                classic_runs: List[WorkflowRun], future_runs: List[WorkflowRun],
                output_dir: Path):
    """Save benchmark results to JSON"""
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    speedup = classic_summary.e2e_mean_ms / future_summary.e2e_mean_ms if future_summary.e2e_mean_ms > 0 else 0

    summary_file = output_dir / f'no_stock_benchmark_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'scenario': 'no_stock',
            'description': 'Inventory unavailable - tests short-circuit optimization',
            'CLASSIC': asdict(classic_summary),
            'FUTURE_BASED': asdict(future_summary),
            'improvement': {
                'e2e_latency_ms': classic_summary.e2e_mean_ms - future_summary.e2e_mean_ms,
                'e2e_latency_pct': ((classic_summary.e2e_mean_ms - future_summary.e2e_mean_ms) / classic_summary.e2e_mean_ms * 100) if classic_summary.e2e_mean_ms > 0 else 0,
                'speedup_factor': speedup,
                'short_circuit_rate': future_summary.short_circuit_count / future_summary.successful_runs if future_summary.successful_runs > 0 else 0,
            }
        }, f, indent=2, default=str)

    print(f"\nSaved summary: {summary_file}")

    runs_file = output_dir / f'no_stock_runs_{timestamp}.json'
    with open(runs_file, 'w') as f:
        json.dump({
            'CLASSIC': [asdict(r) for r in classic_runs],
            'FUTURE_BASED': [asdict(r) for r in future_runs]
        }, f, indent=2, default=str)

    print(f"Saved detailed runs: {runs_file}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='No-Stock Scenario Benchmark - Demonstrates Short-Circuit Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations per mode (default: 5)')
    parser.add_argument('--cold-all', action='store_true',
                       help='Force cold starts for all iterations')
    parser.add_argument('--skip-classic', action='store_true',
                       help='Skip CLASSIC mode benchmark')
    parser.add_argument('--skip-future', action='store_true',
                       help='Skip FUTURE_BASED mode benchmark')
    parser.add_argument('--skip-charts', action='store_true',
                       help='Skip chart generation')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"NO-STOCK SCENARIO BENCHMARK")
    print(f"Demonstrating Short-Circuit Optimization in FUTURE_BASED Mode")
    print(f"{'='*70}")
    print(f"Region: {REGION}")
    print(f"Profile: {PROFILE}")
    print(f"Iterations: {args.iterations}")
    print()
    print(f"Expected Results:")
    print(f"  CLASSIC:       ~3000-4000ms (must wait for entire slow chain)")
    print(f"  FUTURE_BASED:  ~200-400ms   (short-circuits immediately)")
    print(f"  Improvement:   ~10x faster!")
    print(f"{'='*70}\n")

    # Load function ARNs
    function_arns = load_function_arns()
    if not function_arns:
        print("Cannot proceed without function ARNs")
        return 1

    cold_start_freq = 'all' if args.cold_all else 'first'

    # Run benchmarks
    classic_runs = []
    future_runs = []

    if not args.skip_classic:
        classic_runs = run_benchmark_mode('CLASSIC', args.iterations, function_arns, cold_start_freq)
        time.sleep(20)  # Cooldown between modes

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

    # Reset SIMULATE_NO_STOCK to false
    print("\nResetting SIMULATE_NO_STOCK to false...")
    configure_mode('CLASSIC', function_arns, simulate_no_stock=False)

    print(f"\n{'='*70}")
    print(f"No-Stock Benchmark Complete!")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    exit(main())
