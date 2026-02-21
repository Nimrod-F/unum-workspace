#!/usr/bin/env python3
"""
Smart Factory IoT Alert Workflow - Benchmark Suite

Benchmarks CLASSIC vs FUTURE_BASED execution modes for the asymmetric
race topology with 3 parallel branches of different depths.

Features:
- Loads function ARNs dynamically from function-arn.yaml
- Supports --force-critical flag to test short-circuit (CRITICAL_STOP)
- Collects detailed CloudWatch metrics per function
- Generates comparison charts and histograms
- Detects short-circuit behavior in FUTURE_BASED + force_critical runs

Expected Results (Normal):
  CLASSIC:       ~2200ms (Branch C critical path + ActionDispatcher)
  FUTURE_BASED:  ~2100ms (cold start hidden behind Branch C)

Expected Results (Force Critical):
  CLASSIC:       ~2200ms (must wait for all branches before dispatching)
  FUTURE_BASED:  ~300ms  (short-circuits after SafetyCheck!)

Usage:
    python benchmark_iot.py --iterations 5
    python benchmark_iot.py --iterations 5 --force-critical
    python benchmark_iot.py --iterations 10 --cold-all
    python benchmark_iot.py --skip-classic --force-critical
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
DYNAMODB_TABLE = 'unum-intermediate-datastore-iot'

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
    'SensorIngest': {'order': 1, 'expected_ms': 50, 'description': 'Entry point (4KB payload)'},
    'SafetyCheck': {'order': 2, 'expected_ms': 100, 'description': 'Branch A - Fast path'},
    'MachineState': {'order': 3, 'expected_ms': 200, 'description': 'Branch B step 1'},
    'ShiftCheck': {'order': 4, 'expected_ms': 300, 'description': 'Branch B step 2 (terminal)'},
    'Windowing': {'order': 5, 'expected_ms': 400, 'description': 'Branch C step 1'},
    'ComputeFFT': {'order': 6, 'expected_ms': 600, 'description': 'Branch C step 2'},
    'FailureModel': {'order': 7, 'expected_ms': 1000, 'description': 'Branch C step 3 (terminal)'},
    'ActionDispatcher': {'order': 8, 'expected_ms': 200, 'description': 'Fan-in aggregator'},
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
    machine_id: str
    start_time: float
    end_time: float

    # Scenario
    force_critical: bool = False

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
    safety_check_time_ms: float = 0.0
    context_chain_total_ms: float = 0.0
    heavy_chain_total_ms: float = 0.0
    aggregator_invocation_delay_ms: float = 0.0

    # Short-circuit metrics
    short_circuited: bool = False
    time_saved_ms: float = 0.0
    invoker_branch: str = ''

    # Errors
    error: Optional[str] = None

    def calculate_aggregates(self):
        """Calculate aggregate metrics from function data"""
        self.total_duration_ms = sum(m.duration_ms for m in self.function_metrics.values())
        self.total_billed_ms = sum(m.billed_duration_ms for m in self.function_metrics.values())
        self.cold_start_count = sum(1 for m in self.function_metrics.values() if m.cold_start)
        self.total_init_ms = sum(m.init_duration_ms for m in self.function_metrics.values())

        # Branch A: SafetyCheck
        if 'SafetyCheck' in self.function_metrics:
            self.safety_check_time_ms = self.function_metrics['SafetyCheck'].duration_ms

        # Branch B: MachineState + ShiftCheck
        context_times = []
        for func in ['MachineState', 'ShiftCheck']:
            if func in self.function_metrics:
                context_times.append(self.function_metrics[func].duration_ms)
        self.context_chain_total_ms = sum(context_times)

        # Branch C: Windowing + ComputeFFT + FailureModel
        heavy_times = []
        for func in ['Windowing', 'ComputeFFT', 'FailureModel']:
            if func in self.function_metrics:
                heavy_times.append(self.function_metrics[func].duration_ms)
        self.heavy_chain_total_ms = sum(heavy_times)

        # Aggregator invocation delay
        if 'ActionDispatcher' in self.function_metrics:
            agg_invocation_ts = self.function_metrics['ActionDispatcher'].invocation_time_ms
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
    aggregator_invocation_min_ms: float = 0.0
    aggregator_invocation_max_ms: float = 0.0

    # Per-function averages
    avg_sensor_ingest_ms: float = 0.0
    avg_safety_check_ms: float = 0.0
    avg_machine_state_ms: float = 0.0
    avg_shift_check_ms: float = 0.0
    avg_windowing_ms: float = 0.0
    avg_compute_fft_ms: float = 0.0
    avg_failure_model_ms: float = 0.0
    avg_action_dispatcher_ms: float = 0.0

    # Branch totals
    avg_context_chain_ms: float = 0.0
    avg_heavy_chain_ms: float = 0.0

    # Short-circuit metrics
    short_circuit_count: int = 0
    avg_time_saved_ms: float = 0.0

    # Cold starts
    total_cold_starts: int = 0
    avg_init_duration_ms: float = 0.0

    # Invoker distribution
    invoker_distribution: Dict[str, int] = field(default_factory=dict)

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
        print("Deploy the workflow first using: unum-cli.py deploy -t unum-template.yaml")
        return {}

    with open(yaml_path, 'r') as f:
        arns = yaml.safe_load(f)

    print(f"Loaded {len(arns)} function ARNs from function-arn.yaml")
    for name, arn in arns.items():
        print(f"  {name}: {arn.split(':')[-1]}")
    return arns


def extract_function_name(arn: str) -> str:
    """Extract physical function name from ARN for CloudWatch logs"""
    return arn.split(':')[-1]


# ============================================================
# Lambda Configuration
# ============================================================

def configure_mode(mode: str, function_arns: Dict[str, str]):
    """Configure Lambda functions for the specified execution mode"""
    print(f"\n{'='*70}")
    print(f"Configuring {mode} Mode")
    print(f"{'='*70}")

    config = MODE_CONFIGS[mode]

    for func_name, arn in function_arns.items():
        try:
            response = lambda_client.get_function_configuration(FunctionName=arn)
            current_env = response.get('Environment', {}).get('Variables', {})

            current_env['EAGER'] = config['EAGER']

            # Set UNUM_FUTURE_BASED only on ActionDispatcher
            if func_name == 'ActionDispatcher':
                current_env['UNUM_FUTURE_BASED'] = config['UNUM_FUTURE_BASED']

            lambda_client.update_function_configuration(
                FunctionName=arn,
                Environment={'Variables': current_env}
            )

            extra = ""
            if func_name == 'ActionDispatcher':
                extra = f", UNUM_FUTURE_BASED={config['UNUM_FUTURE_BASED']}"
            print(f"  {func_name}: EAGER={config['EAGER']}{extra}")

        except Exception as e:
            print(f"  {func_name}: Error - {e}")

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
            current_env['COLD_START_TRIGGER'] = timestamp

            lambda_client.update_function_configuration(
                FunctionName=arn,
                Environment={'Variables': current_env}
            )
            print(f"    {func_name}")

        except Exception as e:
            print(f"    {func_name}: Error - {e}")

    print("  Waiting 15s for updates to propagate...")
    time.sleep(15)
    print("  Cold starts forced")


# ============================================================
# Workflow Invocation
# ============================================================

def invoke_workflow(machine_id: str, function_arns: Dict[str, str],
                    force_critical: bool = False) -> Dict:
    """Invoke the workflow and measure end-to-end latency"""
    trigger_arn = function_arns.get('SensorIngest')

    if not trigger_arn:
        return {
            'success': False,
            'machine_id': machine_id,
            'error': 'SensorIngest ARN not found'
        }

    # Wrap payload in unum format
    payload = {
        "Data": {
            "Source": "http",
            "Value": {
                "machine_id": machine_id,
                "sensor_id": "SENS-BENCH-001",
                "force_critical": force_critical
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
                'machine_id': machine_id,
                'error': f"Lambda function error: {response_payload}"
            }

        return {
            'success': True,
            'machine_id': machine_id,
            'start_time': start_time,
            'trigger_end_time': trigger_end_time,
            'trigger_latency_ms': trigger_latency_ms,
            'response': response_payload
        }

    except Exception as e:
        return {
            'success': False,
            'machine_id': machine_id,
            'error': str(e)
        }


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
        print(f"      Warning: Could not fetch logs for {function_name}: {e}")
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

    # Parse REPORT line
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

        # Check if we got ActionDispatcher metrics (critical function)
        if all_metrics.get('ActionDispatcher', LambdaMetrics('')).duration_ms > 0:
            break

        if attempt < max_retries:
            print(f"      ActionDispatcher metrics not found, waiting 5s...")
            time.sleep(5)

    return all_metrics


# ============================================================
# Benchmark Execution
# ============================================================

def run_single_iteration(run_id: int, mode: str, machine_id: str,
                         function_arns: Dict[str, str],
                         force_critical: bool = False,
                         force_cold: bool = False) -> WorkflowRun:
    """Run a single workflow iteration"""
    print(f"\n  [{run_id}] Machine: {machine_id}, force_critical={force_critical}")

    if force_cold:
        force_cold_starts(function_arns)

    result = invoke_workflow(machine_id, function_arns, force_critical)

    if not result['success']:
        print(f"    Invocation failed: {result.get('error')}")
        return WorkflowRun(
            run_id=run_id,
            mode=mode,
            machine_id=machine_id,
            start_time=time.time(),
            end_time=time.time(),
            force_critical=force_critical,
            error=result.get('error')
        )

    print(f"    Trigger latency: {result['trigger_latency_ms']:.2f}ms")

    # Wait for workflow to complete
    # Shorter wait for FUTURE_BASED + force_critical (short-circuit expected)
    if force_critical and mode == 'FUTURE_BASED':
        wait_time = 5
    else:
        wait_time = 12
    print(f"    Waiting {wait_time}s for workflow completion...")
    time.sleep(wait_time)

    end_time = time.time()
    metrics = collect_all_metrics(function_arns, result['start_time'], end_time)

    run = WorkflowRun(
        run_id=run_id,
        mode=mode,
        machine_id=machine_id,
        start_time=result['start_time'],
        end_time=end_time,
        force_critical=force_critical,
        trigger_latency_ms=result['trigger_latency_ms'],
        function_metrics=metrics
    )

    # Calculate e2e latency from ActionDispatcher completion time
    if metrics.get('ActionDispatcher', LambdaMetrics('')).invocation_time_ms > 0:
        agg = metrics['ActionDispatcher']
        workflow_start_ms = int(result['start_time'] * 1000)
        workflow_end_ms = agg.invocation_time_ms + int(agg.duration_ms)
        run.e2e_latency_ms = workflow_end_ms - workflow_start_ms

    run.calculate_aggregates()

    # Determine invoker branch
    agg_m = metrics.get('ActionDispatcher', LambdaMetrics(''))
    failure_m = metrics.get('FailureModel', LambdaMetrics(''))
    if agg_m.invocation_time_ms > 0 and failure_m.invocation_time_ms > 0:
        failure_completed_ms = failure_m.invocation_time_ms + int(failure_m.duration_ms)
        if agg_m.invocation_time_ms < failure_completed_ms:
            run.invoker_branch = 'SafetyCheck'
        else:
            run.invoker_branch = 'FailureModel'
    elif agg_m.invocation_time_ms > 0:
        run.invoker_branch = 'SafetyCheck' if run.aggregator_invocation_delay_ms < 5000 else 'FailureModel'

    # Detect short-circuit
    if force_critical and mode == 'FUTURE_BASED' and agg_m.invocation_time_ms > 0:
        if failure_m.invocation_time_ms > 0:
            failure_completed_ms = failure_m.invocation_time_ms + int(failure_m.duration_ms)
            agg_completed_ms = agg_m.invocation_time_ms + int(agg_m.duration_ms)
            if agg_completed_ms < failure_completed_ms:
                run.short_circuited = True
                run.time_saved_ms = failure_completed_ms - agg_completed_ms
        else:
            run.short_circuited = True
            run.time_saved_ms = 1900  # Approximate

    # Print summary
    print(f"    Functions:")
    for func_name in sorted(metrics.keys(),
                            key=lambda x: FUNCTION_INFO.get(x, {}).get('order', 99)):
        m = metrics[func_name]
        if m.duration_ms > 0:
            cold_tag = " [COLD]" if m.cold_start else ""
            print(f"      {func_name:20s}: {m.duration_ms:7.2f}ms{cold_tag}")

    if run.e2e_latency_ms > 0:
        print(f"    E2E Latency: {run.e2e_latency_ms:.2f}ms")

    if run.aggregator_invocation_delay_ms > 0:
        print(f"    ActionDispatcher invoked at: +{run.aggregator_invocation_delay_ms:.2f}ms (by {run.invoker_branch})")

    if force_critical:
        if run.short_circuited:
            print(f"    SHORT-CIRCUIT: Yes (saved ~{run.time_saved_ms:.0f}ms)")
        else:
            print(f"    SHORT-CIRCUIT: No (waited for all branches)")

    return run


def run_benchmark_mode(mode: str, iterations: int, function_arns: Dict[str, str],
                       force_critical: bool = False,
                       cold_start_freq: str = 'first') -> List[WorkflowRun]:
    """Run benchmark for a specific mode"""
    scenario = "force_critical" if force_critical else "normal"

    print(f"\n{'='*70}")
    print(f"Running {mode} Benchmark ({scenario})")
    print(f"Iterations: {iterations}")
    print(f"Cold start frequency: {cold_start_freq}")
    print(f"{'='*70}")

    configure_mode(mode, function_arns)

    runs = []

    for i in range(iterations):
        force_cold = (cold_start_freq == 'all') or (cold_start_freq == 'first' and i == 0)
        machine_id = f"MACH-{mode}-{i+1:03d}-{int(time.time())}"

        run = run_single_iteration(
            i + 1, mode, machine_id, function_arns,
            force_critical=force_critical,
            force_cold=force_cold
        )
        runs.append(run)

        if i < iterations - 1:
            print(f"  Waiting 5s before next iteration...")
            time.sleep(5)

    return runs


# ============================================================
# Results Analysis
# ============================================================

FUNC_ATTR_MAP = {
    'SensorIngest': 'avg_sensor_ingest_ms',
    'SafetyCheck': 'avg_safety_check_ms',
    'MachineState': 'avg_machine_state_ms',
    'ShiftCheck': 'avg_shift_check_ms',
    'Windowing': 'avg_windowing_ms',
    'ComputeFFT': 'avg_compute_fft_ms',
    'FailureModel': 'avg_failure_model_ms',
    'ActionDispatcher': 'avg_action_dispatcher_ms',
}


def compute_summary(mode: str, runs: List[WorkflowRun],
                    force_critical: bool = False) -> BenchmarkSummary:
    """Compute statistical summary from runs"""
    successful = [r for r in runs if r.error is None and r.e2e_latency_ms > 0]
    failed = [r for r in runs if r.error is not None]

    scenario = "force_critical" if force_critical else "normal"

    summary = BenchmarkSummary(
        mode=mode,
        scenario=scenario,
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
    agg_timings = [r.aggregator_invocation_delay_ms for r in successful
                   if r.aggregator_invocation_delay_ms > 0]
    if agg_timings:
        summary.aggregator_invocation_mean_ms = statistics.mean(agg_timings)
        summary.aggregator_invocation_median_ms = statistics.median(agg_timings)
        summary.aggregator_invocation_min_ms = min(agg_timings)
        summary.aggregator_invocation_max_ms = max(agg_timings)

    # Per-function averages
    for func_name, attr_name in FUNC_ATTR_MAP.items():
        durations = [r.function_metrics[func_name].duration_ms
                     for r in successful
                     if func_name in r.function_metrics
                     and r.function_metrics[func_name].duration_ms > 0]
        if durations:
            setattr(summary, attr_name, statistics.mean(durations))

    # Branch totals
    context_totals = [r.context_chain_total_ms for r in successful if r.context_chain_total_ms > 0]
    if context_totals:
        summary.avg_context_chain_ms = statistics.mean(context_totals)

    heavy_totals = [r.heavy_chain_total_ms for r in successful if r.heavy_chain_total_ms > 0]
    if heavy_totals:
        summary.avg_heavy_chain_ms = statistics.mean(heavy_totals)

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

    # Invoker distribution
    for r in successful:
        if r.invoker_branch:
            summary.invoker_distribution[r.invoker_branch] = \
                summary.invoker_distribution.get(r.invoker_branch, 0) + 1

    # Cost calculation (eu-central-1 pricing)
    summary.total_billed_ms = sum(r.total_billed_ms for r in successful)
    gb_seconds = (summary.total_billed_ms / 1000) * 0.25
    lambda_compute_cost = gb_seconds * 0.0000166667
    request_cost = len(successful) * 8 * 0.0000002  # 8 functions per workflow
    summary.estimated_cost_usd = lambda_compute_cost + request_cost

    return summary


def print_summary(summary: BenchmarkSummary):
    """Print formatted summary"""
    print(f"\n{'='*70}")
    print(f"{summary.mode} Benchmark Summary ({summary.scenario})")
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

        print(f"ActionDispatcher Invocation Timing:")
        print(f"  Mean:   {summary.aggregator_invocation_mean_ms:8.2f}ms")
        print(f"  Median: {summary.aggregator_invocation_median_ms:8.2f}ms")
        print()

        print(f"Per-Function Averages:")
        print(f"  SensorIngest:     {summary.avg_sensor_ingest_ms:8.2f}ms")
        print(f"  SafetyCheck:      {summary.avg_safety_check_ms:8.2f}ms")
        print(f"  MachineState:     {summary.avg_machine_state_ms:8.2f}ms")
        print(f"  ShiftCheck:       {summary.avg_shift_check_ms:8.2f}ms")
        print(f"  Windowing:        {summary.avg_windowing_ms:8.2f}ms")
        print(f"  ComputeFFT:       {summary.avg_compute_fft_ms:8.2f}ms")
        print(f"  FailureModel:     {summary.avg_failure_model_ms:8.2f}ms")
        print(f"  ActionDispatcher: {summary.avg_action_dispatcher_ms:8.2f}ms")
        print()

        print(f"Branch Totals:")
        print(f"  Branch B (context): {summary.avg_context_chain_ms:8.2f}ms")
        print(f"  Branch C (heavy):   {summary.avg_heavy_chain_ms:8.2f}ms")
        print()

        if summary.short_circuit_count > 0:
            print(f"Short-Circuit:")
            print(f"  Count:      {summary.short_circuit_count}/{summary.successful_runs}")
            print(f"  Avg Saved:  {summary.avg_time_saved_ms:8.2f}ms")
            print()

        print(f"Cold Starts: {summary.total_cold_starts}")
        if summary.avg_init_duration_ms > 0:
            print(f"  Avg Init:   {summary.avg_init_duration_ms:8.2f}ms")
        print()

        print(f"Invoker Distribution:")
        for invoker, count in summary.invoker_distribution.items():
            print(f"  {invoker}: {count} ({count / summary.successful_runs * 100:.1f}%)")
        print()

        print(f"Estimated Cost: ${summary.estimated_cost_usd:.6f}")

    print(f"{'='*70}\n")


def compare_results(classic: BenchmarkSummary, future: BenchmarkSummary):
    """Print comparison between CLASSIC and FUTURE_BASED modes"""
    print(f"\n{'='*70}")
    print(f"CLASSIC vs FUTURE_BASED Comparison ({classic.scenario})")
    print(f"{'='*70}\n")

    # E2E Latency improvement
    e2e_improvement_ms = classic.e2e_mean_ms - future.e2e_mean_ms
    e2e_improvement_pct = (e2e_improvement_ms / classic.e2e_mean_ms * 100) if classic.e2e_mean_ms > 0 else 0
    speedup = classic.e2e_mean_ms / future.e2e_mean_ms if future.e2e_mean_ms > 0 else 0

    print(f"E2E Latency:")
    print(f"  CLASSIC:       {classic.e2e_mean_ms:8.2f}ms")
    print(f"  FUTURE_BASED:  {future.e2e_mean_ms:8.2f}ms")
    print(f"  Improvement:   {e2e_improvement_ms:8.2f}ms ({e2e_improvement_pct:+.1f}%)")
    if speedup > 1.5:
        print(f"  Speedup:       {speedup:.1f}x faster!")
    print()

    # Aggregator invocation timing
    agg_improvement_ms = classic.aggregator_invocation_mean_ms - future.aggregator_invocation_mean_ms
    agg_improvement_pct = (agg_improvement_ms / classic.aggregator_invocation_mean_ms * 100) \
        if classic.aggregator_invocation_mean_ms > 0 else 0

    print(f"ActionDispatcher Invocation Delay:")
    print(f"  CLASSIC:       {classic.aggregator_invocation_mean_ms:8.2f}ms  (by FailureModel)")
    print(f"  FUTURE_BASED:  {future.aggregator_invocation_mean_ms:8.2f}ms  (by SafetyCheck)")
    print(f"  Improvement:   {agg_improvement_ms:8.2f}ms ({agg_improvement_pct:+.1f}%)")
    print()

    if future.short_circuit_count > 0:
        print(f"Short-Circuit Analysis:")
        print(f"  FUTURE_BASED short-circuits: {future.short_circuit_count}/{future.successful_runs}")
        print(f"  Avg time saved:              {future.avg_time_saved_ms:.0f}ms")
        print()

    # Cost comparison
    if classic.successful_runs > 0 and future.successful_runs > 0:
        classic_cost_per = classic.estimated_cost_usd / classic.successful_runs
        future_cost_per = future.estimated_cost_usd / future.successful_runs
        cost_savings_pct = ((classic_cost_per - future_cost_per) / classic_cost_per * 100) \
            if classic_cost_per > 0 else 0

        print(f"Cost per Workflow:")
        print(f"  CLASSIC:       ${classic_cost_per:.8f}")
        print(f"  FUTURE_BASED:  ${future_cost_per:.8f}")
        print(f"  Savings:       {cost_savings_pct:+.1f}%")
        print()

    print(f"{'='*70}\n")


# ============================================================
# Chart Generation
# ============================================================

def generate_charts(classic: BenchmarkSummary, future: BenchmarkSummary,
                    classic_runs: List[WorkflowRun], future_runs: List[WorkflowRun],
                    output_dir: Path, force_critical: bool = False):
    """Generate comparison charts"""
    if not CHARTS_AVAILABLE:
        print("Skipping chart generation (matplotlib not available)")
        return

    print("\nGenerating charts...")
    output_dir.mkdir(exist_ok=True)

    scenario = "critical" if force_critical else "normal"
    classic_color = '#e74c3c'
    future_color = '#27ae60'

    # ========================================
    # Chart 1: E2E Latency Comparison
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 7))

    modes = ['CLASSIC', 'FUTURE_BASED']
    means = [classic.e2e_mean_ms, future.e2e_mean_ms]
    mins = [classic.e2e_min_ms, future.e2e_min_ms]
    maxs = [classic.e2e_max_ms, future.e2e_max_ms]

    bars = ax.bar(modes, means, color=[classic_color, future_color], alpha=0.8, edgecolor='black')
    ax.errorbar(range(len(modes)), means,
                yerr=[[m - mn for m, mn in zip(means, mins)],
                      [mx - m for m, mx in zip(means, maxs)]],
                fmt='none', color='black', capsize=5)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(means) * 0.02,
                f'{val:.0f}ms', ha='center', va='bottom', fontsize=14, fontweight='bold')

    improvement_pct = (classic.e2e_mean_ms - future.e2e_mean_ms) / classic.e2e_mean_ms * 100 \
        if classic.e2e_mean_ms > 0 else 0

    title = f'E2E Latency: CLASSIC vs FUTURE_BASED ({scenario})'
    ax.set_ylabel('E2E Latency (ms)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    ax.text(0.5, max(means) * 0.7,
            f'{improvement_pct:+.1f}%', ha='center', fontsize=16,
            fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f'e2e_latency_{scenario}.png', dpi=300)
    plt.close()
    print(f"  Saved: e2e_latency_{scenario}.png")

    # ========================================
    # Chart 2: Per-Function Duration
    # ========================================
    fig, ax = plt.subplots(figsize=(14, 7))

    functions = list(FUNCTION_INFO.keys())
    classic_times = [
        classic.avg_sensor_ingest_ms, classic.avg_safety_check_ms,
        classic.avg_machine_state_ms, classic.avg_shift_check_ms,
        classic.avg_windowing_ms, classic.avg_compute_fft_ms,
        classic.avg_failure_model_ms, classic.avg_action_dispatcher_ms
    ]
    future_times = [
        future.avg_sensor_ingest_ms, future.avg_safety_check_ms,
        future.avg_machine_state_ms, future.avg_shift_check_ms,
        future.avg_windowing_ms, future.avg_compute_fft_ms,
        future.avg_failure_model_ms, future.avg_action_dispatcher_ms
    ]

    x_pos = range(len(functions))
    width = 0.35

    ax.bar([p - width / 2 for p in x_pos], classic_times, width,
           label='CLASSIC', color=classic_color, alpha=0.7)
    ax.bar([p + width / 2 for p in x_pos], future_times, width,
           label='FUTURE_BASED', color=future_color, alpha=0.7)

    ax.set_xlabel('Function', fontsize=12)
    ax.set_ylabel('Average Duration (ms)', fontsize=12)
    ax.set_title(f'Per-Function Duration ({scenario})', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(functions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'per_function_{scenario}.png', dpi=300)
    plt.close()
    print(f"  Saved: per_function_{scenario}.png")

    # ========================================
    # Chart 3: Latency Histogram
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 6))

    classic_latencies = [r.e2e_latency_ms for r in classic_runs
                         if r.error is None and r.e2e_latency_ms > 0]
    future_latencies = [r.e2e_latency_ms for r in future_runs
                        if r.error is None and r.e2e_latency_ms > 0]

    if classic_latencies and future_latencies:
        all_latencies = classic_latencies + future_latencies
        bin_min = min(all_latencies) * 0.8
        bin_max = max(all_latencies) * 1.2
        bins = 15

        ax.hist(classic_latencies, bins=bins, range=(bin_min, bin_max),
                alpha=0.6, color=classic_color, label=f'CLASSIC (mean={classic.e2e_mean_ms:.0f}ms)',
                edgecolor='black')
        ax.hist(future_latencies, bins=bins, range=(bin_min, bin_max),
                alpha=0.6, color=future_color, label=f'FUTURE (mean={future.e2e_mean_ms:.0f}ms)',
                edgecolor='black')

        ax.axvline(classic.e2e_mean_ms, color=classic_color, linestyle='--', linewidth=2)
        ax.axvline(future.e2e_mean_ms, color=future_color, linestyle='--', linewidth=2)

        ax.set_xlabel('E2E Latency (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Latency Distribution ({scenario})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'latency_histogram_{scenario}.png', dpi=300)
    plt.close()
    print(f"  Saved: latency_histogram_{scenario}.png")

    # ========================================
    # Chart 4: Execution Timeline (Gantt)
    # ========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    functions_order = list(FUNCTION_INFO.keys())

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
                             color=color, alpha=0.7, edgecolor='black')
                    ax1.text(start_offset + m.duration_ms / 2, i, f'{m.duration_ms:.0f}ms',
                             ha='center', va='center', fontsize=8, fontweight='bold')

        ax1.set_yticks(range(len(functions_order)))
        ax1.set_yticklabels(functions_order)
        ax1.set_xlabel('Time from Workflow Start (ms)', fontsize=11)
        ax1.set_title(f'CLASSIC Mode Timeline ({scenario})', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        if run.aggregator_invocation_delay_ms > 0:
            ax1.axvline(run.aggregator_invocation_delay_ms, color='red', linestyle='--',
                        label=f'ActionDispatcher at +{run.aggregator_invocation_delay_ms:.0f}ms')
            ax1.legend(fontsize=9)

    # FUTURE_BASED timeline
    future_successful = [r for r in future_runs if r.error is None and r.e2e_latency_ms > 0]
    if future_successful:
        run = future_successful[0]
        for i, func_name in enumerate(functions_order):
            if func_name in run.function_metrics:
                m = run.function_metrics[func_name]
                if m.duration_ms > 0:
                    start_offset = max(0, m.invocation_time_ms - int(run.start_time * 1000))
                    color = '#229954' if m.cold_start else future_color
                    ax2.barh(i, m.duration_ms, left=start_offset, height=0.6,
                             color=color, alpha=0.7, edgecolor='black')
                    ax2.text(start_offset + m.duration_ms / 2, i, f'{m.duration_ms:.0f}ms',
                             ha='center', va='center', fontsize=8, fontweight='bold')

        ax2.set_yticks(range(len(functions_order)))
        ax2.set_yticklabels(functions_order)
        ax2.set_xlabel('Time from Workflow Start (ms)', fontsize=11)
        ax2.set_title(f'FUTURE_BASED Mode Timeline ({scenario})', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        if run.aggregator_invocation_delay_ms > 0:
            ax2.axvline(run.aggregator_invocation_delay_ms, color='green', linestyle='--',
                        label=f'ActionDispatcher at +{run.aggregator_invocation_delay_ms:.0f}ms')
            ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f'execution_timeline_{scenario}.png', dpi=300)
    plt.close()
    print(f"  Saved: execution_timeline_{scenario}.png")

    print(f"\nAll charts saved to: {output_dir}")


# ============================================================
# Results Persistence
# ============================================================

def save_results(classic_summary: BenchmarkSummary, future_summary: BenchmarkSummary,
                 classic_runs: List[WorkflowRun], future_runs: List[WorkflowRun],
                 output_dir: Path, force_critical: bool = False):
    """Save benchmark results to JSON"""
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    scenario = "critical" if force_critical else "normal"

    speedup = classic_summary.e2e_mean_ms / future_summary.e2e_mean_ms \
        if future_summary.e2e_mean_ms > 0 else 0

    summary_file = output_dir / f'benchmark_summary_{scenario}_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'workflow': 'smart-factory-iot-workflow',
            'scenario': scenario,
            'force_critical': force_critical,
            'CLASSIC': asdict(classic_summary),
            'FUTURE_BASED': asdict(future_summary),
            'improvement': {
                'e2e_latency_ms': classic_summary.e2e_mean_ms - future_summary.e2e_mean_ms,
                'e2e_latency_pct': (
                    (classic_summary.e2e_mean_ms - future_summary.e2e_mean_ms)
                    / classic_summary.e2e_mean_ms * 100
                ) if classic_summary.e2e_mean_ms > 0 else 0,
                'speedup_factor': speedup,
                'aggregator_invocation_ms': (
                    classic_summary.aggregator_invocation_mean_ms
                    - future_summary.aggregator_invocation_mean_ms
                ),
            }
        }, f, indent=2, default=str)

    print(f"\nSaved summary: {summary_file}")

    runs_file = output_dir / f'benchmark_runs_{scenario}_{timestamp}.json'
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
        description='Smart Factory IoT Alert Workflow - Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_iot.py --iterations 5                # Normal mode
  python benchmark_iot.py --iterations 5 --force-critical  # Test short-circuit
  python benchmark_iot.py --iterations 10 --cold-all    # Force all cold starts
  python benchmark_iot.py --skip-classic --force-critical   # Only FUTURE_BASED
        """
    )

    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations per mode (default: 5)')
    parser.add_argument('--force-critical', action='store_true',
                        help='Send force_critical=true payload to test short-circuit')
    parser.add_argument('--cold-all', action='store_true',
                        help='Force cold starts for all iterations (default: only first)')
    parser.add_argument('--skip-classic', action='store_true',
                        help='Skip CLASSIC mode benchmark')
    parser.add_argument('--skip-future', action='store_true',
                        help='Skip FUTURE_BASED mode benchmark')
    parser.add_argument('--skip-charts', action='store_true',
                        help='Skip chart generation')

    args = parser.parse_args()

    scenario = "Force Critical (CRITICAL_STOP)" if args.force_critical else "Normal"

    print(f"\n{'='*70}")
    print(f"Smart Factory IoT Alert Workflow - Benchmark Suite")
    print(f"{'='*70}")
    print(f"Region: {REGION}")
    print(f"Profile: {PROFILE}")
    print(f"Iterations: {args.iterations}")
    print(f"Scenario: {scenario}")
    print()

    if args.force_critical:
        print(f"Expected Results (Force Critical):")
        print(f"  CLASSIC:       ~2200ms (must wait for Branch C before acting)")
        print(f"  FUTURE_BASED:  ~300ms  (short-circuits after SafetyCheck!)")
    else:
        print(f"Expected Results (Normal):")
        print(f"  CLASSIC:       ~2200ms (Branch C critical path)")
        print(f"  FUTURE_BASED:  ~2100ms (cold start hidden behind Branch C)")

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
        classic_runs = run_benchmark_mode(
            'CLASSIC', args.iterations, function_arns,
            force_critical=args.force_critical,
            cold_start_freq=cold_start_freq
        )
        time.sleep(20)  # Cooldown between modes

    if not args.skip_future:
        future_runs = run_benchmark_mode(
            'FUTURE_BASED', args.iterations, function_arns,
            force_critical=args.force_critical,
            cold_start_freq=cold_start_freq
        )

    # Compute summaries
    classic_summary = None
    future_summary = None

    if classic_runs:
        classic_summary = compute_summary('CLASSIC', classic_runs, args.force_critical)
        print_summary(classic_summary)

    if future_runs:
        future_summary = compute_summary('FUTURE_BASED', future_runs, args.force_critical)
        print_summary(future_summary)

    # Compare results
    if classic_summary and future_summary:
        compare_results(classic_summary, future_summary)

        # Save results
        results_dir = Path(__file__).parent / 'results'
        save_results(classic_summary, future_summary, classic_runs, future_runs,
                     results_dir, args.force_critical)

        # Generate charts
        if not args.skip_charts:
            generate_charts(classic_summary, future_summary, classic_runs, future_runs,
                            results_dir, args.force_critical)

    print(f"\n{'='*70}")
    print(f"Benchmark Complete!")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    exit(main())
