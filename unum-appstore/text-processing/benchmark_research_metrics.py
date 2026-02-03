#!/usr/bin/env python3
"""
Research-Grade Benchmark: Classic vs Future-Based Execution

Enhanced benchmark with comprehensive metrics for research publication:
1. Latency Metrics (E2E, cold/warm, P95, etc.)
2. Memory Metrics (per-function, aggregator, efficiency)
3. Cost Metrics (Lambda compute, requests, DynamoDB)
4. Resource Utilization (billed duration, cold starts)
5. Fan-In Metrics (pre-resolved, polling overhead)

Usage:
    python benchmark_research_metrics.py --iterations 10
    python benchmark_research_metrics.py --iterations 20 --cold-iterations 5
"""

import boto3
import json
import time
import uuid
import argparse
import threading
import statistics
import re
import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Configuration
REGION = 'eu-central-1'
PROFILE = 'research-profile'
STACK_NAME = 'unum-text-processing'
DYNAMODB_TABLE = 'unum-intermediate-datastore'

# AWS Pricing (eu-central-1, as of 2026)
PRICING = {
    'lambda_gb_second': 0.0000166667,  # $/GB-second
    'lambda_request': 0.0000002,        # $/request
    'dynamodb_wcu': 0.00000125,         # $/WCU
    'dynamodb_rcu': 0.00000025,         # $/RCU
}

# Test input
TEST_INPUT = "Hey @TechNews and @DevCommunity! Check out https://example.com/article and https://docs.example.com/guide for great resources!"


@dataclass
class LambdaMetrics:
    """Metrics from Lambda REPORT log line"""
    function_name: str
    request_id: str
    duration_ms: float = 0.0
    billed_duration_ms: float = 0.0
    memory_size_mb: int = 0
    max_memory_used_mb: int = 0
    init_duration_ms: Optional[float] = None  # Only present on cold starts
    
    @property
    def is_cold_start(self) -> bool:
        return self.init_duration_ms is not None
    
    @property
    def memory_efficiency(self) -> float:
        """Ratio of memory used to allocated"""
        if self.memory_size_mb > 0:
            return self.max_memory_used_mb / self.memory_size_mb
        return 0.0
    
    @property
    def compute_cost(self) -> float:
        """Lambda compute cost for this invocation"""
        gb_seconds = (self.billed_duration_ms / 1000) * (self.memory_size_mb / 1024)
        return gb_seconds * PRICING['lambda_gb_second'] + PRICING['lambda_request']


@dataclass
class BenchmarkResult:
    """Single benchmark run result with comprehensive metrics"""
    session_id: str
    mode: str
    iteration: int
    is_cold: bool
    
    # Timestamps (ms since epoch)
    invocation_time: int = 0
    user_mention_start: int = 0
    user_mention_end: int = 0
    find_url_start: int = 0
    find_url_end: int = 0
    shorten_url_start: int = 0
    shorten_url_end: int = 0
    create_post_start: int = 0
    create_post_end: int = 0
    publish_start: int = 0
    publish_end: int = 0
    
    # Per-function Lambda metrics
    lambda_metrics: Dict[str, LambdaMetrics] = field(default_factory=dict)
    
    # ===== LATENCY METRICS =====
    e2e_latency_ms: float = 0.0
    create_post_invocation_delay_ms: float = 0.0
    fan_in_overhead_ms: float = 0.0
    
    # ===== MEMORY METRICS =====
    total_memory_used_mb: int = 0          # Sum of all functions' max memory
    max_memory_used_mb: int = 0            # Peak memory across all functions
    aggregator_memory_mb: int = 0          # CreatePost memory (key metric)
    avg_memory_efficiency: float = 0.0     # Average used/allocated ratio
    
    # ===== COST METRICS =====
    total_lambda_cost: float = 0.0         # Total Lambda compute + request cost
    total_billed_duration_ms: float = 0.0  # Sum of billed durations
    
    # ===== COLD START METRICS =====
    cold_start_count: int = 0              # Number of cold starts in this run
    total_init_duration_ms: float = 0.0    # Sum of init durations
    
    # ===== FAN-IN METRICS (Future-Based specific) =====
    pre_resolved_count: int = 0            # Inputs ready when aggregator checked
    poll_count: int = 0                    # Number of DynamoDB polls
    
    def calculate_metrics(self):
        """Calculate all derived metrics"""
        # E2E latency
        if self.publish_end and self.invocation_time:
            self.e2e_latency_ms = self.publish_end - self.invocation_time
        
        # CreatePost invocation delay
        if self.create_post_start and self.invocation_time:
            self.create_post_invocation_delay_ms = self.create_post_start - self.invocation_time
        
        # Aggregate Lambda metrics
        if self.lambda_metrics:
            self.total_memory_used_mb = sum(m.max_memory_used_mb for m in self.lambda_metrics.values())
            self.max_memory_used_mb = max((m.max_memory_used_mb for m in self.lambda_metrics.values()), default=0)
            self.total_lambda_cost = sum(m.compute_cost for m in self.lambda_metrics.values())
            self.total_billed_duration_ms = sum(m.billed_duration_ms for m in self.lambda_metrics.values())
            self.cold_start_count = sum(1 for m in self.lambda_metrics.values() if m.is_cold_start)
            self.total_init_duration_ms = sum(m.init_duration_ms or 0 for m in self.lambda_metrics.values())
            
            # Memory efficiency
            efficiencies = [m.memory_efficiency for m in self.lambda_metrics.values() if m.memory_efficiency > 0]
            self.avg_memory_efficiency = statistics.mean(efficiencies) if efficiencies else 0.0
            
            # Aggregator (CreatePost) memory
            if 'CreatePost' in self.lambda_metrics:
                self.aggregator_memory_mb = self.lambda_metrics['CreatePost'].max_memory_used_mb


@dataclass
class ResearchSummary:
    """Comprehensive summary with research-grade metrics"""
    workflow: str
    mode: str
    total_iterations: int
    cold_iterations: int
    warm_iterations: int
    successful_iterations: int
    timestamp: str
    
    # ===== LATENCY METRICS =====
    e2e_latency_mean_ms: float = 0.0
    e2e_latency_median_ms: float = 0.0
    e2e_latency_std_ms: float = 0.0
    e2e_latency_min_ms: float = 0.0
    e2e_latency_max_ms: float = 0.0
    e2e_latency_p95_ms: float = 0.0
    e2e_latency_p99_ms: float = 0.0
    
    cold_e2e_mean_ms: float = 0.0
    cold_e2e_std_ms: float = 0.0
    warm_e2e_mean_ms: float = 0.0
    warm_e2e_std_ms: float = 0.0
    
    create_post_delay_mean_ms: float = 0.0
    create_post_delay_std_ms: float = 0.0
    
    # ===== MEMORY METRICS =====
    avg_total_memory_mb: float = 0.0
    avg_max_memory_mb: float = 0.0
    avg_aggregator_memory_mb: float = 0.0
    avg_memory_efficiency: float = 0.0
    
    # ===== COST METRICS =====
    avg_cost_per_run: float = 0.0
    total_cost: float = 0.0
    cost_per_1m_invocations: float = 0.0
    avg_billed_duration_ms: float = 0.0
    
    # ===== COLD START METRICS =====
    avg_cold_starts_per_run: float = 0.0
    avg_init_duration_ms: float = 0.0
    
    # Individual results
    results: List[Dict] = field(default_factory=list)


class ResearchBenchmark:
    """Research-grade benchmark runner with comprehensive metrics"""
    
    def __init__(self, profile: str = PROFILE, region: str = REGION):
        self.session = boto3.Session(profile_name=profile, region_name=region)
        self.lambda_client = self.session.client('lambda')
        self.logs_client = self.session.client('logs')
        self.cf_client = self.session.client('cloudformation')
        self.dynamodb_client = self.session.client('dynamodb')
        self.cloudwatch_client = self.session.client('cloudwatch')
        
        self.functions = self._get_function_names()
        self.log_groups = {name: f"/aws/lambda/{arn}" for name, arn in self.functions.items()}
        
    def _get_function_names(self) -> Dict[str, str]:
        """Get Lambda function names from CloudFormation"""
        response = self.cf_client.describe_stack_resources(StackName=STACK_NAME)
        functions = {}
        for resource in response['StackResources']:
            if resource['ResourceType'] == 'AWS::Lambda::Function':
                logical_id = resource['LogicalResourceId']
                physical_id = resource['PhysicalResourceId']
                for name in ['UserMention', 'FindUrl', 'ShortenUrl', 'CreatePost', 'Publish']:
                    if name in logical_id:
                        functions[name] = physical_id
                        break
        return functions
    
    def update_mode(self, mode: str):
        """Update Lambda environment variables for the specified mode"""
        eager_value = "true" if mode == "FUTURE_BASED" else "false"
        
        print(f"\n  Updating Lambda functions to {mode} mode (EAGER={eager_value})...")
        
        for name, func_name in self.functions.items():
            try:
                response = self.lambda_client.get_function_configuration(FunctionName=func_name)
                env_vars = response.get('Environment', {}).get('Variables', {})
                env_vars['EAGER'] = eager_value
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_name,
                    Environment={'Variables': env_vars}
                )
            except Exception as e:
                print(f"    Warning: Could not update {name}: {e}")
        
        print("  Waiting 5s for configuration updates...")
        time.sleep(5)
    
    def force_cold_start(self):
        """Force cold starts by updating function configurations"""
        print("  Forcing cold starts...")
        for name, func_name in self.functions.items():
            try:
                response = self.lambda_client.get_function_configuration(FunctionName=func_name)
                env_vars = response.get('Environment', {}).get('Variables', {})
                env_vars['COLD_START_TRIGGER'] = str(uuid.uuid4())[:8]
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_name,
                    Environment={'Variables': env_vars}
                )
            except Exception as e:
                print(f"    Warning: Could not trigger cold start for {name}: {e}")
        
        time.sleep(3)
    
    def invoke_workflow(self) -> Tuple[str, int]:
        """Invoke the workflow and return session ID and invocation timestamp"""
        workflow_session = str(uuid.uuid4())
        
        user_mention_payload = {
            "Session": workflow_session,
            "Fan-out": {"Index": 0},
            "Data": {"Source": "http", "Value": TEST_INPUT}
        }
        
        find_url_payload = {
            "Session": workflow_session,
            "Fan-out": {"Index": 1},
            "Data": {"Source": "http", "Value": TEST_INPUT}
        }
        
        invocation_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        # Invoke both start functions in parallel
        def invoke(func_name, payload):
            self.lambda_client.invoke(
                FunctionName=func_name,
                InvocationType='Event',
                Payload=json.dumps(payload)
            )
        
        t1 = threading.Thread(target=invoke, args=(self.functions['UserMention'], user_mention_payload))
        t2 = threading.Thread(target=invoke, args=(self.functions['FindUrl'], find_url_payload))
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        return workflow_session, invocation_time
    
    def wait_for_completion(self, session_id: str, invocation_time: int, timeout_seconds: int = 60) -> bool:
        """Wait for workflow completion by checking Publish logs"""
        deadline = time.time() + timeout_seconds
        
        while time.time() < deadline:
            try:
                response = self.logs_client.filter_log_events(
                    logGroupName=self.log_groups['Publish'],
                    startTime=invocation_time,
                    filterPattern=f"END RequestId"
                )
                
                if response.get('events', []):
                    return True
                    
            except Exception:
                pass
            
            time.sleep(1)
        
        return False
    
    def parse_report_log(self, log_message: str, function_name: str) -> Optional[LambdaMetrics]:
        """
        Parse Lambda REPORT log line to extract metrics.
        
        Example REPORT line:
        REPORT RequestId: abc123 Duration: 123.45 ms Billed Duration: 124 ms 
        Memory Size: 128 MB Max Memory Used: 67 MB Init Duration: 234.56 ms
        """
        report_pattern = r'REPORT RequestId: (\S+)\s+Duration: ([\d.]+) ms\s+Billed Duration: (\d+) ms\s+Memory Size: (\d+) MB\s+Max Memory Used: (\d+) MB(?:\s+Init Duration: ([\d.]+) ms)?'
        
        match = re.search(report_pattern, log_message)
        if match:
            return LambdaMetrics(
                function_name=function_name,
                request_id=match.group(1),
                duration_ms=float(match.group(2)),
                billed_duration_ms=float(match.group(3)),
                memory_size_mb=int(match.group(4)),
                max_memory_used_mb=int(match.group(5)),
                init_duration_ms=float(match.group(6)) if match.group(6) else None
            )
        return None
    
    def collect_metrics(self, session_id: str, invocation_time: int) -> Tuple[Dict[str, Dict], Dict[str, LambdaMetrics]]:
        """Collect timing metrics and Lambda REPORT metrics from CloudWatch logs"""
        timing_metrics = {}
        lambda_metrics = {}
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        for name, log_group in self.log_groups.items():
            try:
                # Get START events
                start_response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=invocation_time,
                    endTime=end_time,
                    filterPattern="START RequestId"
                )
                
                # Get END events
                end_response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=invocation_time,
                    endTime=end_time,
                    filterPattern="END RequestId"
                )
                
                # Get REPORT events (contains memory and duration metrics)
                report_response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=invocation_time,
                    endTime=end_time,
                    filterPattern="REPORT RequestId"
                )
                
                start_events = start_response.get('events', [])
                end_events = end_response.get('events', [])
                report_events = report_response.get('events', [])
                
                # Timing
                start_ts = max(e['timestamp'] for e in start_events) if start_events else 0
                end_ts = max(e['timestamp'] for e in end_events) if end_events else 0
                
                timing_metrics[name] = {
                    'start': start_ts,
                    'end': end_ts,
                    'duration': end_ts - start_ts if (start_ts and end_ts) else 0
                }
                
                # Parse REPORT for Lambda metrics
                if report_events:
                    latest_report = max(report_events, key=lambda e: e['timestamp'])
                    parsed = self.parse_report_log(latest_report['message'], name)
                    if parsed:
                        lambda_metrics[name] = parsed
                
            except Exception as e:
                timing_metrics[name] = {'start': 0, 'end': 0, 'duration': 0, 'error': str(e)}
        
        return timing_metrics, lambda_metrics
    
    def run_single_benchmark(self, mode: str, iteration: int, is_cold: bool) -> BenchmarkResult:
        """Run a single benchmark iteration with comprehensive metrics"""
        result = BenchmarkResult(
            session_id="",
            mode=mode,
            iteration=iteration,
            is_cold=is_cold
        )
        
        # Invoke workflow
        session_id, invocation_time = self.invoke_workflow()
        result.session_id = session_id
        result.invocation_time = invocation_time
        
        # Wait for completion
        completed = self.wait_for_completion(session_id, invocation_time)
        
        if not completed:
            print(f"    Warning: Workflow did not complete within timeout")
            return result
        
        # Wait for logs to be available
        time.sleep(3)
        
        # Collect metrics
        timing_metrics, lambda_metrics = self.collect_metrics(session_id, invocation_time)
        
        # Populate timing results
        for name, data in timing_metrics.items():
            if name == 'UserMention':
                result.user_mention_start = data['start']
                result.user_mention_end = data['end']
            elif name == 'FindUrl':
                result.find_url_start = data['start']
                result.find_url_end = data['end']
            elif name == 'ShortenUrl':
                result.shorten_url_start = data['start']
                result.shorten_url_end = data['end']
            elif name == 'CreatePost':
                result.create_post_start = data['start']
                result.create_post_end = data['end']
            elif name == 'Publish':
                result.publish_start = data['start']
                result.publish_end = data['end']
        
        # Populate Lambda metrics
        result.lambda_metrics = lambda_metrics
        
        # Calculate all derived metrics
        result.calculate_metrics()
        
        return result
    
    def calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of a list of values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = (len(sorted_values) - 1) * percentile / 100
        lower = int(index)
        upper = lower + 1
        if upper >= len(sorted_values):
            return sorted_values[-1]
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    
    def run_benchmark(self, mode: str, iterations: int = 10, cold_iterations: int = 2) -> ResearchSummary:
        """Run complete benchmark with research-grade metrics"""
        print(f"\n{'='*60}")
        print(f"Running {mode} benchmark ({iterations} iterations, {cold_iterations} cold)")
        print(f"{'='*60}")
        
        # Update mode
        self.update_mode(mode)
        
        results: List[BenchmarkResult] = []
        
        for i in range(iterations):
            is_cold = i < cold_iterations
            
            if is_cold:
                self.force_cold_start()
                print(f"  Iteration {i+1}/{iterations} (COLD)...", end=" ", flush=True)
            else:
                print(f"  Iteration {i+1}/{iterations} (WARM)...", end=" ", flush=True)
            
            result = self.run_single_benchmark(mode, i, is_cold)
            results.append(result)
            
            print(f"E2E: {result.e2e_latency_ms:.0f}ms, "
                  f"Memory: {result.total_memory_used_mb}MB, "
                  f"Cost: ${result.total_lambda_cost:.6f}")
            
            # Brief pause between iterations
            if i < iterations - 1:
                time.sleep(2)
        
        # Calculate summary statistics
        summary = self._calculate_summary(mode, results, iterations, cold_iterations)
        
        return summary
    
    def _calculate_summary(self, mode: str, results: List[BenchmarkResult], 
                           iterations: int, cold_iterations: int) -> ResearchSummary:
        """Calculate comprehensive summary statistics"""
        
        successful_results = [r for r in results if r.e2e_latency_ms > 0]
        cold_results = [r for r in successful_results if r.is_cold]
        warm_results = [r for r in successful_results if not r.is_cold]
        
        e2e_latencies = [r.e2e_latency_ms for r in successful_results]
        cold_latencies = [r.e2e_latency_ms for r in cold_results]
        warm_latencies = [r.e2e_latency_ms for r in warm_results]
        create_post_delays = [r.create_post_invocation_delay_ms for r in successful_results if r.create_post_invocation_delay_ms > 0]
        
        summary = ResearchSummary(
            workflow='text-processing',
            mode=mode,
            total_iterations=iterations,
            cold_iterations=cold_iterations,
            warm_iterations=iterations - cold_iterations,
            successful_iterations=len(successful_results),
            timestamp=datetime.now(timezone.utc).isoformat(),
            
            # Latency metrics
            e2e_latency_mean_ms=statistics.mean(e2e_latencies) if e2e_latencies else 0,
            e2e_latency_median_ms=statistics.median(e2e_latencies) if e2e_latencies else 0,
            e2e_latency_std_ms=statistics.stdev(e2e_latencies) if len(e2e_latencies) > 1 else 0,
            e2e_latency_min_ms=min(e2e_latencies) if e2e_latencies else 0,
            e2e_latency_max_ms=max(e2e_latencies) if e2e_latencies else 0,
            e2e_latency_p95_ms=self.calculate_percentile(e2e_latencies, 95),
            e2e_latency_p99_ms=self.calculate_percentile(e2e_latencies, 99),
            
            cold_e2e_mean_ms=statistics.mean(cold_latencies) if cold_latencies else 0,
            cold_e2e_std_ms=statistics.stdev(cold_latencies) if len(cold_latencies) > 1 else 0,
            warm_e2e_mean_ms=statistics.mean(warm_latencies) if warm_latencies else 0,
            warm_e2e_std_ms=statistics.stdev(warm_latencies) if len(warm_latencies) > 1 else 0,
            
            create_post_delay_mean_ms=statistics.mean(create_post_delays) if create_post_delays else 0,
            create_post_delay_std_ms=statistics.stdev(create_post_delays) if len(create_post_delays) > 1 else 0,
            
            # Memory metrics
            avg_total_memory_mb=statistics.mean([r.total_memory_used_mb for r in successful_results]) if successful_results else 0,
            avg_max_memory_mb=statistics.mean([r.max_memory_used_mb for r in successful_results]) if successful_results else 0,
            avg_aggregator_memory_mb=statistics.mean([r.aggregator_memory_mb for r in successful_results if r.aggregator_memory_mb > 0]) if successful_results else 0,
            avg_memory_efficiency=statistics.mean([r.avg_memory_efficiency for r in successful_results if r.avg_memory_efficiency > 0]) if successful_results else 0,
            
            # Cost metrics
            avg_cost_per_run=statistics.mean([r.total_lambda_cost for r in successful_results]) if successful_results else 0,
            total_cost=sum(r.total_lambda_cost for r in successful_results),
            avg_billed_duration_ms=statistics.mean([r.total_billed_duration_ms for r in successful_results]) if successful_results else 0,
            
            # Cold start metrics
            avg_cold_starts_per_run=statistics.mean([r.cold_start_count for r in successful_results]) if successful_results else 0,
            avg_init_duration_ms=statistics.mean([r.total_init_duration_ms for r in successful_results if r.total_init_duration_ms > 0]) if successful_results else 0,
            
            results=[asdict(r) for r in results]
        )
        
        # Calculate cost per 1M invocations
        if summary.avg_cost_per_run > 0:
            summary.cost_per_1m_invocations = summary.avg_cost_per_run * 1_000_000
        
        return summary


def generate_comparison_report(classic_summary: ResearchSummary, 
                               future_summary: ResearchSummary,
                               output_dir: Path):
    """Generate comprehensive comparison report"""
    
    # Calculate improvements
    def calc_improvement(classic_val, future_val):
        if classic_val > 0:
            return ((classic_val - future_val) / classic_val) * 100
        return 0
    
    report = f"""# Research Benchmark Report: Text-Processing Workflow

## Experiment Setup
- **Workflow**: Text-Processing (asymmetric parallel fan-out/fan-in)
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Region**: {REGION}
- **Total Iterations**: {classic_summary.total_iterations} per mode
- **Cold Iterations**: {classic_summary.cold_iterations} per mode
- **Warm Iterations**: {classic_summary.warm_iterations} per mode

## Workflow Structure
```
              ┌─────────────────────────────────────────────────┐
              │                                                 │
Input ──►[UserMention]──────────────────────────────────────────┤
              │                                                 ├──►[CreatePost]──►[Publish]
              │                                                 │
Input ──►[FindUrl]──►[ShortenUrl]───────────────────────────────┘
              │         (2 steps - SLOWER)
```

---

## 1. Latency Metrics (Primary Research Metrics)

| Metric | CLASSIC | FUTURE_BASED | Improvement |
|--------|---------|--------------|-------------|
| **E2E Mean** | {classic_summary.e2e_latency_mean_ms:.1f}ms | {future_summary.e2e_latency_mean_ms:.1f}ms | **{calc_improvement(classic_summary.e2e_latency_mean_ms, future_summary.e2e_latency_mean_ms):.1f}%** |
| E2E Median | {classic_summary.e2e_latency_median_ms:.1f}ms | {future_summary.e2e_latency_median_ms:.1f}ms | {calc_improvement(classic_summary.e2e_latency_median_ms, future_summary.e2e_latency_median_ms):.1f}% |
| E2E Std Dev | {classic_summary.e2e_latency_std_ms:.1f}ms | {future_summary.e2e_latency_std_ms:.1f}ms | - |
| E2E Min | {classic_summary.e2e_latency_min_ms:.1f}ms | {future_summary.e2e_latency_min_ms:.1f}ms | - |
| E2E Max | {classic_summary.e2e_latency_max_ms:.1f}ms | {future_summary.e2e_latency_max_ms:.1f}ms | - |
| **E2E P95** | {classic_summary.e2e_latency_p95_ms:.1f}ms | {future_summary.e2e_latency_p95_ms:.1f}ms | **{calc_improvement(classic_summary.e2e_latency_p95_ms, future_summary.e2e_latency_p95_ms):.1f}%** |
| E2E P99 | {classic_summary.e2e_latency_p99_ms:.1f}ms | {future_summary.e2e_latency_p99_ms:.1f}ms | {calc_improvement(classic_summary.e2e_latency_p99_ms, future_summary.e2e_latency_p99_ms):.1f}% |

### Cold vs Warm Start Comparison

| Metric | CLASSIC | FUTURE_BASED | Improvement |
|--------|---------|--------------|-------------|
| **Cold Start E2E** | {classic_summary.cold_e2e_mean_ms:.1f}ms | {future_summary.cold_e2e_mean_ms:.1f}ms | **{calc_improvement(classic_summary.cold_e2e_mean_ms, future_summary.cold_e2e_mean_ms):.1f}%** |
| Cold Start Std | {classic_summary.cold_e2e_std_ms:.1f}ms | {future_summary.cold_e2e_std_ms:.1f}ms | - |
| **Warm Start E2E** | {classic_summary.warm_e2e_mean_ms:.1f}ms | {future_summary.warm_e2e_mean_ms:.1f}ms | **{calc_improvement(classic_summary.warm_e2e_mean_ms, future_summary.warm_e2e_mean_ms):.1f}%** |
| Warm Start Std | {classic_summary.warm_e2e_std_ms:.1f}ms | {future_summary.warm_e2e_std_ms:.1f}ms | - |

### Fan-In Timing (CreatePost Invocation)

| Metric | CLASSIC | FUTURE_BASED | Improvement |
|--------|---------|--------------|-------------|
| CreatePost Delay Mean | {classic_summary.create_post_delay_mean_ms:.1f}ms | {future_summary.create_post_delay_mean_ms:.1f}ms | {calc_improvement(classic_summary.create_post_delay_mean_ms, future_summary.create_post_delay_mean_ms):.1f}% |
| CreatePost Delay Std | {classic_summary.create_post_delay_std_ms:.1f}ms | {future_summary.create_post_delay_std_ms:.1f}ms | - |

---

## 2. Memory Metrics

| Metric | CLASSIC | FUTURE_BASED | Overhead |
|--------|---------|--------------|----------|
| Avg Total Memory | {classic_summary.avg_total_memory_mb:.1f}MB | {future_summary.avg_total_memory_mb:.1f}MB | {((future_summary.avg_total_memory_mb - classic_summary.avg_total_memory_mb) / classic_summary.avg_total_memory_mb * 100) if classic_summary.avg_total_memory_mb > 0 else 0:.1f}% |
| Avg Max Memory | {classic_summary.avg_max_memory_mb:.1f}MB | {future_summary.avg_max_memory_mb:.1f}MB | - |
| Avg Aggregator Memory | {classic_summary.avg_aggregator_memory_mb:.1f}MB | {future_summary.avg_aggregator_memory_mb:.1f}MB | {((future_summary.avg_aggregator_memory_mb - classic_summary.avg_aggregator_memory_mb) / classic_summary.avg_aggregator_memory_mb * 100) if classic_summary.avg_aggregator_memory_mb > 0 else 0:.1f}% |
| Memory Efficiency | {classic_summary.avg_memory_efficiency*100:.1f}% | {future_summary.avg_memory_efficiency*100:.1f}% | - |

---

## 3. Cost Metrics

| Metric | CLASSIC | FUTURE_BASED | Savings |
|--------|---------|--------------|---------|
| **Avg Cost/Run** | ${classic_summary.avg_cost_per_run:.6f} | ${future_summary.avg_cost_per_run:.6f} | **{calc_improvement(classic_summary.avg_cost_per_run, future_summary.avg_cost_per_run):.1f}%** |
| Total Benchmark Cost | ${classic_summary.total_cost:.6f} | ${future_summary.total_cost:.6f} | - |
| **Cost/1M Invocations** | ${classic_summary.cost_per_1m_invocations:.2f} | ${future_summary.cost_per_1m_invocations:.2f} | **${classic_summary.cost_per_1m_invocations - future_summary.cost_per_1m_invocations:.2f}** |
| Avg Billed Duration | {classic_summary.avg_billed_duration_ms:.1f}ms | {future_summary.avg_billed_duration_ms:.1f}ms | - |

---

## 4. Cold Start Metrics

| Metric | CLASSIC | FUTURE_BASED | Notes |
|--------|---------|--------------|-------|
| Avg Cold Starts/Run | {classic_summary.avg_cold_starts_per_run:.1f} | {future_summary.avg_cold_starts_per_run:.1f} | Per workflow execution |
| Avg Init Duration | {classic_summary.avg_init_duration_ms:.1f}ms | {future_summary.avg_init_duration_ms:.1f}ms | Total init time per run |

---

## 5. Key Findings

### Performance Benefits
1. **Warm Start Improvement**: {calc_improvement(classic_summary.warm_e2e_mean_ms, future_summary.warm_e2e_mean_ms):.1f}% faster in Future-Based mode
2. **Cold Start Improvement**: {calc_improvement(classic_summary.cold_e2e_mean_ms, future_summary.cold_e2e_mean_ms):.1f}% faster due to parallel cold start overlap
3. **P95 Latency Improvement**: {calc_improvement(classic_summary.e2e_latency_p95_ms, future_summary.e2e_latency_p95_ms):.1f}% - consistent gains at tail

### Why Future-Based Is Faster
- **Early Aggregator Invocation**: CreatePost starts {classic_summary.create_post_delay_mean_ms - future_summary.create_post_delay_mean_ms:.0f}ms earlier on average
- **Parallel Cold Start**: In cold scenarios, CreatePost warms up while ShortenUrl is still executing
- **Async Fan-In**: Aggregator polls for missing inputs rather than waiting for synchronous completion

### Cost Analysis
- **Cost per Million Invocations**: Future-Based saves ${classic_summary.cost_per_1m_invocations - future_summary.cost_per_1m_invocations:.2f} per 1M invocations
- **Annual Savings at Scale**: At 10M invocations/month, saves ~${(classic_summary.cost_per_1m_invocations - future_summary.cost_per_1m_invocations) * 10 * 12:.2f}/year

---

## 6. Methodology

### Benchmark Protocol
1. Configure execution mode via Lambda environment variable (`EAGER=true/false`)
2. Force cold starts by updating Lambda configuration (triggers new container)
3. Invoke workflow via parallel Lambda invocations (UserMention + FindUrl)
4. Wait for completion by monitoring Publish CloudWatch logs
5. Parse REPORT logs to extract memory, duration, and init metrics
6. Calculate aggregate statistics with percentiles

### Measured Points
- **E2E Latency**: Client invocation timestamp → Publish END log timestamp
- **CreatePost Delay**: Client invocation → CreatePost START log timestamp
- **Memory**: Extracted from Lambda REPORT "Max Memory Used"
- **Cost**: Calculated from billed duration × memory × AWS pricing

---

*Generated by benchmark_research_metrics.py*
"""
    
    report_path = output_dir / "RESEARCH_BENCHMARK_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: {report_path}")
    return report


def generate_charts(classic_summary: ResearchSummary, 
                    future_summary: ResearchSummary,
                    output_dir: Path):
    """Generate research-quality charts"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available, skipping chart generation")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. E2E Latency Comparison (Cold vs Warm)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Bar chart
    ax1 = axes[0]
    x = np.arange(2)
    width = 0.35
    
    cold_vals = [classic_summary.cold_e2e_mean_ms, future_summary.cold_e2e_mean_ms]
    warm_vals = [classic_summary.warm_e2e_mean_ms, future_summary.warm_e2e_mean_ms]
    
    bars1 = ax1.bar(x - width/2, cold_vals, width, label='Cold Start', color='#2196F3')
    bars2 = ax1.bar(x + width/2, warm_vals, width, label='Warm Start', color='#4CAF50')
    
    ax1.set_ylabel('E2E Latency (ms)', fontsize=12)
    ax1.set_title('End-to-End Latency by Execution Mode', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['CLASSIC', 'FUTURE_BASED'])
    ax1.legend()
    ax1.bar_label(bars1, fmt='%.0f', padding=3)
    ax1.bar_label(bars2, fmt='%.0f', padding=3)
    
    # Right: Improvement chart
    ax2 = axes[1]
    
    def calc_improvement(c, f):
        return ((c - f) / c * 100) if c > 0 else 0
    
    improvements = [
        calc_improvement(classic_summary.cold_e2e_mean_ms, future_summary.cold_e2e_mean_ms),
        calc_improvement(classic_summary.warm_e2e_mean_ms, future_summary.warm_e2e_mean_ms),
        calc_improvement(classic_summary.e2e_latency_p95_ms, future_summary.e2e_latency_p95_ms),
    ]
    labels = ['Cold Start', 'Warm Start', 'P95 Latency']
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    bars = ax2.bar(labels, improvements, color=colors)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('Future-Based Improvement over Classic', fontsize=14, fontweight='bold')
    ax2.bar_label(bars, fmt='%.1f%%', padding=3)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Memory and Cost Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Memory
    ax1 = axes[0]
    x = np.arange(3)
    width = 0.35
    
    classic_mem = [classic_summary.avg_total_memory_mb, classic_summary.avg_max_memory_mb, classic_summary.avg_aggregator_memory_mb]
    future_mem = [future_summary.avg_total_memory_mb, future_summary.avg_max_memory_mb, future_summary.avg_aggregator_memory_mb]
    
    bars1 = ax1.bar(x - width/2, classic_mem, width, label='CLASSIC', color='#F44336')
    bars2 = ax1.bar(x + width/2, future_mem, width, label='FUTURE_BASED', color='#9C27B0')
    
    ax1.set_ylabel('Memory (MB)', fontsize=12)
    ax1.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Total', 'Max', 'Aggregator'])
    ax1.legend()
    ax1.bar_label(bars1, fmt='%.0f', padding=3)
    ax1.bar_label(bars2, fmt='%.0f', padding=3)
    
    # Right: Cost
    ax2 = axes[1]
    
    costs = [classic_summary.cost_per_1m_invocations, future_summary.cost_per_1m_invocations]
    labels = ['CLASSIC', 'FUTURE_BASED']
    colors = ['#F44336', '#9C27B0']
    
    bars = ax2.bar(labels, costs, color=colors)
    ax2.set_ylabel('Cost ($)', fontsize=12)
    ax2.set_title('Cost per 1M Invocations', fontsize=14, fontweight='bold')
    ax2.bar_label(bars, fmt='$%.2f', padding=3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_cost_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. CreatePost Timing (Key Fan-In Metric)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    delays = [classic_summary.create_post_delay_mean_ms, future_summary.create_post_delay_mean_ms]
    stds = [classic_summary.create_post_delay_std_ms, future_summary.create_post_delay_std_ms]
    labels = ['CLASSIC', 'FUTURE_BASED']
    colors = ['#F44336', '#9C27B0']
    
    bars = ax.bar(labels, delays, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Time from Workflow Start (ms)', fontsize=12)
    ax.set_title('CreatePost (Fan-In) Invocation Delay', fontsize=14, fontweight='bold')
    ax.bar_label(bars, fmt='%.0f ms', padding=3)
    
    # Add annotation
    improvement = calc_improvement(delays[0], delays[1])
    ax.annotate(f'{improvement:.1f}% faster', xy=(1, delays[1]), 
                xytext=(1.3, delays[0]), fontsize=12, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fanin_timing.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Charts saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Research-grade benchmark for text-processing workflow')
    parser.add_argument('--iterations', type=int, default=10, help='Total iterations per mode')
    parser.add_argument('--cold-iterations', type=int, default=2, help='Cold start iterations per mode')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("RESEARCH-GRADE BENCHMARK: Text-Processing Workflow")
    print("Comparing CLASSIC vs FUTURE_BASED Execution Modes")
    print("="*70)
    
    benchmark = ResearchBenchmark()
    
    # Run CLASSIC benchmark
    classic_summary = benchmark.run_benchmark("CLASSIC", args.iterations, args.cold_iterations)
    
    # Save CLASSIC results
    with open(output_dir / 'classic_research_results.json', 'w') as f:
        json.dump(asdict(classic_summary), f, indent=2, default=str)
    
    # Run FUTURE_BASED benchmark
    future_summary = benchmark.run_benchmark("FUTURE_BASED", args.iterations, args.cold_iterations)
    
    # Save FUTURE_BASED results
    with open(output_dir / 'future_research_results.json', 'w') as f:
        json.dump(asdict(future_summary), f, indent=2, default=str)
    
    # Generate comparison report
    report = generate_comparison_report(classic_summary, future_summary, output_dir)
    
    # Generate charts
    generate_charts(classic_summary, future_summary, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    
    def calc_improvement(c, f):
        return ((c - f) / c * 100) if c > 0 else 0
    
    print(f"""
KEY RESULTS:
┌────────────────────────────────────────────────────────────────┐
│ Metric              │ CLASSIC     │ FUTURE_BASED │ Improvement │
├────────────────────────────────────────────────────────────────┤
│ Cold Start E2E      │ {classic_summary.cold_e2e_mean_ms:>8.0f}ms │ {future_summary.cold_e2e_mean_ms:>10.0f}ms │ {calc_improvement(classic_summary.cold_e2e_mean_ms, future_summary.cold_e2e_mean_ms):>10.1f}% │
│ Warm Start E2E      │ {classic_summary.warm_e2e_mean_ms:>8.0f}ms │ {future_summary.warm_e2e_mean_ms:>10.0f}ms │ {calc_improvement(classic_summary.warm_e2e_mean_ms, future_summary.warm_e2e_mean_ms):>10.1f}% │
│ E2E P95             │ {classic_summary.e2e_latency_p95_ms:>8.0f}ms │ {future_summary.e2e_latency_p95_ms:>10.0f}ms │ {calc_improvement(classic_summary.e2e_latency_p95_ms, future_summary.e2e_latency_p95_ms):>10.1f}% │
├────────────────────────────────────────────────────────────────┤
│ Avg Total Memory    │ {classic_summary.avg_total_memory_mb:>8.0f}MB │ {future_summary.avg_total_memory_mb:>10.0f}MB │          - │
│ Cost/1M Invocations │ ${classic_summary.cost_per_1m_invocations:>7.2f}  │ ${future_summary.cost_per_1m_invocations:>9.2f}  │ ${classic_summary.cost_per_1m_invocations - future_summary.cost_per_1m_invocations:>8.2f} │
└────────────────────────────────────────────────────────────────┘

Output files saved to: {output_dir}/
""")


if __name__ == '__main__':
    main()
