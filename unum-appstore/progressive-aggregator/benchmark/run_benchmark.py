#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for Progressive-Aggregator Workflow

Compares three execution modes:
1. CLASSIC - Synchronous, blocking fan-in (last invoker executes)
2. EAGER - Polling-based blocking fan-in with LazyInput proxy
3. FUTURE_BASED - Async fan-in with parallel background polling

Metrics Collected:
- End-to-End Latency (invoke → completion)
- Per-Function Duration (CloudWatch REPORT logs)
- Billed Duration (for cost calculation)
- Cold Start Duration (Init Duration)
- Memory Usage (Max Memory Used)
- Fan-In Wait Time (from aggregator logs)
- DynamoDB Read/Write Operations
- Poll Count (for EAGER/FUTURE_BASED modes)

Usage:
    python run_benchmark.py --mode CLASSIC --iterations 10
    python run_benchmark.py --mode EAGER --iterations 10
    python run_benchmark.py --mode FUTURE_BASED --iterations 10
    python run_benchmark.py --all --iterations 10  # Run all modes
    python run_benchmark.py --analyze results/     # Analyze existing results
"""

import boto3
import json
import time
import argparse
import os
import re
import statistics
import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import subprocess
import sys


# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get('AWS_REGION', 'eu-central-1')
STACK_NAME = 'progressive-aggregator'

# Function names (will be detected from CloudFormation)
FUNCTIONS = {}

# DynamoDB table
DYNAMODB_TABLE = 'unum-intermediary-progressive'

# Log group prefix
LOG_GROUP_PREFIX = '/aws/lambda/'

# Workflow structure
WORKFLOW_STRUCTURE = {
    'FanOut': {'count': 1, 'fan_out_to': 'Source', 'fan_out_size': 5},
    'Source': {'count': 5, 'fan_in_to': 'Aggregator'},
    'Aggregator': {'count': 1, 'terminal': True, 'fan_in_size': 5},
}
TOTAL_INVOCATIONS = 7  # 1 + 5 + 1

# Pricing (eu-central-1, as of 2024)
PRICING = {
    'lambda_gb_second': 0.0000166667,
    'lambda_request': 0.0000002,
    'dynamodb_wcu': 0.00000125,  # per WCU
    'dynamodb_rcu': 0.00000025,  # per RCU
}

# Mode configurations
MODE_CONFIGS = {
    'CLASSIC': {
        'Eager': False,
        'UNUM_FUTURE_BASED': 'false',
    },
    'EAGER': {
        'Eager': True,
        'UNUM_FUTURE_BASED': 'false',
    },
    'FUTURE_BASED': {
        'Eager': True,
        'UNUM_FUTURE_BASED': 'true',
    },
}


# ============================================================
# Data Classes
# ============================================================

@dataclass
class LambdaMetrics:
    """Metrics extracted from a single Lambda REPORT log"""
    function_name: str
    request_id: str
    duration_ms: float
    billed_duration_ms: float
    memory_size_mb: int
    max_memory_used_mb: int
    init_duration_ms: Optional[float] = None  # Cold start only
    
    @property
    def is_cold_start(self) -> bool:
        return self.init_duration_ms is not None


@dataclass
class FanInMetrics:
    """Metrics for fan-in wait operations"""
    function_name: str
    session_id: str
    mode: str
    fan_in_size: int
    initially_ready: int = 0
    wait_duration_ms: float = 0.0
    poll_count: int = 0
    dynamo_reads: int = 0
    dynamo_writes: int = 0
    pre_resolved_count: int = 0  # From background polling
    strategy: str = ''


@dataclass
class WorkflowRun:
    """Complete metrics for a single workflow execution"""
    run_id: int
    session_id: str
    mode: str
    start_time: float
    end_time: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    
    # Run type
    run_type: Optional[str] = None  # 'cold', 'warm'
    
    # Per-function metrics
    lambda_metrics: List[LambdaMetrics] = field(default_factory=list)
    
    # Fan-in metrics
    fanin_metrics: List[FanInMetrics] = field(default_factory=list)
    
    # Aggregated metrics
    total_duration_ms: float = 0.0
    total_billed_duration_ms: float = 0.0
    cold_start_count: int = 0
    total_init_duration_ms: float = 0.0
    max_memory_used_mb: int = 0
    
    # DynamoDB operations
    dynamo_reads: int = 0
    dynamo_writes: int = 0
    
    # Error tracking
    error: Optional[str] = None
    
    def compute_aggregates(self):
        """Compute aggregate metrics from per-function data"""
        if self.lambda_metrics:
            self.total_duration_ms = sum(m.duration_ms for m in self.lambda_metrics)
            self.total_billed_duration_ms = sum(m.billed_duration_ms for m in self.lambda_metrics)
            self.cold_start_count = sum(1 for m in self.lambda_metrics if m.is_cold_start)
            self.total_init_duration_ms = sum(m.init_duration_ms or 0 for m in self.lambda_metrics)
            self.max_memory_used_mb = max((m.max_memory_used_mb for m in self.lambda_metrics), default=0)
        
        if self.fanin_metrics:
            self.dynamo_reads = sum(m.dynamo_reads for m in self.fanin_metrics)
            self.dynamo_writes = sum(m.dynamo_writes for m in self.fanin_metrics)


@dataclass
class BenchmarkSummary:
    """Statistical summary for a benchmark run"""
    mode: str
    iterations: int
    successful_runs: int
    failed_runs: int
    timestamp: str
    workflow: str = 'progressive-aggregator'
    
    # E2E Latency
    e2e_latency_mean_ms: float = 0.0
    e2e_latency_median_ms: float = 0.0
    e2e_latency_std_ms: float = 0.0
    e2e_latency_min_ms: float = 0.0
    e2e_latency_max_ms: float = 0.0
    e2e_latency_p95_ms: float = 0.0
    e2e_latency_p99_ms: float = 0.0
    
    # Total Lambda Duration
    total_duration_mean_ms: float = 0.0
    total_duration_median_ms: float = 0.0
    
    # Billed Duration
    billed_duration_mean_ms: float = 0.0
    billed_duration_total_ms: float = 0.0
    
    # Cold Starts
    cold_start_rate: float = 0.0
    avg_init_duration_ms: float = 0.0
    
    # Fan-In Wait Time
    fanin_wait_mean_ms: float = 0.0
    fanin_wait_max_ms: float = 0.0
    avg_poll_count: float = 0.0
    avg_pre_resolved: float = 0.0  # Future-based optimization metric
    
    # DynamoDB
    avg_dynamo_reads: float = 0.0
    avg_dynamo_writes: float = 0.0
    total_dynamo_reads: int = 0
    total_dynamo_writes: int = 0
    
    # Cost estimates
    lambda_compute_cost: float = 0.0
    lambda_request_cost: float = 0.0
    dynamodb_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_run: float = 0.0


# ============================================================
# Benchmark Runner
# ============================================================

class ProgressiveAggregatorBenchmark:
    """Benchmark runner for progressive-aggregator workflow"""
    
    def __init__(self, region: str = REGION):
        self.region = region
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.logs_client = boto3.client('logs', region_name=region)
        self.cf_client = boto3.client('cloudformation', region_name=region)
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        
        # Discover function ARNs from CloudFormation
        self._discover_functions()
        
        self.table = self.dynamodb.Table(DYNAMODB_TABLE)
    
    def _discover_functions(self):
        """Discover Lambda function ARNs from CloudFormation stack"""
        global FUNCTIONS
        try:
            response = self.cf_client.describe_stack_resources(StackName=STACK_NAME)
            for resource in response['StackResources']:
                if resource['ResourceType'] == 'AWS::Lambda::Function':
                    logical_id = resource['LogicalResourceId']
                    physical_id = resource['PhysicalResourceId']
                    # Map logical name to function name
                    if 'FanOut' in logical_id:
                        FUNCTIONS['FanOut'] = physical_id
                    elif 'Source' in logical_id:
                        FUNCTIONS['Source'] = physical_id
                    elif 'Aggregator' in logical_id:
                        FUNCTIONS['Aggregator'] = physical_id
            print(f"  Discovered functions: {list(FUNCTIONS.keys())}")
        except Exception as e:
            print(f"  Warning: Could not discover functions from stack: {e}")
            # Use default naming pattern
            FUNCTIONS['FanOut'] = f'{STACK_NAME}-FanOutFunction'
            FUNCTIONS['Source'] = f'{STACK_NAME}-SourceFunction'
            FUNCTIONS['Aggregator'] = f'{STACK_NAME}-AggregatorFunction'
    
    # --------------------------------------------------------
    # Cold Start Management
    # --------------------------------------------------------
    
    def force_cold_starts(self):
        """Force all Lambda functions to cold start"""
        print("    Forcing cold starts on all functions...")
        timestamp = str(int(time.time()))
        
        for func_name, func_arn in FUNCTIONS.items():
            try:
                response = self.lambda_client.get_function_configuration(
                    FunctionName=func_arn
                )
                current_env = response.get('Environment', {}).get('Variables', {})
                current_env['FORCE_COLD'] = timestamp
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={'Variables': current_env}
                )
                print(f"      {func_name}: updated")
            except Exception as e:
                print(f"      {func_name}: failed ({e})")
        
        print("    Waiting for Lambda updates to propagate...")
        time.sleep(5)
        
        for func_name, func_arn in FUNCTIONS.items():
            self._wait_for_function_active(func_arn)
        
        print("    All functions ready with fresh containers")
    
    def _wait_for_function_active(self, function_arn: str, timeout: int = 60):
        """Wait for Lambda function to be in Active state"""
        start = time.time()
        while (time.time() - start) < timeout:
            try:
                response = self.lambda_client.get_function(FunctionName=function_arn)
                state = response['Configuration']['State']
                last_update = response['Configuration'].get('LastUpdateStatus', 'Successful')
                if state == 'Active' and last_update == 'Successful':
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False
    
    # --------------------------------------------------------
    # Workflow Invocation
    # --------------------------------------------------------
    
    def invoke_workflow(self, session_id: str = None) -> Tuple[str, float, str]:
        """
        Invoke the progressive-aggregator workflow.
        Returns (request_id, invoke_timestamp, session_id)
        """
        if session_id is None:
            session_id = f"bench-{int(time.time() * 1000)}"
        
        payload = {
            "Data": {
                "Source": "http",
                "Value": {}
            },
            "Session": session_id
        }
        
        start_time = time.time()
        response = self.lambda_client.invoke(
            FunctionName=FUNCTIONS['FanOut'],
            InvocationType='Event',  # Async invocation
            Payload=json.dumps(payload)
        )
        
        request_id = response.get('ResponseMetadata', {}).get('RequestId', '')
        return request_id, start_time, session_id
    
    def wait_for_completion(self, session_id: str, start_time: float,
                           timeout_seconds: int = 120) -> Tuple[bool, float]:
        """
        Wait for workflow completion by checking Aggregator logs.
        Returns (success, end_time)
        
        NOTE: We need to be careful about CloudWatch log propagation delay.
        Logs can take several seconds to appear in CloudWatch after being written.
        We track the actual log timestamp to ensure we're seeing the right run.
        """
        log_group = f"{LOG_GROUP_PREFIX}{FUNCTIONS['Aggregator']}"
        start_ms = int(start_time * 1000)
        
        # Track if we've seen any logs from this run (to detect propagation delay)
        seen_start_log = False
        
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                # First check for START log to confirm run began
                if not seen_start_log:
                    start_response = self.logs_client.filter_log_events(
                        logGroupName=log_group,
                        startTime=start_ms,
                        filterPattern='START RequestId'
                    )
                    if start_response.get('events'):
                        for event in start_response['events']:
                            # Only count if timestamp is after our start time
                            if event.get('timestamp', 0) >= start_ms:
                                seen_start_log = True
                                break
                
                # Look for completion
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=start_ms,
                    filterPattern='"[AGGREGATOR] COMPLETED"'
                )
                
                if response.get('events'):
                    for event in response['events']:
                        log_timestamp = event.get('timestamp', 0)
                        # Only accept completion logs that are AFTER our start time
                        # with a small buffer for clock skew
                        if log_timestamp >= start_ms - 1000:
                            end_time = time.time()
                            return True, end_time
            except Exception as e:
                pass
            
            time.sleep(0.5)
        
        return False, time.time()
    
    # --------------------------------------------------------
    # Metrics Collection
    # --------------------------------------------------------
    
    def collect_lambda_metrics(self, session_id: str, start_time: float,
                               end_time: float) -> List[LambdaMetrics]:
        """Collect Lambda metrics from CloudWatch REPORT logs"""
        metrics = []
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + 60000  # Add 1 minute buffer
        
        for func_name, func_arn in FUNCTIONS.items():
            log_group = f"{LOG_GROUP_PREFIX}{func_arn}"
            
            try:
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=start_ms,
                    endTime=end_ms,
                    filterPattern='REPORT RequestId'
                )
                
                for event in response.get('events', []):
                    metric = self._parse_report_log(func_name, event['message'])
                    if metric:
                        metrics.append(metric)
            except Exception as e:
                print(f"      Warning: Could not get logs for {func_name}: {e}")
        
        return metrics
    
    def _parse_report_log(self, func_name: str, message: str) -> Optional[LambdaMetrics]:
        """Parse a REPORT log line into LambdaMetrics"""
        try:
            # Extract RequestId
            req_match = re.search(r'RequestId:\s*([a-f0-9-]+)', message)
            request_id = req_match.group(1) if req_match else ''
            
            # Extract Duration
            dur_match = re.search(r'Duration:\s*([\d.]+)\s*ms', message)
            duration = float(dur_match.group(1)) if dur_match else 0.0
            
            # Extract Billed Duration
            billed_match = re.search(r'Billed Duration:\s*([\d.]+)\s*ms', message)
            billed = float(billed_match.group(1)) if billed_match else duration
            
            # Extract Memory Size
            mem_size_match = re.search(r'Memory Size:\s*(\d+)\s*MB', message)
            mem_size = int(mem_size_match.group(1)) if mem_size_match else 128
            
            # Extract Max Memory Used
            max_mem_match = re.search(r'Max Memory Used:\s*(\d+)\s*MB', message)
            max_mem = int(max_mem_match.group(1)) if max_mem_match else 0
            
            # Extract Init Duration (cold start)
            init_match = re.search(r'Init Duration:\s*([\d.]+)\s*ms', message)
            init_duration = float(init_match.group(1)) if init_match else None
            
            return LambdaMetrics(
                function_name=func_name,
                request_id=request_id,
                duration_ms=duration,
                billed_duration_ms=billed,
                memory_size_mb=mem_size,
                max_memory_used_mb=max_mem,
                init_duration_ms=init_duration
            )
        except Exception as e:
            return None
    
    def collect_fanin_metrics(self, session_id: str, start_time: float,
                              end_time: float, mode: str) -> List[FanInMetrics]:
        """Collect fan-in metrics from Aggregator logs"""
        metrics = []
        log_group = f"{LOG_GROUP_PREFIX}{FUNCTIONS['Aggregator']}"
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + 60000
        
        fanin = FanInMetrics(
            function_name='Aggregator',
            session_id=session_id,
            mode=mode,
            fan_in_size=5
        )
        
        instant_count = 0  # Track INSTANT resolves separately
        
        try:
            response = self.logs_client.filter_log_events(
                logGroupName=log_group,
                startTime=start_ms,
                endTime=end_ms
            )
            
            for event in response.get('events', []):
                msg = event['message']
                
                # Pre-resolved count from summary: "Inputs that were pre-resolved (background polling): X/Y"
                pre_resolved_match = re.search(r'pre-resolved.*?(\d+)/(\d+)', msg, re.IGNORECASE)
                if pre_resolved_match:
                    fanin.pre_resolved_count = int(pre_resolved_match.group(1))
                
                # Count INSTANT resolves: "⚡ INSTANT! Got value in 0ms"
                # NOTE: Only count for FUTURE_BASED mode - in CLASSIC, fast reads are just fast DynamoDB, not background polling
                if 'INSTANT' in msg and mode == 'FUTURE_BASED':
                    instant_count += 1
                
                # Wait duration: "📥 Received after waiting 18ms"
                wait_match = re.search(r'Received after waiting\s*(\d+)ms', msg)
                if wait_match:
                    fanin.wait_duration_ms += float(wait_match.group(1))
                
                # Poll count from debug logs: "Total resolved so far: X/Y"
                poll_match = re.search(r'Total resolved.*?(\d+)/(\d+)', msg)
                if poll_match:
                    fanin.poll_count = max(fanin.poll_count, int(poll_match.group(1)))
                
                # DynamoDB operations - count actual operations
                if 'GetItem' in msg or 'get_item' in msg:
                    fanin.dynamo_reads += 1
                if 'PutItem' in msg or 'put_item' in msg:
                    fanin.dynamo_writes += 1
            
            # Use instant_count as pre_resolved if summary not found (only for FUTURE_BASED)
            if fanin.pre_resolved_count == 0 and instant_count > 0 and mode == 'FUTURE_BASED':
                fanin.pre_resolved_count = instant_count
        
        except Exception as e:
            print(f"      Warning: Could not collect fan-in metrics: {e}")
        
        if mode == 'CLASSIC':
            fanin.strategy = 'last_invoker'
        elif mode == 'EAGER':
            fanin.strategy = 'LazyInput'
        else:
            fanin.strategy = 'UnumFuture_BackgroundPolling'
        
        metrics.append(fanin)
        return metrics
    
    def get_accurate_e2e_from_logs(self, start_time: float, end_time: float) -> Optional[float]:
        """
        Calculate accurate E2E latency from CloudWatch log timestamps.
        Returns E2E in milliseconds, or None if unable to determine.
        
        E2E = (Aggregator REPORT timestamp) - (FanOut START timestamp)
        """
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + 120000  # 2 minute buffer
        
        fanout_start_ts = None
        aggregator_end_ts = None
        
        try:
            # Get FanOut START timestamp
            fanout_log_group = f"{LOG_GROUP_PREFIX}{FUNCTIONS['FanOut']}"
            response = self.logs_client.filter_log_events(
                logGroupName=fanout_log_group,
                startTime=start_ms,
                endTime=end_ms,
                filterPattern='START RequestId'
            )
            for event in response.get('events', []):
                ts = event.get('timestamp', 0)
                if ts >= start_ms:
                    fanout_start_ts = ts
                    break
            
            # Get Aggregator REPORT timestamp (marks completion)
            agg_log_group = f"{LOG_GROUP_PREFIX}{FUNCTIONS['Aggregator']}"
            response = self.logs_client.filter_log_events(
                logGroupName=agg_log_group,
                startTime=start_ms,
                endTime=end_ms,
                filterPattern='REPORT RequestId'
            )
            for event in response.get('events', []):
                ts = event.get('timestamp', 0)
                if ts >= start_ms:
                    aggregator_end_ts = ts  # Take the latest one
            
            if fanout_start_ts and aggregator_end_ts:
                return aggregator_end_ts - fanout_start_ts
                
        except Exception as e:
            print(f"      Warning: Could not get accurate E2E: {e}")
        
        return None
    
    # --------------------------------------------------------
    # Single Run
    # --------------------------------------------------------
    
    def run_single(self, run_id: int, mode: str, 
                   force_cold: bool = False) -> WorkflowRun:
        """Execute a single benchmark run"""
        session_id = f"bench-{mode}-{run_id}-{int(time.time() * 1000)}"
        
        run = WorkflowRun(
            run_id=run_id,
            session_id=session_id,
            mode=mode,
            start_time=0,
            run_type='cold' if force_cold else 'warm'
        )
        
        try:
            # Force cold starts if requested
            if force_cold:
                self.force_cold_starts()
            
            # Invoke workflow
            request_id, start_time, session_id = self.invoke_workflow(session_id)
            run.start_time = start_time
            run.session_id = session_id
            
            # Wait for completion
            success, end_time = self.wait_for_completion(session_id, start_time)
            
            if success:
                run.end_time = end_time
                
                # Give CloudWatch time to fully ingest all logs
                time.sleep(5)
                
                # Get accurate E2E from CloudWatch log timestamps
                accurate_e2e = self.get_accurate_e2e_from_logs(start_time, end_time)
                if accurate_e2e:
                    run.e2e_latency_ms = accurate_e2e
                else:
                    # Fallback to wall-clock time
                    run.e2e_latency_ms = (end_time - start_time) * 1000
                
                # Collect metrics
                run.lambda_metrics = self.collect_lambda_metrics(
                    session_id, start_time, end_time
                )
                run.fanin_metrics = self.collect_fanin_metrics(
                    session_id, start_time, end_time, mode
                )
                
                run.compute_aggregates()
            else:
                run.error = "Timeout waiting for workflow completion"
        
        except Exception as e:
            run.error = str(e)
        
        return run
    
    # --------------------------------------------------------
    # Full Benchmark
    # --------------------------------------------------------
    
    def run_benchmark(self, mode: str, iterations: int = 10,
                      warmup_runs: int = 2,
                      cold_iterations: int = 0) -> Tuple[List[WorkflowRun], BenchmarkSummary]:
        """Run complete benchmark for a mode"""
        
        # Skip warmup if only doing cold starts (warmup defeats the purpose)
        if iterations == 0 and cold_iterations > 0:
            warmup_runs = 0
        
        print(f"\n{'='*60}")
        print(f"  BENCHMARK: {mode} Mode")
        print(f"  Iterations: {iterations} (+ {warmup_runs} warmup)")
        print(f"  Cold start runs: {cold_iterations}")
        print(f"{'='*60}")
        
        runs = []
        
        # Warmup runs (not counted)
        if warmup_runs > 0:
            print(f"\n  Warmup runs ({warmup_runs})...")
            for i in range(warmup_runs):
                print(f"    Warmup {i+1}/{warmup_runs}...", end=' ')
                run = self.run_single(i, mode, force_cold=False)
                status = "OK" if not run.error else f"FAILED: {run.error}"
                print(status)
                time.sleep(1)
        
        # Cold start runs
        if cold_iterations > 0:
            print(f"\n  Cold start runs ({cold_iterations})...")
            for i in range(cold_iterations):
                print(f"    Cold {i+1}/{cold_iterations}...", end=' ')
                run = self.run_single(i, mode, force_cold=True)
                if run.error:
                    print(f"FAILED: {run.error}")
                else:
                    print(f"E2E: {run.e2e_latency_ms:.0f}ms, "
                          f"Cold starts: {run.cold_start_count}")
                runs.append(run)
                # Wait longer between cold runs to ensure CloudWatch logs fully propagate
                # before the next run starts (prevents picking up stale completion logs)
                time.sleep(10)
        
        # Warm runs
        print(f"\n  Warm runs ({iterations})...")
        for i in range(iterations):
            print(f"    Run {i+1}/{iterations}...", end=' ')
            run = self.run_single(len(runs), mode, force_cold=False)
            if run.error:
                print(f"FAILED: {run.error}")
            else:
                print(f"E2E: {run.e2e_latency_ms:.0f}ms")
            runs.append(run)
            time.sleep(1)
        
        # Compute summary
        summary = self._compute_summary(mode, runs)
        
        return runs, summary
    
    def _compute_summary(self, mode: str, runs: List[WorkflowRun]) -> BenchmarkSummary:
        """Compute statistical summary from runs"""
        successful = [r for r in runs if r.error is None]
        failed = [r for r in runs if r.error is not None]
        
        summary = BenchmarkSummary(
            mode=mode,
            iterations=len(runs),
            successful_runs=len(successful),
            failed_runs=len(failed),
            timestamp=datetime.datetime.now().isoformat()
        )
        
        if not successful:
            return summary
        
        # E2E Latency
        e2e_latencies = [r.e2e_latency_ms for r in successful if r.e2e_latency_ms]
        if e2e_latencies:
            summary.e2e_latency_mean_ms = statistics.mean(e2e_latencies)
            summary.e2e_latency_median_ms = statistics.median(e2e_latencies)
            summary.e2e_latency_std_ms = statistics.stdev(e2e_latencies) if len(e2e_latencies) > 1 else 0
            summary.e2e_latency_min_ms = min(e2e_latencies)
            summary.e2e_latency_max_ms = max(e2e_latencies)
            sorted_e2e = sorted(e2e_latencies)
            p95_idx = int(len(sorted_e2e) * 0.95)
            p99_idx = int(len(sorted_e2e) * 0.99)
            summary.e2e_latency_p95_ms = sorted_e2e[min(p95_idx, len(sorted_e2e)-1)]
            summary.e2e_latency_p99_ms = sorted_e2e[min(p99_idx, len(sorted_e2e)-1)]
        
        # Total Duration
        total_durations = [r.total_duration_ms for r in successful]
        if total_durations:
            summary.total_duration_mean_ms = statistics.mean(total_durations)
            summary.total_duration_median_ms = statistics.median(total_durations)
        
        # Billed Duration
        billed_durations = [r.total_billed_duration_ms for r in successful]
        if billed_durations:
            summary.billed_duration_mean_ms = statistics.mean(billed_durations)
            summary.billed_duration_total_ms = sum(billed_durations)
        
        # Cold Starts
        cold_starts = sum(r.cold_start_count for r in successful)
        total_invocations = len(successful) * TOTAL_INVOCATIONS
        summary.cold_start_rate = cold_starts / total_invocations if total_invocations > 0 else 0
        
        init_durations = [r.total_init_duration_ms for r in successful if r.total_init_duration_ms > 0]
        if init_durations:
            summary.avg_init_duration_ms = statistics.mean(init_durations)
        
        # Fan-In Metrics
        fanin_waits = []
        poll_counts = []
        pre_resolved_counts = []
        for r in successful:
            for fm in r.fanin_metrics:
                if fm.wait_duration_ms > 0:
                    fanin_waits.append(fm.wait_duration_ms)
                poll_counts.append(fm.poll_count)
                pre_resolved_counts.append(fm.pre_resolved_count)
        
        if fanin_waits:
            summary.fanin_wait_mean_ms = statistics.mean(fanin_waits)
            summary.fanin_wait_max_ms = max(fanin_waits)
        if poll_counts:
            summary.avg_poll_count = statistics.mean(poll_counts)
        if pre_resolved_counts:
            summary.avg_pre_resolved = statistics.mean(pre_resolved_counts)
        
        # DynamoDB
        summary.total_dynamo_reads = sum(r.dynamo_reads for r in successful)
        summary.total_dynamo_writes = sum(r.dynamo_writes for r in successful)
        summary.avg_dynamo_reads = summary.total_dynamo_reads / len(successful)
        summary.avg_dynamo_writes = summary.total_dynamo_writes / len(successful)
        
        # Cost Estimation
        # Lambda compute: GB-seconds * price
        total_gb_seconds = sum(
            r.total_billed_duration_ms / 1000 * 0.128  # 128MB = 0.128GB
            for r in successful
        )
        summary.lambda_compute_cost = total_gb_seconds * PRICING['lambda_gb_second']
        
        # Lambda requests
        summary.lambda_request_cost = len(successful) * TOTAL_INVOCATIONS * PRICING['lambda_request']
        
        # DynamoDB
        summary.dynamodb_cost = (
            summary.total_dynamo_reads * PRICING['dynamodb_rcu'] +
            summary.total_dynamo_writes * PRICING['dynamodb_wcu']
        )
        
        summary.total_cost = (
            summary.lambda_compute_cost +
            summary.lambda_request_cost +
            summary.dynamodb_cost
        )
        summary.cost_per_run = summary.total_cost / len(successful) if successful else 0
        
        return summary


# ============================================================
# Mode Switching
# ============================================================

def get_current_deployed_mode(region: str = REGION) -> str:
    """Check Lambda env vars to determine currently deployed mode"""
    import boto3
    lambda_client = boto3.client('lambda', region_name=region)
    
    # Use direct function name pattern if FUNCTIONS not yet populated
    aggregator_func = FUNCTIONS.get('Aggregator') or f'{STACK_NAME}-AggregatorFunction-z0rG4dhEmMGR'
    
    try:
        response = lambda_client.get_function_configuration(
            FunctionName=aggregator_func
        )
        env_vars = response.get('Environment', {}).get('Variables', {})
        
        eager = env_vars.get('EAGER', 'false').lower() == 'true'
        future_based = env_vars.get('UNUM_FUTURE_BASED', 'false').lower() == 'true'
        
        if future_based:
            return 'FUTURE_BASED'
        elif eager:
            return 'EAGER'
        else:
            return 'CLASSIC'
    except Exception as e:
        print(f"  Warning: Could not detect current mode: {e}")
        return None


def switch_mode(mode: str, project_dir: Path, region: str = REGION):
    """Switch the workflow to specified mode by updating unum-template.yaml"""
    
    # Check if already in target mode
    current_mode = get_current_deployed_mode(region)
    if current_mode == mode:
        print(f"  Already deployed in {mode} mode - skipping rebuild")
        return 'SKIP'
    
    print(f"  Switching from {current_mode} to {mode} mode...")
    template_file = project_dir / 'unum-template.yaml'
    
    if not template_file.exists():
        print(f"Error: {template_file} not found")
        return False
    
    content = template_file.read_text()
    config = MODE_CONFIGS[mode]
    
    # Update Eager setting in Globals
    if config['Eager']:
        content = re.sub(r'Eager:\s*(True|False)', 'Eager: True', content)
    else:
        content = re.sub(r'Eager:\s*(True|False)', 'Eager: False', content)
    
    # Update UNUM_FUTURE_BASED in Environment sections
    future_val = config['UNUM_FUTURE_BASED']
    content = re.sub(
        r'UNUM_FUTURE_BASED:\s*["\']?(true|false)["\']?',
        f'UNUM_FUTURE_BASED: "{future_val}"',
        content
    )
    
    template_file.write_text(content)
    print(f"  Updated unum-template.yaml for {mode} mode")
    
    return True


def rebuild_and_deploy(project_dir: Path, unum_cli_path: Path):
    """Rebuild and redeploy the stack"""
    print("  Rebuilding and deploying...")
    
    os.chdir(project_dir)
    
    # Run unum-cli template, build, deploy
    commands = [
        [sys.executable, str(unum_cli_path), 'template'],
        [sys.executable, str(unum_cli_path), 'build'],
        [sys.executable, str(unum_cli_path), 'deploy'],
    ]
    
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error running {' '.join(cmd)}:")
            print(result.stderr)
            return False
    
    print("  Deploy successful!")
    return True


# ============================================================
# Results Analysis and Export
# ============================================================

def save_results(runs: List[WorkflowRun], summary: BenchmarkSummary, 
                 output_dir: Path):
    """Save benchmark results to JSON files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save runs
    runs_file = output_dir / f'benchmark_{summary.mode}_{timestamp}_runs.json'
    runs_data = [asdict(r) for r in runs]
    with open(runs_file, 'w') as f:
        json.dump(runs_data, f, indent=2, default=str)
    print(f"  Saved runs to {runs_file}")
    
    # Save summary
    summary_file = output_dir / f'benchmark_{summary.mode}_{timestamp}_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"  Saved summary to {summary_file}")
    
    return runs_file, summary_file


def print_summary(summary: BenchmarkSummary):
    """Print formatted summary"""
    print(f"\n{'='*60}")
    print(f"  RESULTS: {summary.mode} Mode")
    print(f"{'='*60}")
    print(f"  Runs: {summary.successful_runs} successful, {summary.failed_runs} failed")
    print()
    print(f"  E2E Latency:")
    print(f"    Mean:   {summary.e2e_latency_mean_ms:.1f} ms")
    print(f"    Median: {summary.e2e_latency_median_ms:.1f} ms")
    print(f"    Std:    {summary.e2e_latency_std_ms:.1f} ms")
    print(f"    Min:    {summary.e2e_latency_min_ms:.1f} ms")
    print(f"    Max:    {summary.e2e_latency_max_ms:.1f} ms")
    print(f"    P95:    {summary.e2e_latency_p95_ms:.1f} ms")
    print()
    print(f"  Lambda Metrics:")
    print(f"    Total Duration (mean):  {summary.total_duration_mean_ms:.1f} ms")
    print(f"    Billed Duration (mean): {summary.billed_duration_mean_ms:.1f} ms")
    print(f"    Cold Start Rate: {summary.cold_start_rate*100:.1f}%")
    print()
    print(f"  Fan-In Metrics:")
    print(f"    Wait Time (mean): {summary.fanin_wait_mean_ms:.1f} ms")
    print(f"    Wait Time (max):  {summary.fanin_wait_max_ms:.1f} ms")
    print(f"    Avg Poll Count:   {summary.avg_poll_count:.1f}")
    print(f"    Avg Pre-Resolved: {summary.avg_pre_resolved:.1f}/5 (background polling)")
    print()
    print(f"  DynamoDB:")
    print(f"    Avg Reads/Run:  {summary.avg_dynamo_reads:.1f}")
    print(f"    Avg Writes/Run: {summary.avg_dynamo_writes:.1f}")
    print()
    print(f"  Cost Estimates:")
    print(f"    Lambda Compute: ${summary.lambda_compute_cost:.6f}")
    print(f"    Lambda Request: ${summary.lambda_request_cost:.6f}")
    print(f"    DynamoDB:       ${summary.dynamodb_cost:.6f}")
    print(f"    Total:          ${summary.total_cost:.6f}")
    print(f"    Per Run:        ${summary.cost_per_run:.6f}")
    print(f"{'='*60}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Progressive-Aggregator Benchmark Runner'
    )
    parser.add_argument('--mode', choices=['CLASSIC', 'EAGER', 'FUTURE_BASED'],
                        help='Execution mode to benchmark')
    parser.add_argument('--all', action='store_true',
                        help='Run benchmark for all modes')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of warm iterations per mode')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup runs')
    parser.add_argument('--cold', type=int, default=0,
                        help='Number of cold start runs')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--no-deploy', action='store_true',
                        help='Skip mode switching and deployment')
    parser.add_argument('--region', type=str, default=REGION,
                        help='AWS region')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent
    unum_cli_path = project_dir.parent.parent / 'unum' / 'unum-cli' / 'unum-cli.py'
    output_dir = script_dir / args.output
    
    print(f"\nProgressive-Aggregator Benchmark")
    print(f"  Region: {args.region}")
    print(f"  Output: {output_dir}")
    
    modes = ['CLASSIC', 'EAGER', 'FUTURE_BASED'] if args.all else [args.mode]
    
    if not modes[0]:
        parser.error("Must specify --mode or --all")
    
    all_summaries = {}
    
    for mode in modes:
        print(f"\n{'#'*60}")
        print(f"  MODE: {mode}")
        print(f"{'#'*60}")
        
        # Switch mode and deploy
        if not args.no_deploy:
            result = switch_mode(mode, project_dir, args.region)
            if result == 'SKIP':
                pass  # Already in correct mode, continue to benchmark
            elif not result:
                continue
            else:
                if not rebuild_and_deploy(project_dir, unum_cli_path):
                    continue
                time.sleep(10)  # Wait for deployment
        
        # Run benchmark
        benchmark = ProgressiveAggregatorBenchmark(region=args.region)
        runs, summary = benchmark.run_benchmark(
            mode=mode,
            iterations=args.iterations,
            warmup_runs=args.warmup,
            cold_iterations=args.cold
        )
        
        # Save results
        save_results(runs, summary, output_dir)
        
        # Print summary
        print_summary(summary)
        
        all_summaries[mode] = summary
    
    # Print comparison if multiple modes
    if len(all_summaries) > 1:
        print_comparison(all_summaries)


def print_comparison(summaries: Dict[str, BenchmarkSummary]):
    """Print comparison table across modes"""
    print(f"\n{'='*80}")
    print(f"  COMPARISON ACROSS MODES")
    print(f"{'='*80}")
    
    # Header
    modes = list(summaries.keys())
    print(f"{'Metric':<30}", end='')
    for mode in modes:
        print(f"{mode:>16}", end='')
    print()
    print("-" * (30 + 16 * len(modes)))
    
    # Metrics
    metrics = [
        ('E2E Latency (mean)', 'e2e_latency_mean_ms', 'ms'),
        ('E2E Latency (median)', 'e2e_latency_median_ms', 'ms'),
        ('E2E Latency (P95)', 'e2e_latency_p95_ms', 'ms'),
        ('Total Duration (mean)', 'total_duration_mean_ms', 'ms'),
        ('Fan-In Wait (mean)', 'fanin_wait_mean_ms', 'ms'),
        ('Fan-In Wait (max)', 'fanin_wait_max_ms', 'ms'),
        ('Avg Poll Count', 'avg_poll_count', ''),
        ('Avg Pre-Resolved', 'avg_pre_resolved', '/5'),
        ('DynamoDB Reads/Run', 'avg_dynamo_reads', ''),
        ('Cost Per Run', 'cost_per_run', '$'),
    ]
    
    for label, attr, unit in metrics:
        print(f"{label:<30}", end='')
        for mode in modes:
            value = getattr(summaries[mode], attr)
            if unit == '$':
                print(f"${value:>14.6f}", end='')
            elif unit:
                print(f"{value:>14.1f}{unit}", end='')
            else:
                print(f"{value:>16.1f}", end='')
        print()
    
    print(f"{'='*80}")
    
    # Highlight winner
    classic_e2e = summaries.get('CLASSIC', BenchmarkSummary('',0,0,0,'')).e2e_latency_mean_ms
    future_e2e = summaries.get('FUTURE_BASED', BenchmarkSummary('',0,0,0,'')).e2e_latency_mean_ms
    
    if classic_e2e > 0 and future_e2e > 0:
        improvement = (classic_e2e - future_e2e) / classic_e2e * 100
        print(f"\n  FUTURE_BASED vs CLASSIC: {improvement:.1f}% latency improvement")


if __name__ == '__main__':
    main()
