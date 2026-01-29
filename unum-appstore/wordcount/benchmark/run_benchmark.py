#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for Wordcount MapReduce Workflow

Compares execution modes:
1. CLASSIC - Synchronous, blocking fan-in (last invoker executes)
2. FUTURE_BASED - Async fan-in with parallel background polling

Workflow Structure:
  UnumMap0 → Mapper(N) → Partition → Reducer(3) → Summary
             ^fan-in^              ^fan-in^

Metrics Collected:
- End-to-End Latency (invoke → completion)
- Per-Function Duration (CloudWatch REPORT logs)
- Billed Duration (for cost calculation)
- Cold Start Duration (Init Duration)
- Memory Usage (Max Memory Used)
- Fan-In Wait Time
- S3 Operations count

Usage:
    python run_benchmark.py --mode CLASSIC --iterations 5 --cold 5
    python run_benchmark.py --mode FUTURE_BASED --iterations 5 --cold 5
    python run_benchmark.py --all --iterations 5 --cold 5
"""

import boto3
import json
import time
import argparse
import os
import re
import statistics
import datetime
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import subprocess
import sys


# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get('AWS_REGION', 'eu-central-1')
STACK_NAME = 'unum-mapreduce-wordcount-dynamo-new'

# S3 bucket for wordcount data
S3_BUCKET = 'wordcount-benchmark-133480914851'

# Function names (will be detected from function-arn.yaml)
FUNCTIONS = {}

# DynamoDB table
DYNAMODB_TABLE = 'unum-dynamo-test-table'

# Log group prefix
LOG_GROUP_PREFIX = '/aws/lambda/'

# Default data generation parameters
DEFAULT_NUM_MAPPERS = 20
DEFAULT_WORDS_PER_MAPPER = 1000
DEFAULT_VOCAB_SIZE = 500

# Workflow structure - 5 functions with 2 fan-in points
WORKFLOW_STRUCTURE = {
    'UnumMap0': {'count': 1, 'fan_out_to': 'Mapper'},
    'Mapper': {'count': 'N', 'fan_in_to': 'Partition'},  # N = input array length
    'Partition': {'count': 1, 'fan_out_to': 'Reducer'},
    'Reducer': {'count': 3, 'fan_in_to': 'Summary'},  # Fixed 3 reducers
    'Summary': {'count': 1, 'terminal': True},
}

# Pricing (eu-central-1, as of 2024)
PRICING = {
    'lambda_gb_second': 0.0000166667,
    'lambda_request': 0.0000002,
    'dynamodb_wcu': 0.00000125,
    'dynamodb_rcu': 0.00000025,
    's3_put': 0.000005,
    's3_get': 0.0000004,
}

# Mode configurations
MODE_CONFIGS = {
    'CLASSIC': {
        'Eager': False,
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
    init_duration_ms: Optional[float] = None
    
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
    pre_resolved_count: int = 0
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
    
    # Run configuration
    run_type: Optional[str] = None  # 'cold', 'warm'
    num_mappers: int = 0
    words_per_mapper: int = 0
    total_words: int = 0
    
    # Per-function metrics
    lambda_metrics: List[LambdaMetrics] = field(default_factory=list)
    
    # Fan-in metrics (2 fan-in points: Partition, Summary)
    fanin_metrics: List[FanInMetrics] = field(default_factory=list)
    
    # Aggregated metrics
    total_duration_ms: float = 0.0
    total_billed_duration_ms: float = 0.0
    cold_start_count: int = 0
    total_init_duration_ms: float = 0.0
    max_memory_used_mb: int = 0
    
    # S3 operations
    s3_puts: int = 0
    s3_gets: int = 0
    
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
    workflow: str = 'wordcount'
    
    # Data scale
    num_mappers: int = 0
    words_per_mapper: int = 0
    total_words: int = 0
    
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
    
    # Fan-In Wait Time (combined for both fan-in points)
    fanin_wait_mean_ms: float = 0.0
    fanin_wait_max_ms: float = 0.0
    avg_poll_count: float = 0.0
    avg_pre_resolved: float = 0.0
    
    # S3 Operations
    avg_s3_puts: float = 0.0
    avg_s3_gets: float = 0.0
    
    # DynamoDB
    avg_dynamo_reads: float = 0.0
    avg_dynamo_writes: float = 0.0
    total_dynamo_reads: int = 0
    total_dynamo_writes: int = 0
    
    # Cost estimates
    lambda_compute_cost: float = 0.0
    lambda_request_cost: float = 0.0
    dynamodb_cost: float = 0.0
    s3_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_run: float = 0.0


# ============================================================
# Data Generation
# ============================================================

def generate_wordcount_payload(
    num_mappers: int = DEFAULT_NUM_MAPPERS,
    words_per_mapper: int = DEFAULT_WORDS_PER_MAPPER,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    bucket: str = S3_BUCKET
) -> dict:
    """Generate test payload for wordcount benchmark."""
    
    # Generate vocabulary
    vocab = [f"word{i}" for i in range(vocab_size)]
    
    # Generate text items
    items = []
    for _ in range(num_mappers):
        words = random.choices(vocab, k=words_per_mapper)
        text = ' '.join(words)
        items.append({
            "text": text,
            "destination": bucket
        })
    
    payload = {
        "Data": {
            "Source": "http",
            "Value": items
        }
    }
    
    return payload


# ============================================================
# S3 Cleanup
# ============================================================

def clear_s3_reducer_data(bucket_name: str = S3_BUCKET, region: str = REGION, async_mode: bool = False):
    """Clear all reducer directories from S3 bucket using AWS CLI (faster for large buckets)."""
    import subprocess
    
    if async_mode:
        # Fire and forget - don't wait for completion
        try:
            subprocess.Popen(
                ['aws', 's3', 'rm', f's3://{bucket_name}/', '--recursive', '--region', region],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return -1  # Unknown count in async mode
        except Exception as e:
            print(f"      Warning: Error starting async cleanup: {e}")
            return 0
    
    total_deleted = 0
    prefixes = ['reducer0/', 'reducer1/', 'reducer2/']
    
    for prefix in prefixes:
        try:
            # Use AWS CLI for fast deletion - it handles large numbers better
            # Use 60 second timeout per prefix (180 total max)
            result = subprocess.run(
                ['aws', 's3', 'rm', f's3://{bucket_name}/{prefix}', '--recursive', '--region', region],
                capture_output=True, text=True, timeout=60
            )
            # Count deleted objects from output
            if result.stdout:
                deleted = result.stdout.count('delete:')
                total_deleted += deleted
        except subprocess.TimeoutExpired:
            print(f"      Warning: Timeout cleaning {prefix}, continuing...")
        except Exception as e:
            print(f"      Warning: Error cleaning {prefix}: {e}")
    
    return total_deleted


# ============================================================
# Benchmark Runner
# ============================================================

class WordcountBenchmark:
    """Benchmark runner for wordcount MapReduce workflow"""
    
    def __init__(self, region: str = REGION, 
                 num_mappers: int = DEFAULT_NUM_MAPPERS,
                 words_per_mapper: int = DEFAULT_WORDS_PER_MAPPER):
        self.region = region
        self.num_mappers = num_mappers
        self.words_per_mapper = words_per_mapper
        self.total_words = num_mappers * words_per_mapper
        
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.logs_client = boto3.client('logs', region_name=region)
        self.s3 = boto3.resource('s3', region_name=region)
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        
        # Load function ARNs from function-arn.yaml
        self._load_function_arns()
        
        self.table = self.dynamodb.Table(DYNAMODB_TABLE)
        
        # Calculate total expected invocations
        # 1 UnumMap0 + N Mappers + 1 Partition + 3 Reducers + 1 Summary
        self.total_invocations = 1 + num_mappers + 1 + 3 + 1
    
    def _load_function_arns(self):
        """Load Lambda function ARNs from function-arn.yaml"""
        global FUNCTIONS
        
        script_dir = Path(__file__).parent.resolve()
        arn_file = script_dir.parent / 'function-arn.yaml'
        
        if arn_file.exists():
            import yaml
            with open(arn_file) as f:
                FUNCTIONS = yaml.safe_load(f)
            print(f"  Loaded functions: {list(FUNCTIONS.keys())}")
        else:
            print(f"  Warning: {arn_file} not found, using defaults")
            FUNCTIONS = {
                'UnumMap0': f'{STACK_NAME}-UnumMap0Function',
                'Mapper': f'{STACK_NAME}-MapperFunction',
                'Partition': f'{STACK_NAME}-PartitionFunction',
                'Reducer': f'{STACK_NAME}-ReducerFunction',
                'Summary': f'{STACK_NAME}-SummaryFunction',
            }
    
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
    # S3 Cleanup
    # --------------------------------------------------------
    
    def cleanup_s3(self, async_mode: bool = False):
        """Clear S3 reducer directories before each run"""
        print("    Clearing S3 reducer data...")
        deleted = clear_s3_reducer_data(S3_BUCKET, self.region, async_mode=async_mode)
        if async_mode:
            print(f"      Cleanup started (async)")
        else:
            print(f"      Deleted {deleted} objects")
        return deleted
    
    # --------------------------------------------------------
    # Workflow Invocation
    # --------------------------------------------------------
    
    def invoke_workflow(self, session_id: str = None) -> Tuple[str, float, str]:
        """
        Invoke the wordcount workflow.
        Returns (request_id, invoke_timestamp, session_id)
        """
        if session_id is None:
            session_id = f"bench-{int(time.time() * 1000)}"
        
        # Generate payload
        payload = generate_wordcount_payload(
            num_mappers=self.num_mappers,
            words_per_mapper=self.words_per_mapper
        )
        payload['Session'] = session_id
        
        start_time = time.time()
        response = self.lambda_client.invoke(
            FunctionName=FUNCTIONS['UnumMap0'],
            InvocationType='Event',  # Async invocation
            Payload=json.dumps(payload)
        )
        
        request_id = response.get('ResponseMetadata', {}).get('RequestId', '')
        return request_id, start_time, session_id
    
    def wait_for_completion(self, session_id: str, start_time: float,
                           timeout_seconds: int = 300) -> Tuple[bool, float]:
        """
        Wait for workflow completion by checking Summary function logs.
        Returns (success, end_time)
        
        Wordcount is more complex and takes longer, so use longer timeout.
        """
        # Extract function name from ARN (last part after :function:)
        summary_arn = FUNCTIONS['Summary']
        func_name = summary_arn.split(':function:')[-1] if ':function:' in summary_arn else summary_arn
        log_group = f"{LOG_GROUP_PREFIX}{func_name}"
        start_ms = int(start_time * 1000)
        
        print(f"\n    Waiting for completion (log_group={log_group}, start_ms={start_ms})...", flush=True)
        
        deadline = time.time() + timeout_seconds
        poll_count = 0
        while time.time() < deadline:
            poll_count += 1
            try:
                # Look for REPORT log which indicates Summary completed
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=start_ms,
                    filterPattern='REPORT RequestId'
                )
                
                events = response.get('events', [])
                if poll_count % 10 == 0:
                    print(f"      Poll #{poll_count}: found {len(events)} events", flush=True)
                
                if events:
                    for event in events:
                        log_timestamp = event.get('timestamp', 0)
                        if log_timestamp >= start_ms - 1000:
                            end_time = time.time()
                            print(f"    Completed! (log_ts={log_timestamp}, start_ms={start_ms})", flush=True)
                            return True, end_time
            except Exception as e:
                if poll_count % 10 == 0:
                    print(f"      Poll #{poll_count}: error - {e}", flush=True)
            
            time.sleep(1)
        
        print(f"    Timeout after {poll_count} polls", flush=True)
        return False, time.time()
    
    # --------------------------------------------------------
    # Metrics Collection
    # --------------------------------------------------------
    
    def collect_lambda_metrics(self, session_id: str, start_time: float,
                               end_time: float) -> List[LambdaMetrics]:
        """Collect Lambda metrics from CloudWatch REPORT logs"""
        metrics = []
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + 120000  # Add 2 minute buffer
        
        for func_name, func_arn in FUNCTIONS.items():
            # Extract function name from ARN for log group
            actual_func_name = func_arn.split(':function:')[-1] if ':function:' in func_arn else func_arn
            log_group = f"{LOG_GROUP_PREFIX}{actual_func_name}"
            
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
            req_match = re.search(r'RequestId:\s*([a-f0-9-]+)', message)
            request_id = req_match.group(1) if req_match else ''
            
            dur_match = re.search(r'Duration:\s*([\d.]+)\s*ms', message)
            duration = float(dur_match.group(1)) if dur_match else 0.0
            
            billed_match = re.search(r'Billed Duration:\s*([\d.]+)\s*ms', message)
            billed = float(billed_match.group(1)) if billed_match else duration
            
            mem_size_match = re.search(r'Memory Size:\s*(\d+)\s*MB', message)
            mem_size = int(mem_size_match.group(1)) if mem_size_match else 128
            
            max_mem_match = re.search(r'Max Memory Used:\s*(\d+)\s*MB', message)
            max_mem = int(max_mem_match.group(1)) if max_mem_match else 0
            
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
        """
        Collect fan-in metrics from Partition and Summary logs.
        Wordcount has 2 fan-in points:
        1. Mapper → Partition (N mappers fan-in)
        2. Reducer → Summary (3 reducers fan-in)
        """
        metrics = []
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + 120000
        
        # Collect from both fan-in functions
        fanin_functions = [
            ('Partition', self.num_mappers),  # Fan-in from mappers
            ('Summary', 3),  # Fan-in from reducers
        ]
        
        for func_name, fan_in_size in fanin_functions:
            fanin = FanInMetrics(
                function_name=func_name,
                session_id=session_id,
                mode=mode,
                fan_in_size=fan_in_size
            )
            
            instant_count = 0
            # Extract function name from ARN for log group
            actual_func_name = FUNCTIONS[func_name].split(':function:')[-1] if ':function:' in FUNCTIONS[func_name] else FUNCTIONS[func_name]
            log_group = f"{LOG_GROUP_PREFIX}{actual_func_name}"
            
            try:
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=start_ms,
                    endTime=end_ms
                )
                
                for event in response.get('events', []):
                    msg = event['message']
                    
                    # Pre-resolved count
                    pre_resolved_match = re.search(r'pre-resolved.*?(\d+)/(\d+)', msg, re.IGNORECASE)
                    if pre_resolved_match:
                        fanin.pre_resolved_count = int(pre_resolved_match.group(1))
                    
                    # INSTANT resolves (only for FUTURE_BASED)
                    if 'INSTANT' in msg and mode == 'FUTURE_BASED':
                        instant_count += 1
                    
                    # Wait duration
                    wait_match = re.search(r'Received after waiting\s*(\d+)ms', msg)
                    if wait_match:
                        fanin.wait_duration_ms += float(wait_match.group(1))
                    
                    # Poll count
                    poll_match = re.search(r'Total resolved.*?(\d+)/(\d+)', msg)
                    if poll_match:
                        fanin.poll_count = max(fanin.poll_count, int(poll_match.group(1)))
                
                if fanin.pre_resolved_count == 0 and instant_count > 0 and mode == 'FUTURE_BASED':
                    fanin.pre_resolved_count = instant_count
            
            except Exception as e:
                print(f"      Warning: Could not collect fan-in metrics for {func_name}: {e}")
            
            fanin.strategy = 'last_invoker' if mode == 'CLASSIC' else 'UnumFuture_BackgroundPolling'
            metrics.append(fanin)
        
        return metrics
    
    def get_accurate_e2e_from_logs(self, start_time: float, end_time: float) -> Optional[float]:
        """
        Calculate accurate E2E latency from CloudWatch log timestamps.
        Returns E2E in milliseconds, or None if unable to determine.
        
        E2E = (Summary REPORT timestamp) - (UnumMap0 START timestamp)
        """
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + 180000  # 3 minute buffer for longer workflow
        
        unummap0_start_ts = None
        summary_end_ts = None
        
        try:
            # Get UnumMap0 START timestamp
            unummap0_func_name = FUNCTIONS['UnumMap0'].split(':function:')[-1] if ':function:' in FUNCTIONS['UnumMap0'] else FUNCTIONS['UnumMap0']
            unummap0_log_group = f"{LOG_GROUP_PREFIX}{unummap0_func_name}"
            response = self.logs_client.filter_log_events(
                logGroupName=unummap0_log_group,
                startTime=start_ms,
                endTime=end_ms,
                filterPattern='START RequestId'
            )
            for event in response.get('events', []):
                ts = event.get('timestamp', 0)
                if ts >= start_ms:
                    unummap0_start_ts = ts
                    break
            
            # Get Summary REPORT timestamp (marks completion)
            summary_func_name = FUNCTIONS['Summary'].split(':function:')[-1] if ':function:' in FUNCTIONS['Summary'] else FUNCTIONS['Summary']
            summary_log_group = f"{LOG_GROUP_PREFIX}{summary_func_name}"
            response = self.logs_client.filter_log_events(
                logGroupName=summary_log_group,
                startTime=start_ms,
                endTime=end_ms,
                filterPattern='REPORT RequestId'
            )
            for event in response.get('events', []):
                ts = event.get('timestamp', 0)
                if ts >= start_ms:
                    summary_end_ts = ts
            
            if unummap0_start_ts and summary_end_ts:
                return summary_end_ts - unummap0_start_ts
                
        except Exception as e:
            print(f"      Warning: Could not get accurate E2E: {e}")
        
        return None
    
    # --------------------------------------------------------
    # Single Run
    # --------------------------------------------------------
    
    def run_single(self, run_id: int, mode: str, 
                   force_cold: bool = False, skip_cleanup: bool = False) -> WorkflowRun:
        """Execute a single benchmark run"""
        session_id = f"bench-{mode}-{run_id}-{int(time.time() * 1000)}"
        
        run = WorkflowRun(
            run_id=run_id,
            session_id=session_id,
            mode=mode,
            start_time=0,
            run_type='cold' if force_cold else 'warm',
            num_mappers=self.num_mappers,
            words_per_mapper=self.words_per_mapper,
            total_words=self.total_words
        )
        
        try:
            # Clean up S3 before each run (skip for warm runs to speed up)
            if not skip_cleanup:
                self.cleanup_s3()
            else:
                print("    Skipping S3 cleanup for warm run...")
            
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
                time.sleep(8)  # Longer wait for more complex workflow
                
                # Get accurate E2E from CloudWatch log timestamps
                accurate_e2e = self.get_accurate_e2e_from_logs(start_time, end_time)
                if accurate_e2e:
                    run.e2e_latency_ms = accurate_e2e
                    print(f"(accurate E2E from logs)", end=' ', flush=True)
                else:
                    run.e2e_latency_ms = (end_time - start_time) * 1000
                    print(f"(fallback E2E from wall-clock)", end=' ', flush=True)
                
                # Collect metrics
                run.lambda_metrics = self.collect_lambda_metrics(
                    session_id, start_time, end_time
                )
                run.fanin_metrics = self.collect_fanin_metrics(
                    session_id, start_time, end_time, mode
                )
                
                run.compute_aggregates()
                
                # Estimate S3 operations (mappers write, reducers read)
                run.s3_puts = self.total_words  # Each word = 1 S3 PUT
                run.s3_gets = self.total_words  # Reducers read all
            else:
                run.error = "Timeout waiting for workflow completion"
        
        except Exception as e:
            run.error = str(e)
        
        return run
    
    # --------------------------------------------------------
    # Full Benchmark
    # --------------------------------------------------------
    
    def run_benchmark(self, mode: str, iterations: int = 5,
                      warmup_runs: int = 1,
                      cold_iterations: int = 0) -> Tuple[List[WorkflowRun], BenchmarkSummary]:
        """Run complete benchmark for a mode"""
        
        if iterations == 0 and cold_iterations > 0:
            warmup_runs = 0
        
        print(f"\n{'='*60}")
        print(f"  BENCHMARK: {mode} Mode (Wordcount)")
        print(f"  Data Scale: {self.num_mappers} mappers × {self.words_per_mapper} words = {self.total_words:,} total")
        print(f"  Iterations: {iterations} warm + {cold_iterations} cold (+ {warmup_runs} warmup)")
        print(f"{'='*60}")
        
        runs = []
        
        # Warmup runs
        if warmup_runs > 0:
            print(f"\n  Warmup runs ({warmup_runs})...")
            for i in range(warmup_runs):
                print(f"    Warmup {i+1}/{warmup_runs}...", end=' ', flush=True)
                run = self.run_single(i, mode, force_cold=False, skip_cleanup=True)
                status = "OK" if not run.error else f"FAILED: {run.error}"
                print(status)
                time.sleep(2)
        
        # Cold start runs
        if cold_iterations > 0:
            print(f"\n  Cold start runs ({cold_iterations})...")
            for i in range(cold_iterations):
                print(f"    Cold {i+1}/{cold_iterations}...", end=' ', flush=True)
                run = self.run_single(i, mode, force_cold=True)
                if run.error:
                    print(f"FAILED: {run.error}")
                else:
                    print(f"E2E: {run.e2e_latency_ms:.0f}ms, Cold starts: {run.cold_start_count}")
                runs.append(run)
                time.sleep(15)  # Longer wait for complex workflow
        
        # Warm runs
        if iterations > 0:
            print(f"\n  Warm runs ({iterations})...")
            for i in range(iterations):
                print(f"    Run {i+1}/{iterations}...", end=' ', flush=True)
                # Skip S3 cleanup for warm runs to speed up benchmark
                run = self.run_single(len(runs), mode, force_cold=False, skip_cleanup=True)
                if run.error:
                    print(f"FAILED: {run.error}")
                else:
                    print(f"E2E: {run.e2e_latency_ms:.0f}ms")
                runs.append(run)
                time.sleep(2)
        
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
            timestamp=datetime.datetime.now().isoformat(),
            num_mappers=self.num_mappers,
            words_per_mapper=self.words_per_mapper,
            total_words=self.total_words
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
        total_invocations = len(successful) * self.total_invocations
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
        
        # S3 Operations
        s3_puts = [r.s3_puts for r in successful]
        s3_gets = [r.s3_gets for r in successful]
        if s3_puts:
            summary.avg_s3_puts = statistics.mean(s3_puts)
        if s3_gets:
            summary.avg_s3_gets = statistics.mean(s3_gets)
        
        # DynamoDB
        summary.total_dynamo_reads = sum(r.dynamo_reads for r in successful)
        summary.total_dynamo_writes = sum(r.dynamo_writes for r in successful)
        summary.avg_dynamo_reads = summary.total_dynamo_reads / len(successful)
        summary.avg_dynamo_writes = summary.total_dynamo_writes / len(successful)
        
        # Cost Estimation
        total_gb_seconds = sum(
            r.total_billed_duration_ms / 1000 * 0.128
            for r in successful
        )
        summary.lambda_compute_cost = total_gb_seconds * PRICING['lambda_gb_second']
        summary.lambda_request_cost = len(successful) * self.total_invocations * PRICING['lambda_request']
        summary.dynamodb_cost = (
            summary.total_dynamo_reads * PRICING['dynamodb_rcu'] +
            summary.total_dynamo_writes * PRICING['dynamodb_wcu']
        )
        summary.s3_cost = (
            sum(r.s3_puts for r in successful) * PRICING['s3_put'] +
            sum(r.s3_gets for r in successful) * PRICING['s3_get']
        )
        
        summary.total_cost = (
            summary.lambda_compute_cost +
            summary.lambda_request_cost +
            summary.dynamodb_cost +
            summary.s3_cost
        )
        summary.cost_per_run = summary.total_cost / len(successful) if successful else 0
        
        return summary


# ============================================================
# Mode Switching
# ============================================================

def get_current_deployed_mode(region: str = REGION) -> Optional[str]:
    """Check Lambda env vars to determine currently deployed mode"""
    lambda_client = boto3.client('lambda', region_name=region)
    
    # Try to get Summary function config
    summary_func = FUNCTIONS.get('Summary')
    if not summary_func:
        # Load from file
        script_dir = Path(__file__).parent.resolve()
        arn_file = script_dir.parent / 'function-arn.yaml'
        if arn_file.exists():
            import yaml
            with open(arn_file) as f:
                arns = yaml.safe_load(f)
                summary_func = arns.get('Summary')
    
    if not summary_func:
        return None
    
    try:
        response = lambda_client.get_function_configuration(
            FunctionName=summary_func
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
    
    # Update Eager setting
    if config['Eager']:
        content = re.sub(r'Eager:\s*(true|false|True|False)', 'Eager: true', content)
    else:
        content = re.sub(r'Eager:\s*(true|false|True|False)', 'Eager: false', content)
    
    template_file.write_text(content)
    print(f"  Updated unum-template.yaml for {mode} mode")
    
    return True


def rebuild_and_deploy(project_dir: Path, unum_cli_path: Path):
    """Rebuild and redeploy the stack"""
    print("  Rebuilding and deploying...")
    
    os.chdir(project_dir)
    
    commands = [
        [sys.executable, str(unum_cli_path), 'template'],
        [sys.executable, str(unum_cli_path), 'build'],
        [sys.executable, str(unum_cli_path), 'deploy'],
    ]
    
    for cmd in commands:
        print(f"    Running: {' '.join(cmd[-2:])}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    Error: {result.stderr}")
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
    
    runs_file = output_dir / f'benchmark_{summary.mode}_{timestamp}_runs.json'
    runs_data = [asdict(r) for r in runs]
    with open(runs_file, 'w') as f:
        json.dump(runs_data, f, indent=2, default=str)
    print(f"  Saved runs to {runs_file}")
    
    summary_file = output_dir / f'benchmark_{summary.mode}_{timestamp}_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"  Saved summary to {summary_file}")
    
    return runs_file, summary_file


def print_summary(summary: BenchmarkSummary):
    """Print formatted summary"""
    print(f"\n{'='*60}")
    print(f"  RESULTS: {summary.mode} Mode (Wordcount)")
    print(f"{'='*60}")
    print(f"  Data: {summary.num_mappers} mappers × {summary.words_per_mapper} words = {summary.total_words:,}")
    print(f"  Runs: {summary.successful_runs} successful, {summary.failed_runs} failed")
    print()
    print(f"  E2E Latency:")
    print(f"    Mean:   {summary.e2e_latency_mean_ms:.1f} ms")
    print(f"    Median: {summary.e2e_latency_median_ms:.1f} ms")
    print(f"    Std:    {summary.e2e_latency_std_ms:.1f} ms")
    print(f"    Min:    {summary.e2e_latency_min_ms:.1f} ms")
    print(f"    Max:    {summary.e2e_latency_max_ms:.1f} ms")
    print()
    print(f"  Lambda Metrics:")
    print(f"    Total Duration (mean):  {summary.total_duration_mean_ms:.1f} ms")
    print(f"    Billed Duration (mean): {summary.billed_duration_mean_ms:.1f} ms")
    print(f"    Cold Start Rate: {summary.cold_start_rate*100:.1f}%")
    if summary.avg_init_duration_ms > 0:
        print(f"    Avg Init Duration: {summary.avg_init_duration_ms:.1f} ms")
    print()
    print(f"  Fan-In Metrics (2 fan-in points):")
    print(f"    Wait Time (mean): {summary.fanin_wait_mean_ms:.1f} ms")
    print(f"    Avg Pre-Resolved: {summary.avg_pre_resolved:.1f}")
    print()
    print(f"  Storage Operations:")
    print(f"    S3 PUTs/Run:  {summary.avg_s3_puts:.0f}")
    print(f"    S3 GETs/Run:  {summary.avg_s3_gets:.0f}")
    print(f"    DynamoDB Reads/Run:  {summary.avg_dynamo_reads:.1f}")
    print(f"    DynamoDB Writes/Run: {summary.avg_dynamo_writes:.1f}")
    print()
    print(f"  Cost Estimates:")
    print(f"    Lambda Compute: ${summary.lambda_compute_cost:.6f}")
    print(f"    Lambda Request: ${summary.lambda_request_cost:.6f}")
    print(f"    DynamoDB:       ${summary.dynamodb_cost:.6f}")
    print(f"    S3:             ${summary.s3_cost:.6f}")
    print(f"    Total:          ${summary.total_cost:.6f}")
    print(f"    Per Run:        ${summary.cost_per_run:.6f}")
    print(f"{'='*60}")


def print_comparison(summaries: Dict[str, BenchmarkSummary]):
    """Print comparison table across modes"""
    print(f"\n{'='*80}")
    print(f"  COMPARISON ACROSS MODES (Wordcount)")
    print(f"{'='*80}")
    
    modes = list(summaries.keys())
    print(f"{'Metric':<30}", end='')
    for mode in modes:
        print(f"{mode:>16}", end='')
    print()
    print("-" * (30 + 16 * len(modes)))
    
    metrics = [
        ('E2E Latency (mean)', 'e2e_latency_mean_ms', 'ms'),
        ('E2E Latency (median)', 'e2e_latency_median_ms', 'ms'),
        ('E2E Latency (std)', 'e2e_latency_std_ms', 'ms'),
        ('Total Duration (mean)', 'total_duration_mean_ms', 'ms'),
        ('Cold Start Rate', 'cold_start_rate', '%'),
        ('Fan-In Wait (mean)', 'fanin_wait_mean_ms', 'ms'),
        ('Avg Pre-Resolved', 'avg_pre_resolved', ''),
        ('S3 PUTs/Run', 'avg_s3_puts', ''),
        ('Cost Per Run', 'cost_per_run', '$'),
    ]
    
    for label, attr, unit in metrics:
        print(f"{label:<30}", end='')
        for mode in modes:
            value = getattr(summaries[mode], attr)
            if unit == '$':
                print(f"${value:>14.6f}", end='')
            elif unit == '%':
                print(f"{value*100:>14.1f}%", end='')
            elif unit:
                print(f"{value:>14.1f}{unit}", end='')
            else:
                print(f"{value:>16.1f}", end='')
        print()
    
    print(f"{'='*80}")
    
    # Highlight winner
    if len(summaries) >= 2:
        classic_e2e = summaries.get('CLASSIC', BenchmarkSummary('',0,0,0,'')).e2e_latency_mean_ms
        future_e2e = summaries.get('FUTURE_BASED', BenchmarkSummary('',0,0,0,'')).e2e_latency_mean_ms
        
        if classic_e2e > 0 and future_e2e > 0:
            improvement = (classic_e2e - future_e2e) / classic_e2e * 100
            winner = "FUTURE_BASED" if improvement > 0 else "CLASSIC"
            print(f"\n  {winner} is {abs(improvement):.1f}% {'faster' if improvement > 0 else 'slower'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Wordcount MapReduce Benchmark Runner'
    )
    parser.add_argument('--mode', choices=['CLASSIC', 'FUTURE_BASED'],
                        help='Execution mode to benchmark')
    parser.add_argument('--all', action='store_true',
                        help='Run benchmark for all modes')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of warm iterations per mode')
    parser.add_argument('--warmup', type=int, default=1,
                        help='Number of warmup runs')
    parser.add_argument('--cold', type=int, default=0,
                        help='Number of cold start runs')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--no-deploy', action='store_true',
                        help='Skip mode switching and deployment')
    parser.add_argument('--region', type=str, default=REGION,
                        help='AWS region')
    
    # Data scale options
    parser.add_argument('--mappers', type=int, default=DEFAULT_NUM_MAPPERS,
                        help=f'Number of mappers (default: {DEFAULT_NUM_MAPPERS})')
    parser.add_argument('--words', type=int, default=DEFAULT_WORDS_PER_MAPPER,
                        help=f'Words per mapper (default: {DEFAULT_WORDS_PER_MAPPER})')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent
    unum_cli_path = project_dir.parent.parent / 'unum' / 'unum-cli' / 'unum-cli.py'
    output_dir = script_dir / args.output
    
    print(f"\nWordcount MapReduce Benchmark")
    print(f"  Region: {args.region}")
    print(f"  Data Scale: {args.mappers} mappers × {args.words} words = {args.mappers * args.words:,} total")
    print(f"  Output: {output_dir}")
    
    modes = ['CLASSIC', 'FUTURE_BASED'] if args.all else [args.mode]
    
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
                pass
            elif not result:
                continue
            else:
                if not rebuild_and_deploy(project_dir, unum_cli_path):
                    continue
                time.sleep(10)
        
        # Run benchmark
        benchmark = WordcountBenchmark(
            region=args.region,
            num_mappers=args.mappers,
            words_per_mapper=args.words
        )
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


if __name__ == '__main__':
    main()
