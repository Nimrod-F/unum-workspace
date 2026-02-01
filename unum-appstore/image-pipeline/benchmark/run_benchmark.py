#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for Image Pipeline Workflow

Compares execution modes:
1. CLASSIC - Synchronous fan-in (last invoker executes aggregator/publisher)
2. FUTURE_BASED - Async fan-in with parallel background polling

Workflow Structure:
  ImageLoader → [Thumbnail, Transform, Filters, Contour] → Publisher
                ^----------- fan-out -----------^        ^fan-in^

Expected Task Durations (real PIL computation):
  - Thumbnail:  ~50-150ms   (resize to 128x128 - FASTEST)
  - Transform:  ~100-300ms  (rotate + flip)
  - Filters:    ~200-500ms  (blur + sharpen)
  - Contour:    ~800-2000ms (edge detection - SLOWEST)

Metrics Collected:
  - End-to-End Latency (invoke → completion)
  - Per-Function Duration (CloudWatch REPORT logs)
  - Billed Duration (for cost calculation)
  - Cold Start Duration (Init Duration)
  - Memory Usage (Max Memory Used)
  - Fan-In Wait Time
  - Which branch triggered the Publisher
  - Pre-resolved count (FUTURE_BASED benefit)

Usage:
    python run_benchmark.py --mode CLASSIC --iterations 5 --cold
    python run_benchmark.py --mode FUTURE_BASED --iterations 5 --cold
    python run_benchmark.py --all --iterations 5 --cold
    python run_benchmark.py --analyze results/
"""

import boto3
import json
import time
import argparse
import os
import re
import statistics
import datetime
import yaml
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get('AWS_REGION', 'eu-central-1')
STACK_NAME = 'image-pipeline'
PROFILE = os.environ.get('AWS_PROFILE', 'research-profile')

# DynamoDB table
DYNAMODB_TABLE = 'unum-intermediate-datastore'

# Log group prefix
LOG_GROUP_PREFIX = '/aws/lambda/'

# Test image configuration
TEST_BUCKET = os.environ.get('TEST_BUCKET', 'unum-benchmark-images')
TEST_KEY = os.environ.get('TEST_KEY', 'test-images/sample-1920x1080.jpg')

# Workflow structure
WORKFLOW_STRUCTURE = {
    'ImageLoader': {'count': 1, 'fan_out_to': ['Thumbnail', 'Transform', 'Filters', 'Contour']},
    'Thumbnail': {'count': 1, 'fan_in_to': 'Publisher', 'index': 0, 'expected_ms': 100},
    'Transform': {'count': 1, 'fan_in_to': 'Publisher', 'index': 1, 'expected_ms': 200},
    'Filters': {'count': 1, 'fan_in_to': 'Publisher', 'index': 2, 'expected_ms': 350},
    'Contour': {'count': 1, 'fan_in_to': 'Publisher', 'index': 3, 'expected_ms': 1400},
    'Publisher': {'count': 1, 'terminal': True},
}

# Branches for fan-in
FAN_IN_BRANCHES = ['Thumbnail', 'Transform', 'Filters', 'Contour']

# Pricing (eu-central-1)
PRICING = {
    'lambda_gb_second': 0.0000166667,
    'lambda_request': 0.0000002,
    'dynamodb_wcu': 0.00000125,
    'dynamodb_rcu': 0.00000025,
}

# Mode configurations
MODE_CONFIGS = {
    'CLASSIC': {
        'Eager': 'true',
        'UNUM_FUTURE_BASED': 'false',
    },
    'FUTURE_BASED': {
        'Eager': 'true',
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
        return self.init_duration_ms is not None and self.init_duration_ms > 0
    
    @property
    def memory_efficiency(self) -> float:
        """Ratio of used memory to allocated memory"""
        if self.memory_size_mb > 0:
            return self.max_memory_used_mb / self.memory_size_mb
        return 0.0


@dataclass
class FanInMetrics:
    """Metrics for fan-in operations"""
    function_name: str
    session_id: str
    mode: str  # CLASSIC, FUTURE_BASED
    invoker_branch: str  # Which branch triggered the publisher
    initially_ready: int = 0
    total_branches: int = 4  # Thumbnail, Transform, Filters, Contour
    wait_duration_ms: float = 0.0
    poll_count: int = 0
    pre_resolved_count: int = 0  # Branches already complete when publisher starts


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
    run_type: str = 'cold'  # 'cold' or 'warm'
    test_image: str = ''
    
    # Per-function metrics
    lambda_metrics: List[LambdaMetrics] = field(default_factory=list)
    
    # Fan-in metrics
    fanin_metrics: Optional[FanInMetrics] = None
    invoker_branch: str = ''
    
    # Aggregated metrics
    total_duration_ms: float = 0.0
    total_billed_duration_ms: float = 0.0
    cold_start_count: int = 0
    total_init_duration_ms: float = 0.0
    max_memory_used_mb: int = 0
    total_memory_used_mb: int = 0
    publisher_memory_mb: int = 0
    avg_memory_efficiency: float = 0.0
    
    # Per-function durations
    image_loader_duration_ms: float = 0.0
    thumbnail_duration_ms: float = 0.0
    transform_duration_ms: float = 0.0
    filters_duration_ms: float = 0.0
    contour_duration_ms: float = 0.0
    publisher_duration_ms: float = 0.0
    
    # Cost
    estimated_cost_usd: float = 0.0
    
    # Pre-resolved metrics (FUTURE_BASED benefit)
    pre_resolved_count: int = 0
    
    # Timing variance (difference between fastest and slowest branch)
    branch_variance_ms: float = 0.0
    
    # Error tracking
    error: Optional[str] = None
    
    def compute_aggregates(self):
        """Compute aggregate metrics from per-function data"""
        if not self.lambda_metrics:
            return
            
        self.total_duration_ms = sum(m.duration_ms for m in self.lambda_metrics)
        self.total_billed_duration_ms = sum(m.billed_duration_ms for m in self.lambda_metrics)
        self.cold_start_count = sum(1 for m in self.lambda_metrics if m.is_cold_start)
        self.total_init_duration_ms = sum(m.init_duration_ms or 0 for m in self.lambda_metrics)
        self.max_memory_used_mb = max((m.max_memory_used_mb for m in self.lambda_metrics), default=0)
        self.total_memory_used_mb = sum(m.max_memory_used_mb for m in self.lambda_metrics)
        
        # Calculate memory efficiency
        efficiencies = [m.memory_efficiency for m in self.lambda_metrics if m.memory_efficiency > 0]
        self.avg_memory_efficiency = statistics.mean(efficiencies) if efficiencies else 0.0
        
        # Extract per-function durations
        branch_durations = []
        for m in self.lambda_metrics:
            fname = m.function_name.lower()
            if 'imageloader' in fname:
                self.image_loader_duration_ms = m.duration_ms
            elif 'thumbnail' in fname:
                self.thumbnail_duration_ms = m.duration_ms
                branch_durations.append(m.duration_ms)
            elif 'transform' in fname:
                self.transform_duration_ms = m.duration_ms
                branch_durations.append(m.duration_ms)
            elif 'filter' in fname:
                self.filters_duration_ms = m.duration_ms
                branch_durations.append(m.duration_ms)
            elif 'contour' in fname:
                self.contour_duration_ms = m.duration_ms
                branch_durations.append(m.duration_ms)
            elif 'publisher' in fname:
                self.publisher_duration_ms = m.duration_ms
                self.publisher_memory_mb = m.max_memory_used_mb
        
        # Calculate branch variance
        if branch_durations:
            self.branch_variance_ms = max(branch_durations) - min(branch_durations)
        
        # Calculate cost
        self.estimated_cost_usd = 0.0
        for m in self.lambda_metrics:
            gb_seconds = (m.billed_duration_ms / 1000) * (m.memory_size_mb / 1024)
            self.estimated_cost_usd += gb_seconds * PRICING['lambda_gb_second']
            self.estimated_cost_usd += PRICING['lambda_request']


@dataclass
class BenchmarkSummary:
    """Statistical summary for a benchmark run"""
    workflow: str
    mode: str
    iterations: int
    successful_runs: int
    failed_runs: int
    timestamp: str
    
    # Test configuration
    test_image: str = ''
    
    # E2E Latency
    e2e_latency_mean_ms: float = 0.0
    e2e_latency_median_ms: float = 0.0
    e2e_latency_std_ms: float = 0.0
    e2e_latency_min_ms: float = 0.0
    e2e_latency_max_ms: float = 0.0
    
    # Cold/Warm specific
    cold_e2e_mean_ms: float = 0.0
    warm_e2e_mean_ms: float = 0.0
    
    # Total Lambda Duration
    total_duration_mean_ms: float = 0.0
    total_duration_median_ms: float = 0.0
    
    # Per-function average durations
    image_loader_mean_ms: float = 0.0
    thumbnail_mean_ms: float = 0.0
    transform_mean_ms: float = 0.0
    filters_mean_ms: float = 0.0
    contour_mean_ms: float = 0.0
    publisher_mean_ms: float = 0.0
    
    # Billed Duration
    billed_duration_mean_ms: float = 0.0
    billed_duration_total_ms: float = 0.0
    
    # Cold Starts
    cold_start_rate: float = 0.0
    avg_init_duration_ms: float = 0.0
    total_cold_starts: int = 0
    
    # Fan-In metrics
    invoker_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Branch variance (key metric for FUTURE_BASED benefit)
    branch_variance_mean_ms: float = 0.0
    
    # Pre-resolved (FUTURE_BASED benefit)
    avg_pre_resolved_count: float = 0.0
    
    # Cost estimates
    lambda_compute_cost: float = 0.0
    lambda_request_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_run: float = 0.0


# ============================================================
# Benchmark Runner
# ============================================================

class ImagePipelineBenchmark:
    """Benchmark runner for image pipeline workflow"""
    
    def __init__(self, region: str = REGION, profile: str = PROFILE):
        self.region = region
        self.profile = profile
        
        # Set up boto3 session with profile
        session = boto3.Session(profile_name=profile, region_name=region)
        self.lambda_client = session.client('lambda')
        self.logs_client = session.client('logs')
        self.dynamodb = session.resource('dynamodb')
        
        # Load function ARNs
        self.functions = self._load_function_arns()
        
    def _load_function_arns(self) -> Dict[str, str]:
        """Load function ARNs from function-arn.yaml"""
        functions = {}
        yaml_path = Path(__file__).parent.parent / 'function-arn.yaml'
        
        if yaml_path.exists():
            with open(yaml_path) as f:
                functions = yaml.safe_load(f)
            print(f"✓ Loaded {len(functions)} functions from function-arn.yaml")
        else:
            print(f"⚠ Warning: function-arn.yaml not found at {yaml_path}")
        
        return functions
    
    def configure_mode(self, mode: str):
        """Configure all functions for the specified execution mode"""
        print(f"\n  Configuring {mode} mode...")
        
        config = MODE_CONFIGS[mode]
        
        for func_name, func_arn in self.functions.items():
            try:
                env_vars = {
                    'CHECKPOINT': 'true',
                    'DEBUG': 'true',
                    'FAAS_PLATFORM': 'aws',
                    'GC': 'false',
                    'UNUM_INTERMEDIARY_DATASTORE_NAME': DYNAMODB_TABLE,
                    'UNUM_INTERMEDIARY_DATASTORE_TYPE': 'dynamodb',
                    'EAGER': config['Eager'],
                }
                
                # Add UNUM_FUTURE_BASED for Publisher (fan-in function)
                if func_name == 'Publisher':
                    env_vars['UNUM_FUTURE_BASED'] = config['UNUM_FUTURE_BASED']
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={'Variables': env_vars}
                )
            except Exception as e:
                print(f"    ⚠ Warning: Could not update {func_name}: {e}")
        
        # Wait for updates to propagate
        time.sleep(8)
        print(f"  ✓ All functions configured for {mode} mode")
    
    def force_cold_start(self, mode: str):
        """Force cold starts by updating function configurations with timestamp"""
        print(f"  Forcing cold starts...")
        
        config = MODE_CONFIGS[mode]
        
        for func_name, func_arn in self.functions.items():
            try:
                env_vars = {
                    'COLD_START_TRIGGER': str(time.time()),
                    'CHECKPOINT': 'true',
                    'DEBUG': 'true',
                    'FAAS_PLATFORM': 'aws',
                    'GC': 'false',
                    'UNUM_INTERMEDIARY_DATASTORE_NAME': DYNAMODB_TABLE,
                    'UNUM_INTERMEDIARY_DATASTORE_TYPE': 'dynamodb',
                    'EAGER': config['Eager'],
                }
                
                if func_name == 'Publisher':
                    env_vars['UNUM_FUTURE_BASED'] = config['UNUM_FUTURE_BASED']
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={'Variables': env_vars}
                )
            except Exception as e:
                pass  # Ignore errors during cold start forcing
        
        time.sleep(8)
    
    def invoke_workflow(self, bucket: str = TEST_BUCKET, key: str = TEST_KEY) -> Tuple[str, float, float]:
        """Invoke the image pipeline workflow"""
        session_id = str(int(time.time() * 1000))
        
        payload = {
            "Data": {
                "Source": "http",
                "Value": {
                    "bucket": bucket,
                    "key": key
                }
            }
        }
        
        start_time = time.time()
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=self.functions['ImageLoader'],
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            end_time = time.time()
            
            if 'FunctionError' in response:
                error_payload = json.loads(response['Payload'].read())
                raise Exception(f"Lambda error: {error_payload}")
            
            return session_id, start_time, end_time
            
        except Exception as e:
            return session_id, start_time, time.time()
    
    def extract_lambda_metrics(self, function_name: str, start_time: float, 
                                end_time: float) -> Optional[LambdaMetrics]:
        """Extract Lambda metrics from CloudWatch logs"""
        func_arn = self.functions.get(function_name, '')
        if not func_arn:
            return None
        
        actual_func_name = func_arn.split(':')[-1]
        log_group = f"{LOG_GROUP_PREFIX}{actual_func_name}"
        
        try:
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000) + 60000  # Add 60 seconds buffer
            
            response = self.logs_client.filter_log_events(
                logGroupName=log_group,
                startTime=start_ms,
                endTime=end_ms,
                filterPattern='REPORT'
            )
            
            for event in response.get('events', []):
                message = event['message']
                
                # Parse REPORT log
                match = re.search(
                    r'REPORT RequestId: ([\w-]+)\s+'
                    r'Duration: ([\d.]+) ms\s+'
                    r'Billed Duration: (\d+) ms\s+'
                    r'Memory Size: (\d+) MB\s+'
                    r'Max Memory Used: (\d+) MB'
                    r'(?:\s+Init Duration: ([\d.]+) ms)?',
                    message
                )
                
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
                    
        except Exception as e:
            pass
        
        return None
    
    def get_invoker_branch(self, start_time: float, end_time: float) -> str:
        """Determine which branch triggered the Publisher"""
        invoke_patterns = [
            'invoking Publisher',
            'invoking next',
            'Successfully claimed',
            'all branches ready',
            'CLASSIC invoking',
        ]
        
        skip_patterns = [
            'already claimed',
            'waiting for others',
            'Skipping fan-in',
        ]
        
        invokers = []
        
        for branch in FAN_IN_BRANCHES:
            func_arn = self.functions.get(branch, '')
            if not func_arn:
                continue
            
            func_name = func_arn.split(':')[-1]
            log_group = f"{LOG_GROUP_PREFIX}{func_name}"
            
            try:
                response = self.logs_client.filter_log_events(
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
        else:
            return 'unknown'
    
    def get_pre_resolved_count(self, start_time: float, end_time: float) -> int:
        """Get count of pre-resolved futures (FUTURE_BASED benefit metric)"""
        func_arn = self.functions.get('Publisher', '')
        if not func_arn:
            return 0
        
        func_name = func_arn.split(':')[-1]
        log_group = f"{LOG_GROUP_PREFIX}{func_name}"
        
        try:
            response = self.logs_client.filter_log_events(
                logGroupName=log_group,
                startTime=int(start_time * 1000),
                endTime=int(end_time * 1000) + 60000,
                filterPattern='pre-resolved'
            )
            
            for event in response.get('events', []):
                match = re.search(r'(\d+)\s*pre-resolved', event['message'])
                if match:
                    return int(match.group(1))
                    
        except Exception:
            pass
        
        return 0
    
    def run_single_iteration(self, run_id: int, mode: str, run_type: str,
                             bucket: str = TEST_BUCKET, key: str = TEST_KEY) -> WorkflowRun:
        """Run a single benchmark iteration"""
        session_id, start_time, end_time = self.invoke_workflow(bucket, key)
        e2e_latency_ms = (end_time - start_time) * 1000
        
        run = WorkflowRun(
            run_id=run_id,
            session_id=session_id,
            mode=mode,
            start_time=start_time,
            end_time=end_time,
            e2e_latency_ms=e2e_latency_ms,
            run_type=run_type,
            test_image=f"s3://{bucket}/{key}"
        )
        
        print(f"    E2E Latency: {e2e_latency_ms:.1f}ms")
        
        # Wait for CloudWatch logs
        print(f"    Waiting for CloudWatch logs...")
        time.sleep(15)
        
        # Collect metrics from all functions
        all_functions = ['ImageLoader', 'Thumbnail', 'Transform', 'Filters', 'Contour', 'Publisher']
        
        for func_name in all_functions:
            metrics = self.extract_lambda_metrics(func_name, start_time, end_time)
            if metrics:
                run.lambda_metrics.append(metrics)
        
        # Get invoker and pre-resolved count
        run.invoker_branch = self.get_invoker_branch(start_time, end_time)
        run.pre_resolved_count = self.get_pre_resolved_count(start_time, end_time)
        
        # Compute aggregates
        run.compute_aggregates()
        
        print(f"    Invoker: {run.invoker_branch}")
        print(f"    Cold starts: {run.cold_start_count}")
        print(f"    Branch variance: {run.branch_variance_ms:.1f}ms")
        print(f"    Pre-resolved: {run.pre_resolved_count}")
        print(f"    Memory: max={run.max_memory_used_mb}MB, publisher={run.publisher_memory_mb}MB")
        print(f"    Cost: ${run.estimated_cost_usd:.6f}")
        
        return run
    
    def run_benchmark(self, mode: str, iterations: int = 5, 
                      cold_iterations: int = 2, warm_iterations: int = 3,
                      force_cold: bool = True) -> List[WorkflowRun]:
        """Run complete benchmark for a mode"""
        print(f"\n{'='*70}")
        print(f"  IMAGE PIPELINE BENCHMARK - {mode} MODE")
        print(f"  {cold_iterations} cold + {warm_iterations} warm = {iterations} total iterations")
        print(f"{'='*70}")
        
        # Configure mode
        self.configure_mode(mode)
        
        runs = []
        
        # Cold start iterations
        for i in range(cold_iterations):
            print(f"\n  Run {i+1}/{iterations} (COLD)")
            
            if force_cold:
                self.force_cold_start(mode)
            
            try:
                run = self.run_single_iteration(i + 1, mode, 'cold')
                runs.append(run)
            except Exception as e:
                print(f"    ✗ Error: {e}")
                runs.append(WorkflowRun(
                    run_id=i + 1,
                    session_id='',
                    mode=mode,
                    start_time=time.time(),
                    run_type='cold',
                    error=str(e)
                ))
        
        # Warm iterations (no cold start forcing)
        for i in range(warm_iterations):
            print(f"\n  Run {cold_iterations + i + 1}/{iterations} (WARM)")
            
            try:
                run = self.run_single_iteration(cold_iterations + i + 1, mode, 'warm')
                runs.append(run)
            except Exception as e:
                print(f"    ✗ Error: {e}")
                runs.append(WorkflowRun(
                    run_id=cold_iterations + i + 1,
                    session_id='',
                    mode=mode,
                    start_time=time.time(),
                    run_type='warm',
                    error=str(e)
                ))
            
            # Brief pause between warm runs
            if i < warm_iterations - 1:
                time.sleep(3)
        
        return runs
    
    def compute_summary(self, runs: List[WorkflowRun], mode: str) -> BenchmarkSummary:
        """Compute statistical summary from runs"""
        successful = [r for r in runs if r.error is None]
        failed = [r for r in runs if r.error is not None]
        
        summary = BenchmarkSummary(
            workflow='image-pipeline',
            mode=mode,
            iterations=len(runs),
            successful_runs=len(successful),
            failed_runs=len(failed),
            timestamp=datetime.datetime.now().isoformat(),
            test_image=f"s3://{TEST_BUCKET}/{TEST_KEY}"
        )
        
        if not successful:
            return summary
        
        # E2E Latency statistics
        latencies = [r.e2e_latency_ms for r in successful if r.e2e_latency_ms]
        if latencies:
            summary.e2e_latency_mean_ms = statistics.mean(latencies)
            summary.e2e_latency_median_ms = statistics.median(latencies)
            summary.e2e_latency_std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0
            summary.e2e_latency_min_ms = min(latencies)
            summary.e2e_latency_max_ms = max(latencies)
        
        # Cold vs Warm
        cold_latencies = [r.e2e_latency_ms for r in successful if r.run_type == 'cold' and r.e2e_latency_ms]
        warm_latencies = [r.e2e_latency_ms for r in successful if r.run_type == 'warm' and r.e2e_latency_ms]
        
        if cold_latencies:
            summary.cold_e2e_mean_ms = statistics.mean(cold_latencies)
        if warm_latencies:
            summary.warm_e2e_mean_ms = statistics.mean(warm_latencies)
        
        # Duration statistics
        durations = [r.total_duration_ms for r in successful if r.total_duration_ms]
        if durations:
            summary.total_duration_mean_ms = statistics.mean(durations)
            summary.total_duration_median_ms = statistics.median(durations)
        
        # Per-function means
        summary.image_loader_mean_ms = statistics.mean([r.image_loader_duration_ms for r in successful if r.image_loader_duration_ms]) if any(r.image_loader_duration_ms for r in successful) else 0
        summary.thumbnail_mean_ms = statistics.mean([r.thumbnail_duration_ms for r in successful if r.thumbnail_duration_ms]) if any(r.thumbnail_duration_ms for r in successful) else 0
        summary.transform_mean_ms = statistics.mean([r.transform_duration_ms for r in successful if r.transform_duration_ms]) if any(r.transform_duration_ms for r in successful) else 0
        summary.filters_mean_ms = statistics.mean([r.filters_duration_ms for r in successful if r.filters_duration_ms]) if any(r.filters_duration_ms for r in successful) else 0
        summary.contour_mean_ms = statistics.mean([r.contour_duration_ms for r in successful if r.contour_duration_ms]) if any(r.contour_duration_ms for r in successful) else 0
        summary.publisher_mean_ms = statistics.mean([r.publisher_duration_ms for r in successful if r.publisher_duration_ms]) if any(r.publisher_duration_ms for r in successful) else 0
        
        # Billed duration
        billed = [r.total_billed_duration_ms for r in successful if r.total_billed_duration_ms]
        if billed:
            summary.billed_duration_mean_ms = statistics.mean(billed)
            summary.billed_duration_total_ms = sum(billed)
        
        # Cold starts
        total_cold = sum(r.cold_start_count for r in successful)
        total_functions = len(successful) * 6  # 6 functions per run
        summary.cold_start_rate = total_cold / total_functions if total_functions > 0 else 0
        summary.total_cold_starts = total_cold
        
        init_durations = [r.total_init_duration_ms for r in successful if r.total_init_duration_ms]
        if init_durations:
            summary.avg_init_duration_ms = statistics.mean(init_durations)
        
        # Invoker distribution
        for r in successful:
            if r.invoker_branch:
                summary.invoker_distribution[r.invoker_branch] = summary.invoker_distribution.get(r.invoker_branch, 0) + 1
        
        # Branch variance
        variances = [r.branch_variance_ms for r in successful if r.branch_variance_ms]
        if variances:
            summary.branch_variance_mean_ms = statistics.mean(variances)
        
        # Pre-resolved count (FUTURE_BASED)
        pre_resolved = [r.pre_resolved_count for r in successful]
        if pre_resolved:
            summary.avg_pre_resolved_count = statistics.mean(pre_resolved)
        
        # Cost
        costs = [r.estimated_cost_usd for r in successful if r.estimated_cost_usd]
        if costs:
            summary.total_cost = sum(costs)
            summary.cost_per_run = statistics.mean(costs)
        
        return summary


def print_comparison_report(classic_summary: BenchmarkSummary, future_summary: BenchmarkSummary):
    """Print a comparison report between modes"""
    print("\n" + "=" * 80)
    print("  IMAGE PIPELINE BENCHMARK COMPARISON REPORT")
    print("  CLASSIC vs FUTURE_BASED Execution")
    print("=" * 80)
    
    print(f"\n  Test Configuration:")
    print(f"    Image: {classic_summary.test_image}")
    print(f"    Iterations: {classic_summary.iterations} per mode")
    print(f"    Workflow: ImageLoader → [Thumbnail, Transform, Filters, Contour] → Publisher")
    
    print("\n" + "-" * 80)
    print("  END-TO-END LATENCY (ms)")
    print("-" * 80)
    print(f"  {'Metric':<25} {'CLASSIC':>15} {'FUTURE_BASED':>15} {'Improvement':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    
    improvement = classic_summary.e2e_latency_mean_ms - future_summary.e2e_latency_mean_ms
    improvement_pct = (improvement / classic_summary.e2e_latency_mean_ms * 100) if classic_summary.e2e_latency_mean_ms > 0 else 0
    
    print(f"  {'Mean':<25} {classic_summary.e2e_latency_mean_ms:>15.1f} {future_summary.e2e_latency_mean_ms:>15.1f} {improvement:>+12.1f}ms ({improvement_pct:+.1f}%)")
    print(f"  {'Median':<25} {classic_summary.e2e_latency_median_ms:>15.1f} {future_summary.e2e_latency_median_ms:>15.1f}")
    print(f"  {'Std Dev':<25} {classic_summary.e2e_latency_std_ms:>15.1f} {future_summary.e2e_latency_std_ms:>15.1f}")
    print(f"  {'Min':<25} {classic_summary.e2e_latency_min_ms:>15.1f} {future_summary.e2e_latency_min_ms:>15.1f}")
    print(f"  {'Max':<25} {classic_summary.e2e_latency_max_ms:>15.1f} {future_summary.e2e_latency_max_ms:>15.1f}")
    
    print("\n" + "-" * 80)
    print("  COLD vs WARM LATENCY (ms)")
    print("-" * 80)
    print(f"  {'Run Type':<25} {'CLASSIC':>15} {'FUTURE_BASED':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'Cold Start Mean':<25} {classic_summary.cold_e2e_mean_ms:>15.1f} {future_summary.cold_e2e_mean_ms:>15.1f}")
    print(f"  {'Warm Start Mean':<25} {classic_summary.warm_e2e_mean_ms:>15.1f} {future_summary.warm_e2e_mean_ms:>15.1f}")
    
    print("\n" + "-" * 80)
    print("  PER-FUNCTION DURATION (ms)")
    print("-" * 80)
    print(f"  {'Function':<25} {'CLASSIC':>15} {'FUTURE_BASED':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'ImageLoader':<25} {classic_summary.image_loader_mean_ms:>15.1f} {future_summary.image_loader_mean_ms:>15.1f}")
    print(f"  {'Thumbnail (fastest)':<25} {classic_summary.thumbnail_mean_ms:>15.1f} {future_summary.thumbnail_mean_ms:>15.1f}")
    print(f"  {'Transform':<25} {classic_summary.transform_mean_ms:>15.1f} {future_summary.transform_mean_ms:>15.1f}")
    print(f"  {'Filters':<25} {classic_summary.filters_mean_ms:>15.1f} {future_summary.filters_mean_ms:>15.1f}")
    print(f"  {'Contour (slowest)':<25} {classic_summary.contour_mean_ms:>15.1f} {future_summary.contour_mean_ms:>15.1f}")
    print(f"  {'Publisher':<25} {classic_summary.publisher_mean_ms:>15.1f} {future_summary.publisher_mean_ms:>15.1f}")
    
    print("\n" + "-" * 80)
    print("  INVOKER DISTRIBUTION (which branch triggered Publisher)")
    print("-" * 80)
    print(f"  {'Branch':<25} {'CLASSIC':>15} {'FUTURE_BASED':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    
    all_branches = set(classic_summary.invoker_distribution.keys()) | set(future_summary.invoker_distribution.keys())
    for branch in sorted(all_branches):
        classic_count = classic_summary.invoker_distribution.get(branch, 0)
        future_count = future_summary.invoker_distribution.get(branch, 0)
        print(f"  {branch:<25} {classic_count:>15} {future_count:>15}")
    
    print("\n" + "-" * 80)
    print("  KEY METRICS")
    print("-" * 80)
    print(f"  {'Metric':<35} {'CLASSIC':>15} {'FUTURE_BASED':>15}")
    print(f"  {'-'*35} {'-'*15} {'-'*15}")
    print(f"  {'Branch Variance (max-min) ms':<35} {classic_summary.branch_variance_mean_ms:>15.1f} {future_summary.branch_variance_mean_ms:>15.1f}")
    print(f"  {'Pre-resolved Count (avg)':<35} {0:>15} {future_summary.avg_pre_resolved_count:>15.1f}")
    print(f"  {'Cold Start Rate':<35} {classic_summary.cold_start_rate:>14.1%} {future_summary.cold_start_rate:>14.1%}")
    print(f"  {'Total Cold Starts':<35} {classic_summary.total_cold_starts:>15} {future_summary.total_cold_starts:>15}")
    print(f"  {'Cost per Run ($)':<35} {classic_summary.cost_per_run:>15.6f} {future_summary.cost_per_run:>15.6f}")
    
    print("\n" + "-" * 80)
    print("  ANALYSIS")
    print("-" * 80)
    
    # Expected vs Actual behavior
    expected_invoker_classic = "Contour (slowest branch)"
    expected_invoker_future = "Thumbnail (fastest branch)"
    
    print(f"\n  CLASSIC Mode:")
    print(f"    Expected Invoker: {expected_invoker_classic}")
    actual_classic = max(classic_summary.invoker_distribution.items(), key=lambda x: x[1])[0] if classic_summary.invoker_distribution else 'unknown'
    print(f"    Actual Invoker:   {actual_classic}")
    
    print(f"\n  FUTURE_BASED Mode:")
    print(f"    Expected Invoker: {expected_invoker_future}")
    actual_future = max(future_summary.invoker_distribution.items(), key=lambda x: x[1])[0] if future_summary.invoker_distribution else 'unknown'
    print(f"    Actual Invoker:   {actual_future}")
    
    print(f"\n  Latency Improvement: {improvement:.1f}ms ({improvement_pct:.1f}%)")
    print(f"  Branch Variance:    {classic_summary.branch_variance_mean_ms:.1f}ms (expected saving in FUTURE_BASED)")
    
    print("\n" + "=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    
    if improvement > 0:
        print(f"\n  ✓ FUTURE_BASED mode is {improvement:.1f}ms ({improvement_pct:.1f}%) faster than CLASSIC")
        print(f"    This is because the fastest branch (Thumbnail) triggers the Publisher,")
        print(f"    and other results are pre-resolved by the time they're accessed.")
    else:
        print(f"\n  ⚠ CLASSIC mode was faster by {-improvement:.1f}ms")
        print(f"    This may be due to cold start effects or low branch variance.")
    
    print()


def save_results(runs: List[WorkflowRun], summary: BenchmarkSummary, output_dir: str):
    """Save benchmark results to JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save individual runs
    runs_file = f"{output_dir}/benchmark_image-pipeline_{summary.mode}_{timestamp}_runs.json"
    with open(runs_file, 'w') as f:
        json.dump([asdict(r) for r in runs], f, indent=2, default=str)
    print(f"  ✓ Runs saved to {runs_file}")
    
    # Save summary
    summary_file = f"{output_dir}/benchmark_image-pipeline_{summary.mode}_{timestamp}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"  ✓ Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Image Pipeline Benchmark - CLASSIC vs FUTURE_BASED')
    parser.add_argument('--mode', choices=['CLASSIC', 'FUTURE_BASED'], help='Execution mode to benchmark')
    parser.add_argument('--all', action='store_true', help='Run both modes')
    parser.add_argument('--iterations', type=int, default=5, help='Total iterations per mode')
    parser.add_argument('--cold-iterations', type=int, default=2, help='Number of cold start iterations')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--bucket', default=TEST_BUCKET, help='S3 bucket with test image')
    parser.add_argument('--key', default=TEST_KEY, help='S3 key for test image')
    parser.add_argument('--region', default=REGION, help='AWS region')
    parser.add_argument('--profile', default=PROFILE, help='AWS profile')
    
    args = parser.parse_args()
    
    # Calculate warm iterations
    warm_iterations = args.iterations - args.cold_iterations
    
    print("=" * 80)
    print("  IMAGE PIPELINE BENCHMARK")
    print("  CLASSIC vs FUTURE_BASED Execution Mode Comparison")
    print("=" * 80)
    print(f"\n  Configuration:")
    print(f"    Region:            {args.region}")
    print(f"    Profile:           {args.profile}")
    print(f"    Test Image:        s3://{args.bucket}/{args.key}")
    print(f"    Iterations:        {args.iterations} ({args.cold_iterations} cold + {warm_iterations} warm)")
    print(f"    Output Directory:  {args.output_dir}")
    
    benchmark = ImagePipelineBenchmark(region=args.region, profile=args.profile)
    
    results = {}
    summaries = {}
    
    modes_to_run = []
    if args.all:
        modes_to_run = ['CLASSIC', 'FUTURE_BASED']
    elif args.mode:
        modes_to_run = [args.mode]
    else:
        modes_to_run = ['CLASSIC', 'FUTURE_BASED']  # Default to both
    
    for mode in modes_to_run:
        runs = benchmark.run_benchmark(
            mode=mode,
            iterations=args.iterations,
            cold_iterations=args.cold_iterations,
            warm_iterations=warm_iterations
        )
        results[mode] = runs
        
        summary = benchmark.compute_summary(runs, mode)
        summaries[mode] = summary
        
        save_results(runs, summary, args.output_dir)
    
    # Print comparison if both modes were run
    if 'CLASSIC' in summaries and 'FUTURE_BASED' in summaries:
        print_comparison_report(summaries['CLASSIC'], summaries['FUTURE_BASED'])
        
        # Save comparison report
        report_file = f"{args.output_dir}/COMPARISON_REPORT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write("# Image Pipeline Benchmark Comparison Report\n\n")
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- **CLASSIC E2E Mean**: {summaries['CLASSIC'].e2e_latency_mean_ms:.1f}ms\n")
            f.write(f"- **FUTURE_BASED E2E Mean**: {summaries['FUTURE_BASED'].e2e_latency_mean_ms:.1f}ms\n")
            improvement = summaries['CLASSIC'].e2e_latency_mean_ms - summaries['FUTURE_BASED'].e2e_latency_mean_ms
            improvement_pct = (improvement / summaries['CLASSIC'].e2e_latency_mean_ms * 100) if summaries['CLASSIC'].e2e_latency_mean_ms > 0 else 0
            f.write(f"- **Improvement**: {improvement:.1f}ms ({improvement_pct:.1f}%)\n\n")
            f.write("## Invoker Distribution\n\n")
            f.write("| Branch | CLASSIC | FUTURE_BASED |\n")
            f.write("|--------|---------|---------------|\n")
            all_branches = set(summaries['CLASSIC'].invoker_distribution.keys()) | set(summaries['FUTURE_BASED'].invoker_distribution.keys())
            for branch in sorted(all_branches):
                classic_count = summaries['CLASSIC'].invoker_distribution.get(branch, 0)
                future_count = summaries['FUTURE_BASED'].invoker_distribution.get(branch, 0)
                f.write(f"| {branch} | {classic_count} | {future_count} |\n")
        
        print(f"  ✓ Comparison report saved to {report_file}")
    
    print("\n  Benchmark complete!")


if __name__ == "__main__":
    main()
