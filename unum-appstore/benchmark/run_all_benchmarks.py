#!/usr/bin/env python3
"""
Unified Benchmark Runner for Research Workflows

Runs comprehensive benchmarks across multiple workflows to demonstrate
Future-Based execution benefits with fan-in/fan-out patterns.

Workflows:
1. progressive-aggregator - Original baseline (FanOut→Source(5)→Aggregator)
2. ml-training-pipeline - ML ensemble (DataGen→[LR, SVM, RF, GB]→Aggregator)
3. video-analysis - Video processing (Decoder→Detector(batches)→Accumulator)
4. image-processing-pipeline - Image ops ([Meta, Thumb, Resize, Filter, Faces]→Agg)
5. genomics-pipeline - Scientific (Individuals→Merge→Sifting→[Overlap, Freq]→Final)
6. multi-source-dashboard - Multi-source data aggregation (Trigger→[6 sources]→Merge)

Execution Modes:
- CLASSIC: Synchronous fan-in (last invoker executes aggregator)
- EAGER: Polling-based blocking fan-in with LazyInput
- FUTURE_BASED: Async fan-in with parallel background polling

Usage:
    python run_all_benchmarks.py --workflow all --mode all --iterations 10
    python run_all_benchmarks.py --workflow ml-training-pipeline --mode FUTURE_BASED --iterations 5
    python run_all_benchmarks.py --analyze results/  # Analyze existing results
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


# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get('AWS_REGION', 'eu-central-1')

# Workflow configurations
WORKFLOWS = {
    'progressive-aggregator': {
        'stack_name': 'progressive-aggregator',
        'entry_function': 'FanOut',
        'terminal_function': 'Aggregator',
        'dynamodb_table': 'unum-intermediary-progressive',
        'fan_in_sizes': [5],
        'expected_delays': [2.0, 3.0, 4.0, 0.3, 0.5],  # Source delays
        'description': 'Basic fan-out/fan-in with 5 parallel sources',
    },
    'ml-training-pipeline': {
        'stack_name': 'ml-training-pipeline',
        'entry_function': 'DataGenerator',
        'terminal_function': 'ModelAggregator',
        'dynamodb_table': 'unum-intermediary-ml-training',
        'fan_in_sizes': [4],  # 4 models
        'expected_delays': [0.1, 2.0, 8.0, 5.0],  # LR, SVM, RF, GB
        'description': 'ML ensemble: LR(100ms), SVM(2s), RF(8s), GB(5s)',
    },
    'video-analysis': {
        'stack_name': 'video-analysis',
        'entry_function': 'VideoDecoder',
        'terminal_function': 'ResultAccumulator',
        'dynamodb_table': 'unum-intermediary-video',
        'fan_in_sizes': [6],  # 6 frame batches
        'expected_delays': [0.3, 1.5, 4.0, 2.0, 0.5, 6.0],  # Frame complexities
        'description': 'Video: 6 batches with varying complexity (0.3s-6s)',
    },
    'image-processing-pipeline': {
        'stack_name': 'image-processing-pipeline',
        'entry_function': 'ImageLoader',
        'terminal_function': 'ImageAggregator',
        'dynamodb_table': 'unum-intermediary-image-proc',
        'fan_in_sizes': [5],  # 5 operations
        'expected_delays': [0.05, 0.15, 0.4, 1.5, 3.5],  # Meta, Thumb, Resize, Filter, Faces
        'description': 'Image: Meta(50ms), Thumb(150ms), Resize(400ms), Filter(1.5s), Faces(3.5s)',
    },
    'genomics-pipeline': {
        'stack_name': 'genomics-pipeline',
        'entry_function': 'DataSplitter',
        'terminal_function': 'FinalAggregator',
        'dynamodb_table': 'unum-intermediary-genomics',
        'fan_in_sizes': [6, 2],  # 6 individuals, then 2 analyses
        'expected_delays': [0.5, 0.4, 1.5, 1.8, 2.5, 3.5],  # Individual processing times
        'description': 'Genomics: 6 individuals (0.4s-3.5s) + 2 analyses (0.4s, 3s)',
    },
'multi-source-dashboard': {
        'stack_name': 'multi-source-dashboard',
        'entry_function': 'TriggerDashboard',
        'terminal_function': 'MergeDashboardData',
        'dynamodb_table': 'unum-intermediary-multi-dashboard',
        'fan_in_sizes': [6],
        # Update these to match your new app.py files:
        'expected_delays': [1.0, 3.0, 5.0, 7.0, 9.0, 12.0], 
        'description': 'Staircase Benchmark: Sales(1s), Inventory(3s), Marketing(5s), Market(7s), Weather(9s), Competitor(12s)',
    },
    'order-processing-workflow': {
        'stack_name': 'order-processing-workflow',
        'entry_function': 'TriggerFunction',
        'terminal_function': 'Aggregator',
        'dynamodb_table': 'unum-intermediate-datastore-orders',
        'fan_in_sizes': [6],
        # Update these to match your new app.py files:
        'expected_delays': [1.0, 3.0, 5.0, 7.0, 9.0, 12.0], 
        'description': 'Staircase Benchmark: Sales(1s), Inventory(3s), Marketing(5s), Market(7s), Weather(9s), Competitor(12s)',
    },
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

# Pricing (eu-west-1)
PRICING = {
    'lambda_gb_second': 0.0000166667,
    'lambda_request': 0.0000002,
    'dynamodb_wcu': 0.00000125,
    'dynamodb_rcu': 0.00000025,
}

LOG_GROUP_PREFIX = '/aws/lambda/'


# ============================================================
# Data Classes
# ============================================================

@dataclass
class LambdaMetrics:
    """Metrics from Lambda REPORT log"""
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
    """Fan-in wait metrics"""
    function_name: str
    mode: str
    fan_in_size: int
    initially_ready: int = 0
    wait_duration_ms: float = 0.0
    poll_count: int = 0
    pre_resolved_count: int = 0


@dataclass
class WorkflowRun:
    """Single workflow execution metrics"""
    run_id: int
    workflow: str
    session_id: str
    mode: str
    run_type: str  # 'cold' or 'warm'
    start_time: float
    end_time: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    
    lambda_metrics: List[LambdaMetrics] = field(default_factory=list)
    fanin_metrics: List[FanInMetrics] = field(default_factory=list)
    
    total_duration_ms: float = 0.0
    total_billed_duration_ms: float = 0.0
    cold_start_count: int = 0
    pre_resolved_count: int = 0
    
    # Memory metrics (important for FUTURE_BASED overhead analysis)
    max_memory_used_mb: int = 0           # Peak memory across all functions
    total_memory_mb: int = 0              # Sum of max memory per function
    aggregator_memory_mb: int = 0         # Aggregator-specific memory (key metric)
    memory_efficiency: float = 0.0        # max_used / allocated ratio
    
    error: Optional[str] = None
    
    def compute_aggregates(self):
        if self.lambda_metrics:
            self.total_duration_ms = sum(m.duration_ms for m in self.lambda_metrics)
            self.total_billed_duration_ms = sum(m.billed_duration_ms for m in self.lambda_metrics)
            self.cold_start_count = sum(1 for m in self.lambda_metrics if m.is_cold_start)
            
            # Memory aggregates
            self.max_memory_used_mb = max((m.max_memory_used_mb for m in self.lambda_metrics), default=0)
            self.total_memory_mb = sum(m.max_memory_used_mb for m in self.lambda_metrics)
            
            # Find aggregator memory (terminal function)
            for m in self.lambda_metrics:
                if 'aggregator' in m.function_name.lower() or 'accumulator' in m.function_name.lower():
                    self.aggregator_memory_mb = m.max_memory_used_mb
                    break
            
            # Memory efficiency = actual used / allocated
            allocated = sum(m.memory_size_mb for m in self.lambda_metrics)
            if allocated > 0:
                self.memory_efficiency = self.total_memory_mb / allocated
                
        if self.fanin_metrics:
            self.pre_resolved_count = sum(m.pre_resolved_count for m in self.fanin_metrics)


@dataclass  
class WorkflowSummary:
    """Summary statistics for a workflow/mode combination"""
    workflow: str
    mode: str
    iterations: int
    successful_runs: int
    failed_runs: int
    timestamp: str
    
    # E2E Latency
    e2e_latency_mean_ms: float = 0.0
    e2e_latency_median_ms: float = 0.0
    e2e_latency_std_ms: float = 0.0
    e2e_latency_min_ms: float = 0.0
    e2e_latency_max_ms: float = 0.0
    e2e_latency_p95_ms: float = 0.0
    
    # Cold vs Warm
    cold_e2e_mean_ms: float = 0.0
    warm_e2e_mean_ms: float = 0.0
    
    # Future-Based metrics
    avg_pre_resolved: float = 0.0
    
    # Memory metrics (CRITICAL for FUTURE_BASED overhead analysis)
    avg_max_memory_mb: float = 0.0         # Average peak memory per run
    avg_total_memory_mb: float = 0.0       # Average sum of all functions' memory
    avg_aggregator_memory_mb: float = 0.0  # Average aggregator memory (key comparison)
    memory_overhead_vs_classic_pct: float = 0.0  # % increase vs CLASSIC mode
    avg_memory_efficiency: float = 0.0     # Average used/allocated ratio
    
    # Cost
    cost_per_run: float = 0.0
    total_cost: float = 0.0


# ============================================================
# Benchmark Runner
# ============================================================

class UnifiedBenchmarkRunner:
    """Runs benchmarks across all workflows"""
    
    def __init__(self, region: str = REGION):
        self.region = region
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.logs_client = boto3.client('logs', region_name=region)
        self.cf_client = boto3.client('cloudformation', region_name=region)
        
        self.functions_cache: Dict[str, Dict[str, str]] = {}
    
    def discover_functions(self, workflow: str) -> Dict[str, str]:
        """Discover Lambda functions from CloudFormation stack"""
        if workflow in self.functions_cache:
            return self.functions_cache[workflow]
        
        stack_name = WORKFLOWS[workflow]['stack_name']
        functions = {}
        
        try:
            response = self.cf_client.describe_stack_resources(StackName=stack_name)
            for resource in response['StackResources']:
                if resource['ResourceType'] == 'AWS::Lambda::Function':
                    logical_id = resource['LogicalResourceId']
                    physical_id = resource['PhysicalResourceId']
                    # Extract function name from logical ID
                    if logical_id.endswith('Function'):
                        func_name = logical_id[:-8]  # Removes last 8 chars ("Function")
                    else:
                        func_name = logical_id
                    functions[func_name] = physical_id
            print(f"    Discovered {len(functions)} functions for {workflow}")
        except Exception as e:
            print(f"    Warning: Could not discover functions for {workflow}: {e}")
        
        self.functions_cache[workflow] = functions
        return functions
    
    def set_mode(self, workflow: str, mode: str):
        """Configure workflow for specified execution mode"""
        config = MODE_CONFIGS[mode]
        functions = self.discover_functions(workflow)
        
        print(f"    Setting {workflow} to {mode} mode...")
        
        for func_name, func_arn in functions.items():
            try:
                response = self.lambda_client.get_function_configuration(
                    FunctionName=func_arn
                )
                current_env = response.get('Environment', {}).get('Variables', {})
                current_env['EAGER'] = str(config['Eager']).lower()
                current_env['UNUM_FUTURE_BASED'] = config['UNUM_FUTURE_BASED']
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={'Variables': current_env}
                )
            except Exception as e:
                print(f"      Failed to update {func_name}: {e}")
        
        time.sleep(5)
        self._wait_for_functions_ready(workflow)
    
    def _wait_for_functions_ready(self, workflow: str, timeout: int = 60):
        """Wait for all functions to be in Active state"""
        functions = self.discover_functions(workflow)
        start = time.time()
        
        for func_name, func_arn in functions.items():
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
    
    def force_cold_starts(self, workflow: str):
        """Force cold starts by updating environment"""
        functions = self.discover_functions(workflow)
        timestamp = str(int(time.time()))
        
        for func_name, func_arn in functions.items():
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
                print(f"      Failed: {e}")
        
        time.sleep(5)
        self._wait_for_functions_ready(workflow)
    
    def invoke_workflow(self, workflow: str) -> Tuple[str, float, str]:
        """Invoke workflow and return (request_id, start_time, session_id)"""
        functions = self.discover_functions(workflow)
        entry_func = WORKFLOWS[workflow]['entry_function']
        
        session_id = f"{workflow}-{int(time.time() * 1000)}"
        
        # Build payload based on workflow
        if workflow == 'ml-training-pipeline':
            payload = {
                "Data": {"Source": "http", "Value": {"dataset_size": 1000}},
                "Session": session_id
            }
        elif workflow == 'video-analysis':
            payload = {
                "Data": {"Source": "http", "Value": {"video_id": "test_video", "duration_s": 60}},
                "Session": session_id
            }
        elif workflow == 'image-processing-pipeline':
            payload = {
                "Data": {"Source": "http", "Value": {"image_id": "test_img", "width": 4000, "height": 3000}},
                "Session": session_id
            }
        elif workflow == 'genomics-pipeline':
            payload = {
                "Data": {"Source": "http", "Value": {"chromosome": 22, "num_individuals": 6}},
                "Session": session_id
            }
        elif workflow == 'order-processing-workflow':
            payload = {
                "Data": {
                    "Source": "http",
                    "Value": {
                        "order_id": session_id,
                        "customer_id": "BENCH-CUSTOMER",
                        "items": [
                            {"sku": "ITEM-001", "quantity": 2, "price": 49.99},
                            {"sku": "ITEM-002", "quantity": 1, "price": 29.99}
                        ]
                    }
                }
    }
        else:
            payload = {
                "Data": {"Source": "http", "Value": {}},
                "Session": session_id
            }
        
        # Find entry function ARN
        entry_arn = None
        for name, arn in functions.items():
            if entry_func.lower() in name.lower():
                entry_arn = arn
                break
        
        if not entry_arn:
            raise ValueError(f"Could not find entry function {entry_func} for {workflow}")
        
        start_time = time.time()
        response = self.lambda_client.invoke(
            FunctionName=entry_arn,
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
        
        request_id = response.get('ResponseMetadata', {}).get('RequestId', '')
        return request_id, start_time, session_id
    
    def wait_for_completion(self, workflow: str, start_time: float,
                           timeout: int = 180) -> Tuple[bool, float]:
        """Wait for workflow completion by monitoring terminal function logs"""
        functions = self.discover_functions(workflow)
        terminal_func = WORKFLOWS[workflow]['terminal_function']
        
        # Find terminal function ARN
        terminal_arn = None
        for name, arn in functions.items():
            if terminal_func.lower() in name.lower():
                terminal_arn = arn
                break
        
        if not terminal_arn:
            return False, time.time()
        
        log_group = f"{LOG_GROUP_PREFIX}{terminal_arn}"
        start_ms = int(start_time * 1000)
        
        # Completion patterns for different workflows
        completion_patterns = [
            '"COMPLETED"',
            '"pipeline_complete"',
            '"processing_complete"',
            '"merge_complete"',
            '"aggregation_time_ms"',
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
                            if event.get('timestamp', 0) >= start_ms - 1000:
                                return True, time.time()
                except Exception:
                    pass
            
            time.sleep(0.5)
        
        return False, time.time()
    
    def collect_metrics(self, workflow: str, session_id: str,
                       start_time: float, end_time: float) -> WorkflowRun:
        """Collect metrics from CloudWatch logs"""
        functions = self.discover_functions(workflow)
        run = WorkflowRun(
            run_id=0,
            workflow=workflow,
            session_id=session_id,
            mode='',
            run_type='',
            start_time=start_time,
            end_time=end_time,
            e2e_latency_ms=(end_time - start_time) * 1000
        )
        
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + 60000  # +60s buffer
        
        for func_name, func_arn in functions.items():
            log_group = f"{LOG_GROUP_PREFIX}{func_arn}"
            
            try:
                # Get REPORT logs
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=start_ms,
                    endTime=end_ms,
                    filterPattern='REPORT RequestId'
                )
                
                for event in response.get('events', []):
                    message = event.get('message', '')
                    metrics = self._parse_report_log(message, func_name)
                    if metrics:
                        run.lambda_metrics.append(metrics)
                
                # Get pre-resolved count for terminal function
                terminal = WORKFLOWS[workflow]['terminal_function']
                if terminal.lower() in func_name.lower():
                    response = self.logs_client.filter_log_events(
                        logGroupName=log_group,
                        startTime=start_ms,
                        endTime=end_ms,
                        filterPattern='"pre_resolved"'
                    )
                    
                    for event in response.get('events', []):
                        message = event.get('message', '')
                        match = re.search(r'pre_resolved["\s:]+(\d+)', message)
                        if match:
                            run.pre_resolved_count = int(match.group(1))
                            break
            except Exception as e:
                pass
        
        run.compute_aggregates()
        return run
    
    def _parse_report_log(self, message: str, func_name: str) -> Optional[LambdaMetrics]:
        """Parse Lambda REPORT log line"""
        try:
            req_match = re.search(r'RequestId:\s*([a-f0-9-]+)', message)
            dur_match = re.search(r'Duration:\s*([\d.]+)\s*ms', message)
            billed_match = re.search(r'Billed Duration:\s*(\d+)\s*ms', message)
            mem_match = re.search(r'Memory Size:\s*(\d+)\s*MB', message)
            max_mem_match = re.search(r'Max Memory Used:\s*(\d+)\s*MB', message)
            init_match = re.search(r'Init Duration:\s*([\d.]+)\s*ms', message)
            
            if dur_match and billed_match:
                return LambdaMetrics(
                    function_name=func_name,
                    request_id=req_match.group(1) if req_match else '',
                    duration_ms=float(dur_match.group(1)),
                    billed_duration_ms=float(billed_match.group(1)),
                    memory_size_mb=int(mem_match.group(1)) if mem_match else 128,
                    max_memory_used_mb=int(max_mem_match.group(1)) if max_mem_match else 0,
                    init_duration_ms=float(init_match.group(1)) if init_match else None
                )
        except Exception:
            pass
        return None
    
    def run_benchmark(self, workflow: str, mode: str, iterations: int,
                     cold_iterations: int = 2, warm_iterations: int = None,
                     results_dir: str = 'results') -> WorkflowSummary:
        """Run complete benchmark for workflow/mode combination"""
        
        if warm_iterations is None:
            warm_iterations = iterations - cold_iterations
        
        print(f"\n{'='*60}")
        print(f"Benchmark: {workflow} - {mode}")
        print(f"{'='*60}")
        
        # Setup mode
        self.set_mode(workflow, mode)
        
        runs: List[WorkflowRun] = []
        
        # Cold start runs
        print(f"\n  Cold Start Runs ({cold_iterations}):")
        for i in range(cold_iterations):
            print(f"    Run {i+1}/{cold_iterations}...")
            self.force_cold_starts(workflow)
            time.sleep(2)
            
            try:
                req_id, start_time, session_id = self.invoke_workflow(workflow)
                success, end_time = self.wait_for_completion(workflow, start_time)
                
                if success:
                    # Wait for CloudWatch propagation
                    time.sleep(10)
                    run = self.collect_metrics(workflow, session_id, start_time, end_time)
                    run.run_id = i + 1
                    run.mode = mode
                    run.run_type = 'cold'
                    runs.append(run)
                    print(f"      ✓ E2E: {run.e2e_latency_ms:.0f}ms, Pre-resolved: {run.pre_resolved_count}")
                else:
                    print(f"      ✗ Timeout")
            except Exception as e:
                print(f"      ✗ Error: {e}")
        
        # Warm start runs
        print(f"\n  Warm Start Runs ({warm_iterations}):")
        time.sleep(5)  # Let containers warm up
        
        for i in range(warm_iterations):
            print(f"    Run {i+1}/{warm_iterations}...")
            
            try:
                req_id, start_time, session_id = self.invoke_workflow(workflow)
                success, end_time = self.wait_for_completion(workflow, start_time)
                
                if success:
                    time.sleep(8)
                    run = self.collect_metrics(workflow, session_id, start_time, end_time)
                    run.run_id = cold_iterations + i + 1
                    run.mode = mode
                    run.run_type = 'warm'
                    runs.append(run)
                    print(f"      ✓ E2E: {run.e2e_latency_ms:.0f}ms, Pre-resolved: {run.pre_resolved_count}")
                else:
                    print(f"      ✗ Timeout")
                
                time.sleep(2)  # Small delay between runs
            except Exception as e:
                print(f"      ✗ Error: {e}")
        
        # Compute summary
        summary = self._compute_summary(workflow, mode, runs)
        
        # Save results
        self._save_results(workflow, mode, runs, summary, results_dir)
        
        return summary
    
    def _compute_summary(self, workflow: str, mode: str, runs: List[WorkflowRun]) -> WorkflowSummary:
        """Compute summary statistics"""
        successful = [r for r in runs if r.e2e_latency_ms is not None]
        failed = len(runs) - len(successful)
        
        summary = WorkflowSummary(
            workflow=workflow,
            mode=mode,
            iterations=len(runs),
            successful_runs=len(successful),
            failed_runs=failed,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        if successful:
            latencies = [r.e2e_latency_ms for r in successful]
            cold_runs = [r for r in successful if r.run_type == 'cold']
            warm_runs = [r for r in successful if r.run_type == 'warm']
            
            summary.e2e_latency_mean_ms = statistics.mean(latencies)
            summary.e2e_latency_median_ms = statistics.median(latencies)
            summary.e2e_latency_std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0
            summary.e2e_latency_min_ms = min(latencies)
            summary.e2e_latency_max_ms = max(latencies)
            
            if len(latencies) >= 20:
                sorted_lat = sorted(latencies)
                summary.e2e_latency_p95_ms = sorted_lat[int(len(sorted_lat) * 0.95)]
            
            if cold_runs:
                summary.cold_e2e_mean_ms = statistics.mean([r.e2e_latency_ms for r in cold_runs])
            if warm_runs:
                summary.warm_e2e_mean_ms = statistics.mean([r.e2e_latency_ms for r in warm_runs])
            
            summary.avg_pre_resolved = statistics.mean([r.pre_resolved_count for r in successful])
            
            # Memory metrics computation
            max_mem = [r.max_memory_used_mb for r in successful]
            total_mem = [r.total_memory_mb for r in successful]
            agg_mem = [r.aggregator_memory_mb for r in successful if r.aggregator_memory_mb > 0]
            mem_eff = [r.memory_efficiency for r in successful if r.memory_efficiency > 0]

            summary.avg_max_memory_mb = statistics.mean(max_mem) if max_mem else 0
            summary.avg_total_memory_mb = statistics.mean(total_mem) if total_mem else 0
            summary.avg_aggregator_memory_mb = statistics.mean(agg_mem) if agg_mem else 0
            summary.avg_memory_efficiency = statistics.mean(mem_eff) if mem_eff else 0
            
            # Cost estimation
            total_billed = sum(r.total_billed_duration_ms for r in successful)
            summary.total_cost = (total_billed / 1000) * (128 / 1024) * PRICING['lambda_gb_second']
            summary.cost_per_run = summary.total_cost / len(successful) if successful else 0
        
        return summary
    
    def _save_results(self, workflow: str, mode: str, runs: List[WorkflowRun],
                     summary: WorkflowSummary, results_dir: str):
        """Save results to JSON files"""
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save runs
        runs_file = f"{results_dir}/benchmark_{workflow}_{mode}_{timestamp}_runs.json"
        runs_data = []
        for run in runs:
            run_dict = {
                'run_id': run.run_id,
                'workflow': run.workflow,
                'mode': run.mode,
                'run_type': run.run_type,
                'session_id': run.session_id,
                'e2e_latency_ms': run.e2e_latency_ms,
                'total_duration_ms': run.total_duration_ms,
                'total_billed_duration_ms': run.total_billed_duration_ms,
                'cold_start_count': run.cold_start_count,
                'pre_resolved_count': run.pre_resolved_count,
                # Memory metrics
                'max_memory_used_mb': run.max_memory_used_mb,
                'total_memory_mb': run.total_memory_mb,
                'aggregator_memory_mb': run.aggregator_memory_mb,
                'memory_efficiency': run.memory_efficiency,
            }
            runs_data.append(run_dict)
        
        with open(runs_file, 'w') as f:
            json.dump(runs_data, f, indent=2)
        
        # Save summary
        summary_file = f"{results_dir}/benchmark_{workflow}_{mode}_{timestamp}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        
        print(f"\n  Results saved to:")
        print(f"    {runs_file}")
        print(f"    {summary_file}")
    
    def run_all_workflows(self, modes: List[str], iterations: int,
                         cold_iterations: int = 2, results_dir: str = 'results'):
        """Run benchmarks for all workflows across specified modes"""
        
        all_summaries: Dict[str, Dict[str, WorkflowSummary]] = {}
        
        for workflow in WORKFLOWS.keys():
            all_summaries[workflow] = {}
            
            for mode in modes:
                try:
                    summary = self.run_benchmark(
                        workflow=workflow,
                        mode=mode,
                        iterations=iterations,
                        cold_iterations=cold_iterations,
                        results_dir=results_dir
                    )
                    all_summaries[workflow][mode] = summary
                except Exception as e:
                    print(f"\n  ERROR: {workflow}/{mode} failed: {e}")
        
        # Generate comparison report
        self._generate_comparison_report(all_summaries, results_dir)
        
        return all_summaries
    
    def _generate_comparison_report(self, summaries: Dict[str, Dict[str, WorkflowSummary]],
                                    results_dir: str):
        """Generate markdown comparison report"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"{results_dir}/COMPARISON_REPORT_{timestamp}.md"
        
        lines = [
            "# Multi-Workflow Benchmark Comparison Report",
            f"\nGenerated: {datetime.datetime.now().isoformat()}",
            "\n## Executive Summary",
            "\nThis report compares CLASSIC, EAGER, and FUTURE_BASED execution modes",
            "across multiple serverless workflows with fan-in/fan-out patterns.",
            "\n## Workflow Results",
        ]
        
        for workflow, modes in summaries.items():
            config = WORKFLOWS[workflow]
            lines.append(f"\n### {workflow}")
            lines.append(f"\n*{config['description']}*")
            lines.append(f"\nExpected delays: {config['expected_delays']}")
            
            lines.append("\n| Metric | CLASSIC | EAGER | FUTURE_BASED |")
            lines.append("|--------|---------|-------|--------------|")
            
            def get_val(mode, attr, default='-'):
                if mode in modes:
                    return getattr(modes[mode], attr, default)
                return default
            
            lines.append(f"| E2E Mean (ms) | {get_val('CLASSIC', 'e2e_latency_mean_ms', '-'):.0f} | {get_val('EAGER', 'e2e_latency_mean_ms', '-'):.0f} | {get_val('FUTURE_BASED', 'e2e_latency_mean_ms', '-'):.0f} |")
            lines.append(f"| E2E Std Dev | {get_val('CLASSIC', 'e2e_latency_std_ms', '-'):.1f} | {get_val('EAGER', 'e2e_latency_std_ms', '-'):.1f} | {get_val('FUTURE_BASED', 'e2e_latency_std_ms', '-'):.1f} |")
            lines.append(f"| Cold Mean (ms) | {get_val('CLASSIC', 'cold_e2e_mean_ms', '-'):.0f} | {get_val('EAGER', 'cold_e2e_mean_ms', '-'):.0f} | {get_val('FUTURE_BASED', 'cold_e2e_mean_ms', '-'):.0f} |")
            lines.append(f"| Warm Mean (ms) | {get_val('CLASSIC', 'warm_e2e_mean_ms', '-'):.0f} | {get_val('EAGER', 'warm_e2e_mean_ms', '-'):.0f} | {get_val('FUTURE_BASED', 'warm_e2e_mean_ms', '-'):.0f} |")
            lines.append(f"| Avg Pre-Resolved | {get_val('CLASSIC', 'avg_pre_resolved', 0):.1f} | {get_val('EAGER', 'avg_pre_resolved', 0):.1f} | {get_val('FUTURE_BASED', 'avg_pre_resolved', 0):.1f} |")
            lines.append(f"| **Aggregator Mem (MB)** | {get_val('CLASSIC', 'avg_aggregator_memory_mb', 0):.1f} | {get_val('EAGER', 'avg_aggregator_memory_mb', 0):.1f} | {get_val('FUTURE_BASED', 'avg_aggregator_memory_mb', 0):.1f} |")
            lines.append(f"| Total Mem (MB) | {get_val('CLASSIC', 'avg_total_memory_mb', 0):.1f} | {get_val('EAGER', 'avg_total_memory_mb', 0):.1f} | {get_val('FUTURE_BASED', 'avg_total_memory_mb', 0):.1f} |")
            lines.append(f"| Mem Efficiency | {get_val('CLASSIC', 'avg_memory_efficiency', 0):.2f} | {get_val('EAGER', 'avg_memory_efficiency', 0):.2f} | {get_val('FUTURE_BASED', 'avg_memory_efficiency', 0):.2f} |")
            lines.append(f"| Cost/Run ($) | {get_val('CLASSIC', 'cost_per_run', 0):.7f} | {get_val('EAGER', 'cost_per_run', 0):.7f} | {get_val('FUTURE_BASED', 'cost_per_run', 0):.7f} |")
            
            # Calculate improvement and memory overhead
            if 'CLASSIC' in modes and 'FUTURE_BASED' in modes:
                classic_e2e = modes['CLASSIC'].e2e_latency_mean_ms
                future_e2e = modes['FUTURE_BASED'].e2e_latency_mean_ms
                classic_mem = modes['CLASSIC'].avg_aggregator_memory_mb
                future_mem = modes['FUTURE_BASED'].avg_aggregator_memory_mb
                
                if classic_e2e > 0:
                    improvement = (classic_e2e - future_e2e) / classic_e2e * 100
                    lines.append(f"\n**Latency Improvement: {improvement:.1f}%**")
                
                if classic_mem > 0 and future_mem > 0:
                    mem_overhead = (future_mem - classic_mem) / classic_mem * 100
                    lines.append(f"\n**Memory Overhead: {mem_overhead:+.1f}%** (aggregator)")
                    
                    # Efficiency metric: latency improvement per % memory overhead
                    if mem_overhead > 0:
                        efficiency = improvement / mem_overhead
                        lines.append(f"\n**Efficiency Ratio: {efficiency:.2f}** (latency gain / memory cost)")
        
        lines.append("\n## Conclusion")
        lines.append("\nFUTURE_BASED execution mode demonstrates significant latency")
        lines.append("improvements for workflows with heterogeneous parallel task durations.")
        lines.append("The benefit scales with the variance in task completion times.")
        
        with open(report_file, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"\n{'='*60}")
        print(f"Comparison report saved: {report_file}")
        print(f"{'='*60}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Benchmark Runner for Research Workflows'
    )
    parser.add_argument('--workflow', type=str, default='all',
                       choices=['all'] + list(WORKFLOWS.keys()),
                       help='Workflow to benchmark')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'CLASSIC', 'EAGER', 'FUTURE_BASED'],
                       help='Execution mode')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Total iterations per mode')
    parser.add_argument('--cold-iterations', type=int, default=2,
                       help='Cold start iterations')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--analyze', type=str, default=None,
                       help='Analyze existing results directory')
    
    args = parser.parse_args()
    
    runner = UnifiedBenchmarkRunner()
    
    if args.analyze:
        print(f"Analyzing results in {args.analyze}...")
        # TODO: Implement analysis of existing results
        return
    
    modes = ['CLASSIC', 'EAGER', 'FUTURE_BASED'] if args.mode == 'all' else [args.mode]
    
    if args.workflow == 'all':
        runner.run_all_workflows(
            modes=modes,
            iterations=args.iterations,
            cold_iterations=args.cold_iterations,
            results_dir=args.results_dir
        )
    else:
        for mode in modes:
            runner.run_benchmark(
                workflow=args.workflow,
                mode=mode,
                iterations=args.iterations,
                cold_iterations=args.cold_iterations,
                results_dir=args.results_dir
            )


if __name__ == '__main__':
    main()
