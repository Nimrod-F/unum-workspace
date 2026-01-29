#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for Graph Analysis Workflow

Compares execution modes:
1. CLASSIC - Synchronous fan-in (last invoker executes aggregator)
2. FUTURE_BASED - Async fan-in with parallel background polling

Workflow Structure:
  GraphGenerator → [PageRank, BFS, MST] → Aggregator
                   ^--- fan-out ---^      ^fan-in^

Metrics Collected:
- End-to-End Latency (invoke → completion)
- Per-Function Duration (CloudWatch REPORT logs)
- Billed Duration (for cost calculation)
- Cold Start Duration (Init Duration)
- Memory Usage (Max Memory Used)
- Fan-In Wait Time
- Which branch triggered the Aggregator

Usage:
    python run_benchmark.py --mode CLASSIC --iterations 3 --cold
    python run_benchmark.py --mode FUTURE_BASED --iterations 3 --cold
    python run_benchmark.py --all --iterations 3 --cold
"""

import boto3
import json
import time
import argparse
import os
import re
import statistics
import datetime
import subprocess
import sys
import yaml
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get('AWS_REGION', 'eu-central-1')
STACK_NAME = 'graph-analysis'
PROFILE = os.environ.get('AWS_PROFILE', 'research-profile')

# DynamoDB table
DYNAMODB_TABLE = 'unum-intermediate-datastore'

# Log group prefix
LOG_GROUP_PREFIX = '/aws/lambda/'

# Default graph parameters
DEFAULT_NUM_NODES = 50
DEFAULT_EDGE_PROBABILITY = 0.3

# Workflow structure
WORKFLOW_STRUCTURE = {
    'GraphGenerator': {'count': 1, 'fan_out_to': ['PageRank', 'BFS', 'MST']},
    'PageRank': {'count': 1, 'fan_in_to': 'Aggregator', 'index': 0},
    'BFS': {'count': 1, 'fan_in_to': 'Aggregator', 'index': 1},
    'MST': {'count': 1, 'fan_in_to': 'Aggregator', 'index': 2},
    'Aggregator': {'count': 1, 'terminal': True},
}

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
        'Eager': 'false',
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
        return self.init_duration_ms is not None


@dataclass
class FanInMetrics:
    """Metrics for fan-in operations"""
    function_name: str
    session_id: str
    mode: str  # CLASSIC, EAGER, FUTURE
    invoker_branch: str  # Which branch triggered the aggregator
    initially_ready: int = 0
    total_branches: int = 3
    wait_duration_ms: float = 0.0
    poll_count: int = 0


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
    num_nodes: int = DEFAULT_NUM_NODES
    edge_probability: float = DEFAULT_EDGE_PROBABILITY
    
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
    
    # Per-function durations
    graph_generator_duration_ms: float = 0.0
    pagerank_duration_ms: float = 0.0
    bfs_duration_ms: float = 0.0
    mst_duration_ms: float = 0.0
    aggregator_duration_ms: float = 0.0
    
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
            
            # Extract per-function durations
            for m in self.lambda_metrics:
                if 'GraphGenerator' in m.function_name:
                    self.graph_generator_duration_ms = m.duration_ms
                elif 'PageRank' in m.function_name:
                    self.pagerank_duration_ms = m.duration_ms
                elif 'BFS' in m.function_name:
                    self.bfs_duration_ms = m.duration_ms
                elif 'MST' in m.function_name:
                    self.mst_duration_ms = m.duration_ms
                elif 'Aggregator' in m.function_name:
                    self.aggregator_duration_ms = m.duration_ms


@dataclass
class BenchmarkSummary:
    """Statistical summary for a benchmark run"""
    mode: str
    iterations: int
    successful_runs: int
    failed_runs: int
    timestamp: str
    workflow: str = 'graph-analysis'
    
    # Data scale
    num_nodes: int = DEFAULT_NUM_NODES
    edge_probability: float = DEFAULT_EDGE_PROBABILITY
    
    # E2E Latency
    e2e_latency_mean_ms: float = 0.0
    e2e_latency_median_ms: float = 0.0
    e2e_latency_std_ms: float = 0.0
    e2e_latency_min_ms: float = 0.0
    e2e_latency_max_ms: float = 0.0
    
    # Total Lambda Duration
    total_duration_mean_ms: float = 0.0
    total_duration_median_ms: float = 0.0
    
    # Per-function average durations
    graph_generator_mean_ms: float = 0.0
    pagerank_mean_ms: float = 0.0
    bfs_mean_ms: float = 0.0
    mst_mean_ms: float = 0.0
    aggregator_mean_ms: float = 0.0
    
    # Billed Duration
    billed_duration_mean_ms: float = 0.0
    billed_duration_total_ms: float = 0.0
    
    # Cold Starts
    cold_start_rate: float = 0.0
    avg_init_duration_ms: float = 0.0
    total_cold_starts: int = 0
    
    # Fan-In metrics
    invoker_distribution: Dict[str, int] = field(default_factory=dict)
    fanin_wait_mean_ms: float = 0.0
    
    # Cost estimates
    lambda_compute_cost: float = 0.0
    lambda_request_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_run: float = 0.0


# ============================================================
# Benchmark Runner
# ============================================================

class GraphAnalysisBenchmark:
    """Benchmark runner for graph analysis workflow"""
    
    def __init__(self, region: str = REGION, profile: str = PROFILE,
                 num_nodes: int = DEFAULT_NUM_NODES,
                 edge_probability: float = DEFAULT_EDGE_PROBABILITY):
        self.region = region
        self.profile = profile
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        
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
            print(f"Loaded {len(functions)} functions from function-arn.yaml")
        else:
            print(f"Warning: function-arn.yaml not found at {yaml_path}")
        
        return functions
    
    def force_cold_start(self):
        """Force cold start by updating function configurations"""
        print("  Forcing cold starts by updating function configurations...")
        
        for func_name, func_arn in self.functions.items():
            try:
                # Update environment variable to force new container
                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={
                        'Variables': {
                            'COLD_START_TRIGGER': str(time.time()),
                            'CHECKPOINT': 'true',
                            'DEBUG': 'true',
                            'FAAS_PLATFORM': 'aws',
                            'GC': 'false',
                            'UNUM_INTERMEDIARY_DATASTORE_NAME': DYNAMODB_TABLE,
                            'UNUM_INTERMEDIARY_DATASTORE_TYPE': 'dynamodb',
                            'EAGER': os.environ.get('BENCHMARK_EAGER', 'false'),
                        }
                    }
                )
            except Exception as e:
                print(f"    Warning: Could not update {func_name}: {e}")
        
        # Wait for updates to propagate
        time.sleep(5)
    
    def clear_dynamodb_session(self, session_id: str):
        """Clear DynamoDB entries for a session"""
        try:
            table = self.dynamodb.Table(DYNAMODB_TABLE)
            # Scan and delete items with matching session
            response = table.scan(
                FilterExpression='begins_with(#pk, :session)',
                ExpressionAttributeNames={'#pk': 'Name'},
                ExpressionAttributeValues={':session': session_id[:8]}
            )
            
            with table.batch_writer() as batch:
                for item in response.get('Items', []):
                    batch.delete_item(Key={'Name': item['Name']})
                    
        except Exception as e:
            print(f"    Warning: Could not clear DynamoDB: {e}")
    
    def invoke_workflow(self, session_id: str = None) -> Tuple[str, float, float]:
        """Invoke the graph analysis workflow and return session_id, start_time, response"""
        if session_id is None:
            session_id = str(int(time.time() * 1000))
        
        payload = {
            "Data": {
                "Source": "http",
                "Value": {
                    "num_nodes": self.num_nodes,
                    "edge_probability": self.edge_probability
                }
            }
        }
        
        start_time = time.time()
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=self.functions['GraphGenerator'],
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            end_time = time.time()
            
            # Check for errors
            if 'FunctionError' in response:
                error_payload = json.loads(response['Payload'].read())
                raise Exception(f"Lambda error: {error_payload}")
            
            return session_id, start_time, end_time
            
        except Exception as e:
            return session_id, start_time, time.time()
    
    def extract_lambda_metrics(self, function_name: str, start_time: float, 
                                end_time: float) -> List[LambdaMetrics]:
        """Extract Lambda metrics from CloudWatch logs"""
        metrics = []
        
        # Find the log group
        func_arn = self.functions.get(function_name, '')
        if not func_arn:
            return metrics
        
        # Extract the actual function name from ARN
        actual_func_name = func_arn.split(':')[-1]
        log_group = f"{LOG_GROUP_PREFIX}{actual_func_name}"
        
        try:
            # Query logs
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000) + 30000  # Add 30 seconds buffer
            
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
                    metrics.append(LambdaMetrics(
                        function_name=function_name,
                        request_id=match.group(1),
                        duration_ms=float(match.group(2)),
                        billed_duration_ms=float(match.group(3)),
                        memory_size_mb=int(match.group(4)),
                        max_memory_used_mb=int(match.group(5)),
                        init_duration_ms=float(match.group(6)) if match.group(6) else None
                    ))
                    
        except Exception as e:
            print(f"    Warning: Could not get logs for {function_name}: {e}")
        
        return metrics
    
    def extract_fanin_info(self, start_time: float, end_time: float) -> Tuple[str, str]:
        """Extract which branch invoked the aggregator and the mode"""
        invoker = 'unknown'
        mode = 'unknown'
        
        # Check each parallel branch's logs
        for branch in ['PageRank', 'BFS', 'MST']:
            func_arn = self.functions.get(branch, '')
            if not func_arn:
                continue
            
            actual_func_name = func_arn.split(':')[-1]
            log_group = f"{LOG_GROUP_PREFIX}{actual_func_name}"
            
            try:
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000) + 30000
                
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=start_ms,
                    endTime=end_ms,
                    filterPattern='invoking Aggregator'
                )
                
                for event in response.get('events', []):
                    message = event['message']
                    if 'CLASSIC invoking Aggregator' in message:
                        invoker = branch
                        mode = 'CLASSIC'
                        return invoker, mode
                    elif 'EAGER invoking Aggregator' in message:
                        invoker = branch
                        mode = 'EAGER/FUTURE'
                        return invoker, mode
                        
            except Exception as e:
                pass
        
        return invoker, mode
    
    def run_single_iteration(self, run_id: int, mode: str, 
                             run_type: str = 'warm') -> WorkflowRun:
        """Run a single benchmark iteration"""
        print(f"    Iteration {run_id} ({run_type})...")
        
        # Create run object
        run = WorkflowRun(
            run_id=run_id,
            session_id='',
            mode=mode,
            start_time=time.time(),
            run_type=run_type,
            num_nodes=self.num_nodes,
            edge_probability=self.edge_probability
        )
        
        try:
            # Invoke workflow
            session_id, start_time, end_time = self.invoke_workflow()
            run.session_id = session_id
            run.start_time = start_time
            run.end_time = end_time
            run.e2e_latency_ms = (end_time - start_time) * 1000
            
            # Wait for logs to be available
            time.sleep(5)
            
            # Extract metrics from each function
            for func_name in self.functions.keys():
                metrics = self.extract_lambda_metrics(func_name, start_time, end_time)
                run.lambda_metrics.extend(metrics)
            
            # Extract fan-in info
            invoker, fanin_mode = self.extract_fanin_info(start_time, end_time)
            run.invoker_branch = invoker
            
            # Compute aggregates
            run.compute_aggregates()
            
            print(f"      E2E: {run.e2e_latency_ms:.1f}ms, "
                  f"Cold starts: {run.cold_start_count}, "
                  f"Invoker: {invoker}")
            
        except Exception as e:
            run.error = str(e)
            print(f"      Error: {e}")
        
        return run
    
    def run_benchmark(self, mode: str, iterations: int = 3, 
                      cold_start: bool = True) -> BenchmarkSummary:
        """Run complete benchmark for a mode"""
        print(f"\n{'='*60}")
        print(f"Running {mode} benchmark ({iterations} iterations)")
        print(f"{'='*60}")
        
        runs: List[WorkflowRun] = []
        
        for i in range(iterations):
            run_type = 'cold' if cold_start else 'warm'
            
            if cold_start:
                self.force_cold_start()
            
            run = self.run_single_iteration(i + 1, mode, run_type)
            runs.append(run)
            
            # Wait between iterations
            if i < iterations - 1:
                time.sleep(3)
        
        # Generate summary
        summary = self._generate_summary(mode, runs)
        
        return summary
    
    def _generate_summary(self, mode: str, runs: List[WorkflowRun]) -> BenchmarkSummary:
        """Generate statistical summary from runs"""
        successful = [r for r in runs if r.error is None]
        failed = [r for r in runs if r.error is not None]
        
        summary = BenchmarkSummary(
            mode=mode,
            iterations=len(runs),
            successful_runs=len(successful),
            failed_runs=len(failed),
            timestamp=datetime.datetime.now().isoformat(),
            num_nodes=self.num_nodes,
            edge_probability=self.edge_probability
        )
        
        if successful:
            e2e_latencies = [r.e2e_latency_ms for r in successful if r.e2e_latency_ms]
            
            if e2e_latencies:
                summary.e2e_latency_mean_ms = statistics.mean(e2e_latencies)
                summary.e2e_latency_median_ms = statistics.median(e2e_latencies)
                summary.e2e_latency_std_ms = statistics.stdev(e2e_latencies) if len(e2e_latencies) > 1 else 0
                summary.e2e_latency_min_ms = min(e2e_latencies)
                summary.e2e_latency_max_ms = max(e2e_latencies)
            
            # Per-function durations
            gg_durations = [r.graph_generator_duration_ms for r in successful if r.graph_generator_duration_ms > 0]
            pr_durations = [r.pagerank_duration_ms for r in successful if r.pagerank_duration_ms > 0]
            bfs_durations = [r.bfs_duration_ms for r in successful if r.bfs_duration_ms > 0]
            mst_durations = [r.mst_duration_ms for r in successful if r.mst_duration_ms > 0]
            agg_durations = [r.aggregator_duration_ms for r in successful if r.aggregator_duration_ms > 0]
            
            if gg_durations:
                summary.graph_generator_mean_ms = statistics.mean(gg_durations)
            if pr_durations:
                summary.pagerank_mean_ms = statistics.mean(pr_durations)
            if bfs_durations:
                summary.bfs_mean_ms = statistics.mean(bfs_durations)
            if mst_durations:
                summary.mst_mean_ms = statistics.mean(mst_durations)
            if agg_durations:
                summary.aggregator_mean_ms = statistics.mean(agg_durations)
            
            # Total duration
            total_durations = [r.total_duration_ms for r in successful]
            if total_durations:
                summary.total_duration_mean_ms = statistics.mean(total_durations)
                summary.total_duration_median_ms = statistics.median(total_durations)
            
            # Billed duration
            billed = [r.total_billed_duration_ms for r in successful]
            if billed:
                summary.billed_duration_mean_ms = statistics.mean(billed)
                summary.billed_duration_total_ms = sum(billed)
            
            # Cold starts
            total_cold = sum(r.cold_start_count for r in successful)
            total_funcs = sum(len(r.lambda_metrics) for r in successful)
            summary.total_cold_starts = total_cold
            summary.cold_start_rate = total_cold / total_funcs if total_funcs > 0 else 0
            
            init_durations = [r.total_init_duration_ms for r in successful if r.total_init_duration_ms > 0]
            if init_durations:
                summary.avg_init_duration_ms = statistics.mean(init_durations)
            
            # Invoker distribution
            for r in successful:
                if r.invoker_branch:
                    summary.invoker_distribution[r.invoker_branch] = \
                        summary.invoker_distribution.get(r.invoker_branch, 0) + 1
            
            # Cost calculation
            total_billed_seconds = summary.billed_duration_total_ms / 1000
            avg_memory_gb = 0.5  # Assume 512MB average
            
            summary.lambda_compute_cost = total_billed_seconds * avg_memory_gb * PRICING['lambda_gb_second']
            summary.lambda_request_cost = len(successful) * 5 * PRICING['lambda_request']  # 5 functions
            summary.total_cost = summary.lambda_compute_cost + summary.lambda_request_cost
            summary.cost_per_run = summary.total_cost / len(successful) if successful else 0
        
        return summary


def update_template_for_mode(mode: str):
    """Update template.yaml for the specified mode"""
    config = MODE_CONFIGS[mode]
    
    template_path = Path(__file__).parent.parent / 'template.yaml'
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Update EAGER setting
    content = re.sub(
        r'EAGER:\s*(true|false)',
        f"EAGER: {config['Eager']}",
        content
    )
    
    with open(template_path, 'w') as f:
        f.write(content)
    
    print(f"Updated template.yaml for {mode} mode (EAGER={config['Eager']})")


def deploy_for_mode(mode: str):
    """Rebuild and deploy for a specific mode"""
    print(f"\nDeploying for {mode} mode...")
    
    # Update template
    update_template_for_mode(mode)
    
    # Set environment variable for benchmark
    os.environ['BENCHMARK_EAGER'] = MODE_CONFIGS[mode]['Eager']
    
    # Run build and deploy
    app_dir = Path(__file__).parent.parent
    
    try:
        # Build
        subprocess.run(
            ['unum-cli', 'build'],
            cwd=app_dir,
            check=True,
            capture_output=True
        )
        print("  Build successful")
        
        # Deploy
        subprocess.run(
            ['unum-cli', 'deploy'],
            cwd=app_dir,
            check=True,
            capture_output=True
        )
        print("  Deploy successful")
        
        # Wait for deployment to propagate
        time.sleep(10)
        
    except subprocess.CalledProcessError as e:
        print(f"  Deployment failed: {e}")
        raise


def save_results(summaries: List[BenchmarkSummary], filename: str = None):
    """Save benchmark results to JSON"""
    if filename is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"benchmark_results_{timestamp}.json"
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    filepath = results_dir / filename
    
    # Convert to dict
    data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'summaries': [asdict(s) for s in summaries]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"\nResults saved to {filepath}")
    return filepath


def print_comparison(summaries: List[BenchmarkSummary]):
    """Print comparison table"""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    
    headers = ['Metric', 'CLASSIC', 'FUTURE_BASED', 'Improvement']
    
    classic = next((s for s in summaries if s.mode == 'CLASSIC'), None)
    future = next((s for s in summaries if s.mode == 'FUTURE_BASED'), None)
    
    if not classic or not future:
        print("Missing data for comparison")
        return
    
    def improvement(classic_val, future_val):
        if classic_val == 0:
            return "N/A"
        diff = ((classic_val - future_val) / classic_val) * 100
        return f"{diff:+.1f}%"
    
    metrics = [
        ('E2E Latency (mean)', f"{classic.e2e_latency_mean_ms:.1f}ms", 
         f"{future.e2e_latency_mean_ms:.1f}ms",
         improvement(classic.e2e_latency_mean_ms, future.e2e_latency_mean_ms)),
        ('E2E Latency (min)', f"{classic.e2e_latency_min_ms:.1f}ms", 
         f"{future.e2e_latency_min_ms:.1f}ms",
         improvement(classic.e2e_latency_min_ms, future.e2e_latency_min_ms)),
        ('E2E Latency (max)', f"{classic.e2e_latency_max_ms:.1f}ms", 
         f"{future.e2e_latency_max_ms:.1f}ms",
         improvement(classic.e2e_latency_max_ms, future.e2e_latency_max_ms)),
        ('Total Duration (mean)', f"{classic.total_duration_mean_ms:.1f}ms", 
         f"{future.total_duration_mean_ms:.1f}ms",
         improvement(classic.total_duration_mean_ms, future.total_duration_mean_ms)),
        ('Billed Duration (mean)', f"{classic.billed_duration_mean_ms:.1f}ms", 
         f"{future.billed_duration_mean_ms:.1f}ms",
         improvement(classic.billed_duration_mean_ms, future.billed_duration_mean_ms)),
        ('Cold Start Rate', f"{classic.cold_start_rate*100:.1f}%", 
         f"{future.cold_start_rate*100:.1f}%", ""),
        ('Avg Init Duration', f"{classic.avg_init_duration_ms:.1f}ms", 
         f"{future.avg_init_duration_ms:.1f}ms", ""),
        ('Cost per Run', f"${classic.cost_per_run:.6f}", 
         f"${future.cost_per_run:.6f}",
         improvement(classic.cost_per_run, future.cost_per_run)),
    ]
    
    print(f"\n{headers[0]:<25} {headers[1]:<18} {headers[2]:<18} {headers[3]:<15}")
    print("-" * 80)
    
    for metric in metrics:
        print(f"{metric[0]:<25} {metric[1]:<18} {metric[2]:<18} {metric[3]:<15}")
    
    print("\n" + "="*80)
    print("INVOKER DISTRIBUTION (which branch triggered Aggregator)")
    print("="*80)
    print(f"\nCLASSIC mode: {classic.invoker_distribution}")
    print(f"FUTURE mode:  {future.invoker_distribution}")


def main():
    parser = argparse.ArgumentParser(description='Graph Analysis Benchmark Runner')
    parser.add_argument('--mode', choices=['CLASSIC', 'FUTURE_BASED'], 
                        help='Execution mode to benchmark')
    parser.add_argument('--all', action='store_true', 
                        help='Run both CLASSIC and FUTURE_BASED')
    parser.add_argument('--iterations', type=int, default=3, 
                        help='Number of iterations per mode')
    parser.add_argument('--cold', action='store_true', 
                        help='Force cold starts')
    parser.add_argument('--nodes', type=int, default=DEFAULT_NUM_NODES, 
                        help='Number of graph nodes')
    parser.add_argument('--edge-prob', type=float, default=DEFAULT_EDGE_PROBABILITY, 
                        help='Edge probability')
    parser.add_argument('--skip-deploy', action='store_true', 
                        help='Skip deployment (use current config)')
    parser.add_argument('--profile', default=PROFILE, 
                        help='AWS profile to use')
    
    args = parser.parse_args()
    
    if not args.mode and not args.all:
        parser.error("Either --mode or --all must be specified")
    
    summaries = []
    
    modes_to_run = ['CLASSIC', 'FUTURE_BASED'] if args.all else [args.mode]
    
    for mode in modes_to_run:
        print(f"\n{'#'*60}")
        print(f"# Benchmarking {mode} mode")
        print(f"{'#'*60}")
        
        if not args.skip_deploy:
            deploy_for_mode(mode)
        
        benchmark = GraphAnalysisBenchmark(
            profile=args.profile,
            num_nodes=args.nodes,
            edge_probability=args.edge_prob
        )
        
        summary = benchmark.run_benchmark(
            mode=mode,
            iterations=args.iterations,
            cold_start=args.cold
        )
        
        summaries.append(summary)
    
    # Save results
    save_results(summaries)
    
    # Print comparison if both modes were run
    if args.all:
        print_comparison(summaries)
    
    return summaries


if __name__ == '__main__':
    main()
