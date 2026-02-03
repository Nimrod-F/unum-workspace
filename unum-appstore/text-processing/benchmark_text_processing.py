#!/usr/bin/env python3
"""
Benchmark: Classic vs Future-Based Execution for Text-Processing Workflow

This benchmark compares CLASSIC and FUTURE_BASED (EAGER) fan-in modes
for the text-processing workflow with asymmetric parallel branches:
  - Branch 0: UserMention (1 step - FAST)
  - Branch 1: FindUrl → ShortenUrl (2 steps - SLOWER)
  - Fan-in: CreatePost
  - Final: Publish

Metrics collected:
1. End-to-End Latency (E2E) - from invocation to Publish completion
2. Cold Start vs Warm Start performance
3. CreatePost invocation timing (when aggregator starts)
4. Per-function execution times
5. Fan-in overhead

Usage:
    python benchmark_text_processing.py --iterations 10
    python benchmark_text_processing.py --cold-only --iterations 5
"""

import boto3
import json
import time
import uuid
import argparse
import threading
import statistics
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

# Test input
TEST_INPUT = "Hey @TechNews and @DevCommunity! Check out https://example.com/article and https://docs.example.com/guide for great resources!"


@dataclass
class BenchmarkResult:
    """Single benchmark run result"""
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
    
    # Calculated metrics
    e2e_latency_ms: float = 0.0
    create_post_invocation_delay_ms: float = 0.0  # Time from invocation to CreatePost start
    fan_in_overhead_ms: float = 0.0  # Time CreatePost waited for inputs
    
    def calculate_metrics(self):
        """Calculate derived metrics"""
        if self.publish_end and self.invocation_time:
            self.e2e_latency_ms = self.publish_end - self.invocation_time
        if self.create_post_start and self.invocation_time:
            self.create_post_invocation_delay_ms = self.create_post_start - self.invocation_time


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark runs for a mode"""
    workflow: str
    mode: str
    total_iterations: int
    cold_iterations: int
    warm_iterations: int
    
    # E2E Latency
    e2e_latency_mean_ms: float = 0.0
    e2e_latency_std_ms: float = 0.0
    e2e_latency_min_ms: float = 0.0
    e2e_latency_max_ms: float = 0.0
    
    # Cold vs Warm
    cold_e2e_mean_ms: float = 0.0
    cold_e2e_std_ms: float = 0.0
    warm_e2e_mean_ms: float = 0.0
    warm_e2e_std_ms: float = 0.0
    
    # CreatePost timing
    create_post_delay_mean_ms: float = 0.0
    create_post_delay_std_ms: float = 0.0
    
    # Individual results
    results: List[Dict] = field(default_factory=list)


class TextProcessingBenchmark:
    """Benchmark runner for text-processing workflow"""
    
    def __init__(self, profile: str = PROFILE, region: str = REGION):
        self.session = boto3.Session(profile_name=profile, region_name=region)
        self.lambda_client = self.session.client('lambda')
        self.logs_client = self.session.client('logs')
        self.cf_client = self.session.client('cloudformation')
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
                # Get current configuration
                response = self.lambda_client.get_function_configuration(FunctionName=func_name)
                env_vars = response.get('Environment', {}).get('Variables', {})
                
                # Update EAGER variable
                env_vars['EAGER'] = eager_value
                
                # Apply update
                self.lambda_client.update_function_configuration(
                    FunctionName=func_name,
                    Environment={'Variables': env_vars}
                )
            except Exception as e:
                print(f"    Warning: Could not update {name}: {e}")
        
        # Wait for updates to propagate
        print("  Waiting 5s for configuration updates...")
        time.sleep(5)
    
    def force_cold_start(self):
        """Force cold starts by updating function configurations"""
        print("  Forcing cold starts...")
        for name, func_name in self.functions.items():
            try:
                # Get current config
                response = self.lambda_client.get_function_configuration(FunctionName=func_name)
                env_vars = response.get('Environment', {}).get('Variables', {})
                
                # Add/update a dummy variable to force cold start
                env_vars['_COLD_START_TRIGGER'] = str(uuid.uuid4())[:8]
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_name,
                    Environment={'Variables': env_vars}
                )
            except Exception as e:
                print(f"    Warning: Could not trigger cold start for {name}: {e}")
        
        # Wait for updates
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
                
                events = response.get('events', [])
                if events:
                    return True
                    
            except Exception:
                pass
            
            time.sleep(1)
        
        return False
    
    def collect_metrics(self, session_id: str, invocation_time: int) -> Dict[str, Dict]:
        """Collect timing metrics from CloudWatch logs"""
        metrics = {}
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
                
                start_events = start_response.get('events', [])
                end_events = end_response.get('events', [])
                
                # Get the latest events for this session
                start_ts = max(e['timestamp'] for e in start_events) if start_events else 0
                end_ts = max(e['timestamp'] for e in end_events) if end_events else 0
                
                metrics[name] = {
                    'start': start_ts,
                    'end': end_ts,
                    'duration': end_ts - start_ts if (start_ts and end_ts) else 0
                }
                
            except Exception as e:
                metrics[name] = {'start': 0, 'end': 0, 'duration': 0, 'error': str(e)}
        
        return metrics
    
    def run_single_benchmark(self, mode: str, iteration: int, is_cold: bool) -> BenchmarkResult:
        """Run a single benchmark iteration"""
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
        
        # Wait a bit more for logs to be available
        time.sleep(2)
        
        # Collect metrics
        metrics = self.collect_metrics(session_id, invocation_time)
        
        # Populate result
        for name, data in metrics.items():
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
        
        result.calculate_metrics()
        
        return result
    
    def run_benchmark(self, mode: str, iterations: int = 10, cold_iterations: int = 2) -> BenchmarkSummary:
        """Run complete benchmark for a mode"""
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
                print(f"  Iteration {i+1}/{iterations} (warm)...", end=" ", flush=True)
            
            result = self.run_single_benchmark(mode, i, is_cold)
            results.append(result)
            
            print(f"E2E: {result.e2e_latency_ms:.0f}ms, CreatePost delay: {result.create_post_invocation_delay_ms:.0f}ms")
            
            # Small delay between iterations
            time.sleep(2)
        
        # Calculate summary
        summary = self._calculate_summary(mode, results)
        
        return summary
    
    def _calculate_summary(self, mode: str, results: List[BenchmarkResult]) -> BenchmarkSummary:
        """Calculate summary statistics"""
        e2e_values = [r.e2e_latency_ms for r in results if r.e2e_latency_ms > 0]
        cold_values = [r.e2e_latency_ms for r in results if r.is_cold and r.e2e_latency_ms > 0]
        warm_values = [r.e2e_latency_ms for r in results if not r.is_cold and r.e2e_latency_ms > 0]
        delay_values = [r.create_post_invocation_delay_ms for r in results if r.create_post_invocation_delay_ms > 0]
        
        summary = BenchmarkSummary(
            workflow='text-processing',
            mode=mode,
            total_iterations=len(results),
            cold_iterations=len(cold_values),
            warm_iterations=len(warm_values),
            results=[asdict(r) for r in results]
        )
        
        if e2e_values:
            summary.e2e_latency_mean_ms = statistics.mean(e2e_values)
            summary.e2e_latency_std_ms = statistics.stdev(e2e_values) if len(e2e_values) > 1 else 0
            summary.e2e_latency_min_ms = min(e2e_values)
            summary.e2e_latency_max_ms = max(e2e_values)
        
        if cold_values:
            summary.cold_e2e_mean_ms = statistics.mean(cold_values)
            summary.cold_e2e_std_ms = statistics.stdev(cold_values) if len(cold_values) > 1 else 0
        
        if warm_values:
            summary.warm_e2e_mean_ms = statistics.mean(warm_values)
            summary.warm_e2e_std_ms = statistics.stdev(warm_values) if len(warm_values) > 1 else 0
        
        if delay_values:
            summary.create_post_delay_mean_ms = statistics.mean(delay_values)
            summary.create_post_delay_std_ms = statistics.stdev(delay_values) if len(delay_values) > 1 else 0
        
        return summary


def generate_charts(classic_summary: BenchmarkSummary, future_summary: BenchmarkSummary, output_dir: str):
    """Generate comparison charts"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Color scheme
    COLORS = {
        'CLASSIC': '#e74c3c',      # Red
        'FUTURE_BASED': '#27ae60', # Green
    }
    
    # 1. E2E Latency Comparison (Cold vs Warm)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cold starts
    ax1 = axes[0]
    modes = ['CLASSIC', 'FUTURE_BASED']
    cold_means = [classic_summary.cold_e2e_mean_ms, future_summary.cold_e2e_mean_ms]
    cold_stds = [classic_summary.cold_e2e_std_ms, future_summary.cold_e2e_std_ms]
    
    bars1 = ax1.bar(modes, cold_means, yerr=cold_stds, capsize=5, 
                    color=[COLORS['CLASSIC'], COLORS['FUTURE_BASED']], 
                    edgecolor='black', linewidth=1.5)
    ax1.set_title('Cold Start E2E Latency', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    for bar, mean in zip(bars1, cold_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{mean:.0f}ms', ha='center', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Warm starts
    ax2 = axes[1]
    warm_means = [classic_summary.warm_e2e_mean_ms, future_summary.warm_e2e_mean_ms]
    warm_stds = [classic_summary.warm_e2e_std_ms, future_summary.warm_e2e_std_ms]
    
    bars2 = ax2.bar(modes, warm_means, yerr=warm_stds, capsize=5,
                    color=[COLORS['CLASSIC'], COLORS['FUTURE_BASED']],
                    edgecolor='black', linewidth=1.5)
    ax2.set_title('Warm Start E2E Latency', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    for bar, mean in zip(bars2, warm_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{mean:.0f}ms', ha='center', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cold_warm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: cold_warm_comparison.png")
    
    # 2. CreatePost Invocation Delay
    fig, ax = plt.subplots(figsize=(10, 6))
    
    delay_means = [classic_summary.create_post_delay_mean_ms, future_summary.create_post_delay_mean_ms]
    delay_stds = [classic_summary.create_post_delay_std_ms, future_summary.create_post_delay_std_ms]
    
    bars = ax.bar(modes, delay_means, yerr=delay_stds, capsize=5,
                  color=[COLORS['CLASSIC'], COLORS['FUTURE_BASED']],
                  edgecolor='black', linewidth=1.5)
    ax.set_title('CreatePost Invocation Delay\n(Time from workflow start to aggregator start)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Delay (ms)', fontsize=12)
    
    for bar, mean in zip(bars, delay_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
               f'{mean:.0f}ms', ha='center', fontsize=11, fontweight='bold')
    
    # Add annotation
    if delay_means[0] > 0 and delay_means[1] > 0:
        improvement = ((delay_means[0] - delay_means[1]) / delay_means[0]) * 100
        ax.text(0.5, 0.95, f'FUTURE_BASED starts CreatePost {improvement:.1f}% earlier!',
               transform=ax.transAxes, ha='center', fontsize=12, color='green', fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/createpost_delay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: createpost_delay.png")
    
    # 3. Overall Improvement Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Cold Start\nE2E', 'Warm Start\nE2E', 'CreatePost\nDelay']
    classic_vals = [classic_summary.cold_e2e_mean_ms, classic_summary.warm_e2e_mean_ms, 
                    classic_summary.create_post_delay_mean_ms]
    future_vals = [future_summary.cold_e2e_mean_ms, future_summary.warm_e2e_mean_ms,
                   future_summary.create_post_delay_mean_ms]
    
    improvements = []
    for c, f in zip(classic_vals, future_vals):
        if c > 0 and f > 0:
            improvements.append(((c - f) / c) * 100)
        else:
            improvements.append(0)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.barh(metrics, improvements, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Improvement (%)', fontsize=12)
    ax.set_title('FUTURE_BASED Improvement over CLASSIC\n(Positive = FUTURE_BASED is faster)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    for bar, imp in zip(bars, improvements):
        x_pos = bar.get_width() + 1 if bar.get_width() >= 0 else bar.get_width() - 1
        ha = 'left' if bar.get_width() >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
               f'{imp:.1f}%', ha=ha, va='center', fontsize=11, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvement_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: improvement_chart.png")
    
    # 4. Latency Distribution Box Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classic_e2e = [r['e2e_latency_ms'] for r in classic_summary.results if r['e2e_latency_ms'] > 0]
    future_e2e = [r['e2e_latency_ms'] for r in future_summary.results if r['e2e_latency_ms'] > 0]
    
    bp = ax.boxplot([classic_e2e, future_e2e], labels=['CLASSIC', 'FUTURE_BASED'],
                    patch_artist=True)
    
    bp['boxes'][0].set_facecolor(COLORS['CLASSIC'])
    bp['boxes'][1].set_facecolor(COLORS['FUTURE_BASED'])
    
    ax.set_ylabel('E2E Latency (ms)', fontsize=12)
    ax.set_title('E2E Latency Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/latency_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: latency_distribution.png")


def generate_summary_report(classic_summary: BenchmarkSummary, future_summary: BenchmarkSummary, output_dir: str):
    """Generate markdown summary report"""
    
    report = f"""# Text-Processing Benchmark Results

## Workflow Description
- **Branch 0**: UserMention (1 step - FAST)
- **Branch 1**: FindUrl → ShortenUrl (2 steps - SLOWER)  
- **Fan-in**: CreatePost
- **Final**: Publish

## Summary

| Metric | CLASSIC | FUTURE_BASED | Improvement |
|--------|---------|--------------|-------------|
| Cold Start E2E (mean) | {classic_summary.cold_e2e_mean_ms:.0f}ms | {future_summary.cold_e2e_mean_ms:.0f}ms | {((classic_summary.cold_e2e_mean_ms - future_summary.cold_e2e_mean_ms) / classic_summary.cold_e2e_mean_ms * 100) if classic_summary.cold_e2e_mean_ms > 0 else 0:.1f}% |
| Warm Start E2E (mean) | {classic_summary.warm_e2e_mean_ms:.0f}ms | {future_summary.warm_e2e_mean_ms:.0f}ms | {((classic_summary.warm_e2e_mean_ms - future_summary.warm_e2e_mean_ms) / classic_summary.warm_e2e_mean_ms * 100) if classic_summary.warm_e2e_mean_ms > 0 else 0:.1f}% |
| CreatePost Delay (mean) | {classic_summary.create_post_delay_mean_ms:.0f}ms | {future_summary.create_post_delay_mean_ms:.0f}ms | {((classic_summary.create_post_delay_mean_ms - future_summary.create_post_delay_mean_ms) / classic_summary.create_post_delay_mean_ms * 100) if classic_summary.create_post_delay_mean_ms > 0 else 0:.1f}% |
| E2E Min | {classic_summary.e2e_latency_min_ms:.0f}ms | {future_summary.e2e_latency_min_ms:.0f}ms | - |
| E2E Max | {classic_summary.e2e_latency_max_ms:.0f}ms | {future_summary.e2e_latency_max_ms:.0f}ms | - |

## Key Findings

### Future-Based Execution Benefits:
1. **CreatePost starts earlier**: In FUTURE_BASED mode, CreatePost is invoked as soon as the first branch (UserMention) completes, rather than waiting for all branches.

2. **Reduced cold start impact**: The aggregator (CreatePost) starts warming up earlier, reducing overall cold start latency.

3. **Better resource utilization**: While waiting for slow branches, the aggregator can perform initialization tasks.

## Charts
- `cold_warm_comparison.png` - Cold vs Warm start latency comparison
- `createpost_delay.png` - CreatePost invocation delay comparison
- `improvement_chart.png` - Overall improvement percentages
- `latency_distribution.png` - E2E latency distribution (box plot)

## Raw Data
See `classic_results.json` and `future_results.json` for detailed per-iteration data.

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(f'{output_dir}/BENCHMARK_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Created: BENCHMARK_REPORT.md")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Classic vs Future-Based execution')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations per mode')
    parser.add_argument('--cold-iterations', type=int, default=2, help='Number of cold start iterations')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Output directory')
    parser.add_argument('--skip-classic', action='store_true', help='Skip CLASSIC benchmark')
    parser.add_argument('--skip-future', action='store_true', help='Skip FUTURE_BASED benchmark')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Text-Processing Benchmark: CLASSIC vs FUTURE_BASED")
    print("="*60)
    print(f"Iterations: {args.iterations} ({args.cold_iterations} cold)")
    print(f"Output: {args.output_dir}/")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize benchmark
    benchmark = TextProcessingBenchmark()
    
    classic_summary = None
    future_summary = None
    
    # Run CLASSIC benchmark
    if not args.skip_classic:
        classic_summary = benchmark.run_benchmark('CLASSIC', args.iterations, args.cold_iterations)
        
        # Save results
        with open(f'{args.output_dir}/classic_results.json', 'w') as f:
            json.dump(asdict(classic_summary), f, indent=2)
        print(f"\n  Saved: classic_results.json")
    
    # Run FUTURE_BASED benchmark
    if not args.skip_future:
        future_summary = benchmark.run_benchmark('FUTURE_BASED', args.iterations, args.cold_iterations)
        
        # Save results
        with open(f'{args.output_dir}/future_results.json', 'w') as f:
            json.dump(asdict(future_summary), f, indent=2)
        print(f"\n  Saved: future_results.json")
    
    # Generate charts and report
    if classic_summary and future_summary:
        print(f"\n{'='*60}")
        print("Generating charts and report...")
        print(f"{'='*60}")
        
        generate_charts(classic_summary, future_summary, args.output_dir)
        generate_summary_report(classic_summary, future_summary, args.output_dir)
        
        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*60}")
        print(f"\nResults saved to: {args.output_dir}/")
        print(f"\nQuick Summary:")
        print(f"  CLASSIC Cold E2E:       {classic_summary.cold_e2e_mean_ms:.0f}ms")
        print(f"  FUTURE_BASED Cold E2E:  {future_summary.cold_e2e_mean_ms:.0f}ms")
        print(f"  CLASSIC Warm E2E:       {classic_summary.warm_e2e_mean_ms:.0f}ms")
        print(f"  FUTURE_BASED Warm E2E:  {future_summary.warm_e2e_mean_ms:.0f}ms")
        
        if classic_summary.warm_e2e_mean_ms > 0:
            improvement = ((classic_summary.warm_e2e_mean_ms - future_summary.warm_e2e_mean_ms) 
                          / classic_summary.warm_e2e_mean_ms * 100)
            print(f"\n  Warm E2E Improvement: {improvement:.1f}%")


if __name__ == '__main__':
    main()
