#!/usr/bin/env python3
"""
Artificial Delay Benchmark for Image Pipeline
==============================================

This benchmark introduces configurable artificial delays to each branch of the
fan-out/fan-in pipeline to clearly demonstrate the benefits of Future-Based execution.

KEY INSIGHT:
- CLASSIC mode: Fan-in waits for ALL branches → E2E = max(branch_times)
- FUTURE_BASED mode: Fan-in starts with FIRST branch → E2E = min(branch_time) + processing

DELAY SCENARIOS:
1. Uniform: All branches have same delay (baseline)
2. Staggered: Linear increase (0, 1s, 2s, 3s) - simulates realistic variance
3. Extreme: One branch much slower (0, 0, 0, 5s) - highlights maximum benefit
4. Reversed: Slowest computation gets shortest delay - tests adaptation
5. Custom: User-defined delays for each branch

The artificial delay is added via Lambda environment variable (ARTIFICIAL_DELAY_MS)
which the branch functions read and sleep for before completing.

Usage:
    python delay_benchmark.py --scenarios all --iterations 3
    python delay_benchmark.py --scenarios staggered extreme --iterations 5
    python delay_benchmark.py --custom 0,1000,2000,3000 --iterations 3
"""

import boto3
import json
import time
import re
import statistics
import datetime
import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# CONFIGURATION
# ============================================================================

REGION = 'eu-central-1'
PROFILE = 'research-profile'
DYNAMODB_TABLE = 'unum-intermediate-datastore'
LOG_GROUP_PREFIX = '/aws/lambda/'

# Test image
TEST_BUCKET = 'unum-benchmark-images'
TEST_KEY = 'test-images/sample-1920x1080.jpg'

# Branch functions (in order: fastest to slowest natural execution)
BRANCHES = ['Thumbnail', 'Transform', 'Filters', 'Contour']

# Natural execution times (approximate, without artificial delay)
NATURAL_TIMES_MS = {
    'Thumbnail': 80,    # Fastest - simple resize
    'Transform': 120,   # Light - geometric transforms  
    'Filters': 180,     # Medium - blur/sharpen filters
    'Contour': 300,     # Slowest - edge detection
}

# ============================================================================
# DELAY SCENARIOS
# ============================================================================

DELAY_SCENARIOS = {
    'uniform': {
        'name': 'Uniform (Baseline)',
        'description': 'All branches have same delay - establishes baseline',
        'delays': {'Thumbnail': 0, 'Transform': 0, 'Filters': 0, 'Contour': 0}
    },
    'staggered': {
        'name': 'Staggered Delays',
        'description': 'Linear increase: 0s, 1s, 2s, 3s - realistic variance',
        'delays': {'Thumbnail': 0, 'Transform': 1000, 'Filters': 2000, 'Contour': 3000}
    },
    'extreme': {
        'name': 'Extreme Outlier',
        'description': 'One branch 5s slower - highlights maximum benefit',
        'delays': {'Thumbnail': 0, 'Transform': 0, 'Filters': 0, 'Contour': 5000}
    },
    'reversed': {
        'name': 'Reversed Natural Order',
        'description': 'Fastest computation gets longest delay',
        'delays': {'Thumbnail': 3000, 'Transform': 2000, 'Filters': 1000, 'Contour': 0}
    },
    'moderate': {
        'name': 'Moderate Variance',
        'description': 'Small differences: 0, 500ms, 1s, 1.5s',
        'delays': {'Thumbnail': 0, 'Transform': 500, 'Filters': 1000, 'Contour': 1500}
    },
    'bimodal': {
        'name': 'Bimodal Distribution',
        'description': 'Two fast (0s), two slow (2s) branches',
        'delays': {'Thumbnail': 0, 'Transform': 0, 'Filters': 2000, 'Contour': 2000}
    }
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FunctionMetrics:
    """Metrics for a single Lambda function"""
    function_name: str
    duration_ms: float = 0.0
    billed_duration_ms: float = 0.0
    memory_size_mb: int = 0
    max_memory_used_mb: int = 0
    init_duration_ms: float = 0.0
    artificial_delay_ms: float = 0.0
    
    @property
    def is_cold_start(self) -> bool:
        return self.init_duration_ms > 0
    
    @property
    def actual_work_ms(self) -> float:
        """Compute time excluding artificial delay"""
        return max(0, self.duration_ms - self.artificial_delay_ms)


@dataclass 
class BenchmarkRun:
    """Single benchmark run result"""
    run_id: int
    scenario: str
    mode: str  # CLASSIC or FUTURE_BASED
    delays_config: Dict[str, int]
    e2e_latency_ms: float
    invoker_branch: str
    branch_metrics: Dict[str, FunctionMetrics] = field(default_factory=dict)
    cold_starts: int = 0
    error: Optional[str] = None
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    
    @property
    def expected_latency_classic(self) -> float:
        """Expected E2E in CLASSIC = max(branch_total_time)"""
        if not self.branch_metrics:
            return 0
        return max(m.duration_ms for m in self.branch_metrics.values())
    
    @property
    def expected_latency_future(self) -> float:
        """Expected E2E in FUTURE = min(branch_total_time) + Publisher overhead"""
        if not self.branch_metrics:
            return 0
        return min(m.duration_ms for m in self.branch_metrics.values())
    
    @property
    def theoretical_savings_ms(self) -> float:
        """Theoretical time saved by FUTURE_BASED mode"""
        if not self.branch_metrics:
            return 0
        times = [m.duration_ms for m in self.branch_metrics.values()]
        return max(times) - min(times)


@dataclass
class ScenarioResult:
    """Aggregated results for a delay scenario"""
    scenario_name: str
    scenario_description: str
    delays: Dict[str, int]
    classic_runs: List[BenchmarkRun] = field(default_factory=list)
    future_runs: List[BenchmarkRun] = field(default_factory=list)
    
    @property
    def classic_avg_latency(self) -> float:
        valid = [r for r in self.classic_runs if not r.error]
        return statistics.mean([r.e2e_latency_ms for r in valid]) if valid else 0
    
    @property
    def future_avg_latency(self) -> float:
        valid = [r for r in self.future_runs if not r.error]
        return statistics.mean([r.e2e_latency_ms for r in valid]) if valid else 0
    
    @property
    def improvement_ms(self) -> float:
        return self.classic_avg_latency - self.future_avg_latency
    
    @property
    def improvement_pct(self) -> float:
        if self.classic_avg_latency > 0:
            return (self.improvement_ms / self.classic_avg_latency) * 100
        return 0
    
    @property
    def theoretical_max_savings(self) -> float:
        """Maximum theoretical savings based on configured delays"""
        total_times = {b: NATURAL_TIMES_MS[b] + self.delays[b] for b in BRANCHES}
        return max(total_times.values()) - min(total_times.values())


# ============================================================================
# BENCHMARK CLASS
# ============================================================================

class DelayBenchmark:
    """Main benchmark runner"""
    
    def __init__(self):
        self.session = boto3.Session(profile_name=PROFILE, region_name=REGION)
        self.lambda_client = self.session.client('lambda')
        self.logs_client = self.session.client('logs')
        self.functions = self._load_function_arns()
        
    def _load_function_arns(self) -> Dict[str, str]:
        """Load function ARNs from yaml file"""
        yaml_path = Path(__file__).parent.parent / 'function-arn.yaml'
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    
    def _get_function_env(self, func_arn: str) -> Dict[str, str]:
        """Get current environment variables for a function"""
        try:
            config = self.lambda_client.get_function_configuration(FunctionName=func_arn)
            return config.get('Environment', {}).get('Variables', {})
        except Exception:
            return {}
    
    def configure_delays(self, delays: Dict[str, int], verbose: bool = True):
        """Set artificial delay for each branch function"""
        if verbose:
            print("\n  Configuring artificial delays:")
        
        for branch, delay_ms in delays.items():
            func_arn = self.functions.get(branch)
            if not func_arn:
                continue
                
            try:
                # Get current env vars and update delay
                env = self._get_function_env(func_arn)
                env['ARTIFICIAL_DELAY_MS'] = str(delay_ms)
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={'Variables': env}
                )
                
                if verbose:
                    print(f"    {branch}: {delay_ms}ms delay")
                    
            except Exception as e:
                print(f"    ⚠ Failed to configure {branch}: {e}")
        
        # Wait for configurations to propagate
        time.sleep(3)
    
    def configure_mode(self, mode: str, verbose: bool = True):
        """Configure Publisher for CLASSIC or FUTURE_BASED mode"""
        if verbose:
            print(f"\n  Setting execution mode: {mode}")
        
        future_value = 'true' if mode == 'FUTURE_BASED' else 'false'
        
        try:
            env = self._get_function_env(self.functions['Publisher'])
            env['UNUM_FUTURE_BASED'] = future_value
            
            # Ensure other required env vars
            env.setdefault('CHECKPOINT', 'true')
            env.setdefault('DEBUG', 'true')
            env.setdefault('FAAS_PLATFORM', 'aws')
            env.setdefault('GC', 'false')
            env.setdefault('UNUM_INTERMEDIARY_DATASTORE_NAME', DYNAMODB_TABLE)
            env.setdefault('UNUM_INTERMEDIARY_DATASTORE_TYPE', 'dynamodb')
            env.setdefault('EAGER', 'true')
            
            self.lambda_client.update_function_configuration(
                FunctionName=self.functions['Publisher'],
                Environment={'Variables': env}
            )
            
            if verbose:
                print(f"    ✓ Publisher configured for {mode}")
                
        except Exception as e:
            print(f"    ⚠ Failed to configure Publisher: {e}")
        
        time.sleep(3)
    
    def force_cold_starts(self, mode: str):
        """Force cold starts by updating env vars on all functions"""
        print("  Forcing cold starts...")
        
        future_value = 'true' if mode == 'FUTURE_BASED' else 'false'
        trigger = str(time.time())
        
        for func_name, func_arn in self.functions.items():
            try:
                env = self._get_function_env(func_arn)
                env['COLD_START_TRIGGER'] = trigger
                
                if func_name == 'Publisher':
                    env['UNUM_FUTURE_BASED'] = future_value
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={'Variables': env}
                )
            except Exception:
                pass
        
        time.sleep(5)
    
    def invoke_workflow(self) -> Tuple[float, float, float]:
        """Invoke the workflow and return (start_time, end_time, latency_ms)"""
        payload = {
            "Data": {
                "Source": "http",
                "Value": {
                    "bucket": TEST_BUCKET,
                    "key": TEST_KEY
                }
            }
        }
        
        start = time.time()
        response = self.lambda_client.invoke(
            FunctionName=self.functions['ImageLoader'],
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        end = time.time()
        
        if 'FunctionError' in response:
            error = json.loads(response['Payload'].read())
            raise Exception(f"Lambda error: {error}")
        
        return start, end, (end - start) * 1000
    
    def get_function_metrics(self, func_arn: str, start_time: float, 
                             end_time: float, delay_ms: int = 0) -> FunctionMetrics:
        """Extract metrics from CloudWatch logs"""
        func_name = func_arn.split(':')[-1]
        log_group = f"{LOG_GROUP_PREFIX}{func_name}"
        
        metrics = FunctionMetrics(
            function_name=func_name,
            artificial_delay_ms=delay_ms
        )
        
        try:
            response = self.logs_client.filter_log_events(
                logGroupName=log_group,
                startTime=int(start_time * 1000),
                endTime=int(end_time * 1000) + 120000,
                filterPattern='REPORT'
            )
            
            for event in response.get('events', []):
                msg = event['message']
                
                duration = re.search(r'Duration:\s*([\d.]+)\s*ms', msg)
                billed = re.search(r'Billed Duration:\s*([\d.]+)\s*ms', msg)
                mem_size = re.search(r'Memory Size:\s*(\d+)\s*MB', msg)
                max_mem = re.search(r'Max Memory Used:\s*(\d+)\s*MB', msg)
                init = re.search(r'Init Duration:\s*([\d.]+)\s*ms', msg)
                
                if duration:
                    metrics.duration_ms = float(duration.group(1))
                    metrics.billed_duration_ms = float(billed.group(1)) if billed else metrics.duration_ms
                    metrics.memory_size_mb = int(mem_size.group(1)) if mem_size else 128
                    metrics.max_memory_used_mb = int(max_mem.group(1)) if max_mem else 0
                    metrics.init_duration_ms = float(init.group(1)) if init else 0
                    break
                    
        except Exception:
            pass
        
        return metrics
    
    def get_invoker_branch(self, start_time: float, end_time: float) -> str:
        """Determine which branch invoked the Publisher"""
        invoke_patterns = ['invoking Publisher', 'invoking next', 'Successfully claimed']
        skip_patterns = ['already claimed', 'waiting for others', 'Skipping']
        
        invokers = []
        
        for branch in BRANCHES:
            func_arn = self.functions.get(branch)
            if not func_arn:
                continue
            
            func_name = func_arn.split(':')[-1]
            log_group = f"{LOG_GROUP_PREFIX}{func_name}"
            
            try:
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=int(start_time * 1000),
                    endTime=int(end_time * 1000) + 120000,
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
        return 'unknown'
    
    def run_single(self, run_id: int, scenario: str, mode: str, 
                   delays: Dict[str, int], force_cold: bool) -> BenchmarkRun:
        """Execute a single benchmark run"""
        
        if force_cold:
            self.force_cold_starts(mode)
        
        try:
            start, end, latency = self.invoke_workflow()
            
            print(f"    E2E Latency: {latency:.0f}ms")
            
            # Wait for CloudWatch logs
            time.sleep(15)
            
            # Collect metrics for each branch
            branch_metrics = {}
            cold_count = 0
            
            for branch in BRANCHES:
                func_arn = self.functions.get(branch)
                if func_arn:
                    metrics = self.get_function_metrics(
                        func_arn, start, end, delays.get(branch, 0)
                    )
                    branch_metrics[branch] = metrics
                    
                    if metrics.is_cold_start:
                        cold_count += 1
                    
                    print(f"      {branch}: {metrics.duration_ms:.0f}ms "
                          f"(delay: {delays.get(branch, 0)}ms, "
                          f"cold: {metrics.is_cold_start})")
            
            # Identify invoker
            invoker = self.get_invoker_branch(start, end)
            print(f"    Invoker: {invoker}")
            
            return BenchmarkRun(
                run_id=run_id,
                scenario=scenario,
                mode=mode,
                delays_config=delays.copy(),
                e2e_latency_ms=latency,
                invoker_branch=invoker,
                branch_metrics=branch_metrics,
                cold_starts=cold_count,
                start_timestamp=start,
                end_timestamp=end
            )
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return BenchmarkRun(
                run_id=run_id,
                scenario=scenario,
                mode=mode,
                delays_config=delays.copy(),
                e2e_latency_ms=0,
                invoker_branch='',
                error=str(e)
            )
    
    def run_scenario(self, scenario_key: str, iterations: int = 3,
                     force_cold: bool = True) -> ScenarioResult:
        """Run complete benchmark for a delay scenario"""
        scenario = DELAY_SCENARIOS[scenario_key]
        delays = scenario['delays']
        
        print(f"\n{'='*70}")
        print(f"  SCENARIO: {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"{'='*70}")
        print(f"  Delays: {delays}")
        print(f"  Theoretical savings: {self._calc_theoretical_savings(delays):.0f}ms")
        
        result = ScenarioResult(
            scenario_name=scenario['name'],
            scenario_description=scenario['description'],
            delays=delays
        )
        
        # Configure delays once
        self.configure_delays(delays)
        
        # Run CLASSIC mode
        print(f"\n  --- CLASSIC MODE ({iterations} iterations) ---")
        self.configure_mode('CLASSIC')
        
        for i in range(iterations):
            print(f"\n  Run {i+1}/{iterations}:")
            run = self.run_single(i+1, scenario_key, 'CLASSIC', delays, force_cold)
            result.classic_runs.append(run)
            
            if i < iterations - 1:
                time.sleep(2)
        
        # Run FUTURE_BASED mode
        print(f"\n  --- FUTURE_BASED MODE ({iterations} iterations) ---")
        self.configure_mode('FUTURE_BASED')
        
        for i in range(iterations):
            print(f"\n  Run {i+1}/{iterations}:")
            run = self.run_single(i+1, scenario_key, 'FUTURE_BASED', delays, force_cold)
            result.future_runs.append(run)
            
            if i < iterations - 1:
                time.sleep(2)
        
        return result
    
    def _calc_theoretical_savings(self, delays: Dict[str, int]) -> float:
        """Calculate theoretical max savings for a delay config"""
        total_times = {b: NATURAL_TIMES_MS[b] + delays[b] for b in BRANCHES}
        return max(total_times.values()) - min(total_times.values())


# ============================================================================
# REPORTING
# ============================================================================

def print_scenario_summary(result: ScenarioResult):
    """Print summary for a single scenario"""
    print(f"\n  {'='*66}")
    print(f"  {result.scenario_name}")
    print(f"  {'='*66}")
    
    classic_ok = [r for r in result.classic_runs if not r.error]
    future_ok = [r for r in result.future_runs if not r.error]
    
    # Latency comparison
    print(f"\n  {'Metric':<30} {'CLASSIC':>15} {'FUTURE':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    
    if classic_ok:
        classic_avg = statistics.mean([r.e2e_latency_ms for r in classic_ok])
        classic_std = statistics.stdev([r.e2e_latency_ms for r in classic_ok]) if len(classic_ok) > 1 else 0
    else:
        classic_avg, classic_std = 0, 0
        
    if future_ok:
        future_avg = statistics.mean([r.e2e_latency_ms for r in future_ok])
        future_std = statistics.stdev([r.e2e_latency_ms for r in future_ok]) if len(future_ok) > 1 else 0
    else:
        future_avg, future_std = 0, 0
    
    print(f"  {'E2E Latency (avg)':<30} {classic_avg:>12.0f}ms {future_avg:>12.0f}ms")
    print(f"  {'E2E Latency (std)':<30} {classic_std:>12.0f}ms {future_std:>12.0f}ms")
    
    # Improvement
    if classic_avg > 0 and future_avg > 0:
        improvement = classic_avg - future_avg
        improvement_pct = (improvement / classic_avg) * 100
        print(f"\n  Improvement: {improvement:.0f}ms ({improvement_pct:.1f}%)")
        print(f"  Theoretical max: {result.theoretical_max_savings:.0f}ms")
        efficiency = (improvement / result.theoretical_max_savings * 100) if result.theoretical_max_savings > 0 else 0
        print(f"  Efficiency: {efficiency:.0f}% of theoretical maximum")
    
    # Invoker analysis
    print(f"\n  Invoker Distribution:")
    
    classic_invokers = {}
    for r in classic_ok:
        classic_invokers[r.invoker_branch] = classic_invokers.get(r.invoker_branch, 0) + 1
    
    future_invokers = {}
    for r in future_ok:
        future_invokers[r.invoker_branch] = future_invokers.get(r.invoker_branch, 0) + 1
    
    print(f"    CLASSIC: {classic_invokers}")
    print(f"    FUTURE:  {future_invokers}")
    
    # Expected behavior
    delays = result.delays
    expected_slowest = max(delays.items(), key=lambda x: NATURAL_TIMES_MS[x[0]] + x[1])[0]
    expected_fastest = min(delays.items(), key=lambda x: NATURAL_TIMES_MS[x[0]] + x[1])[0]
    
    print(f"\n  Expected behavior:")
    print(f"    CLASSIC should wait for: {expected_slowest} (slowest)")
    print(f"    FUTURE should start with: {expected_fastest} (fastest)")


def print_overall_summary(results: List[ScenarioResult]):
    """Print summary comparing all scenarios"""
    print("\n" + "=" * 80)
    print("  OVERALL BENCHMARK SUMMARY")
    print("  Artificial Delay Benchmark - CLASSIC vs FUTURE_BASED")
    print("=" * 80)
    
    # Header
    print(f"\n  {'Scenario':<25} {'CLASSIC':>12} {'FUTURE':>12} {'Savings':>12} {'%':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    
    for result in results:
        c_avg = result.classic_avg_latency
        f_avg = result.future_avg_latency
        savings = result.improvement_ms
        pct = result.improvement_pct
        
        print(f"  {result.scenario_name:<25} {c_avg:>10.0f}ms {f_avg:>10.0f}ms {savings:>10.0f}ms {pct:>7.1f}%")
    
    # Conclusions
    print("\n" + "-" * 80)
    print("  KEY FINDINGS:")
    print("-" * 80)
    
    if results:
        max_savings = max(results, key=lambda r: r.improvement_ms)
        max_pct = max(results, key=lambda r: r.improvement_pct)
        
        print(f"\n  • Maximum absolute savings: {max_savings.improvement_ms:.0f}ms ({max_savings.scenario_name})")
        print(f"  • Maximum percentage savings: {max_pct.improvement_pct:.1f}% ({max_pct.scenario_name})")
        
        avg_improvement = statistics.mean([r.improvement_pct for r in results if r.improvement_pct > 0])
        print(f"  • Average improvement: {avg_improvement:.1f}%")
    
    print("\n  CONCLUSION:")
    print("  Future-Based execution provides the most benefit when branch execution")
    print("  times vary significantly. The faster branches complete, the sooner the")
    print("  fan-in can begin processing, reducing end-to-end latency.")
    print("=" * 80)


def save_results(results: List[ScenarioResult], filename: str = None):
    """Save results to JSON file"""
    if filename is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"delay_benchmark_{timestamp}.json"
    
    data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'region': REGION,
            'test_image': f's3://{TEST_BUCKET}/{TEST_KEY}',
            'natural_times_ms': NATURAL_TIMES_MS,
        },
        'scenarios': []
    }
    
    for result in results:
        scenario_data = {
            'name': result.scenario_name,
            'description': result.scenario_description,
            'delays': result.delays,
            'classic': {
                'avg_latency_ms': result.classic_avg_latency,
                'runs': [
                    {
                        'e2e_ms': r.e2e_latency_ms,
                        'invoker': r.invoker_branch,
                        'cold_starts': r.cold_starts,
                        'error': r.error,
                        'branch_durations': {
                            b: m.duration_ms for b, m in r.branch_metrics.items()
                        } if r.branch_metrics else {}
                    }
                    for r in result.classic_runs
                ]
            },
            'future': {
                'avg_latency_ms': result.future_avg_latency,
                'runs': [
                    {
                        'e2e_ms': r.e2e_latency_ms,
                        'invoker': r.invoker_branch,
                        'cold_starts': r.cold_starts,
                        'error': r.error,
                        'branch_durations': {
                            b: m.duration_ms for b, m in r.branch_metrics.items()
                        } if r.branch_metrics else {}
                    }
                    for r in result.future_runs
                ]
            },
            'improvement_ms': result.improvement_ms,
            'improvement_pct': result.improvement_pct,
            'theoretical_max_savings_ms': result.theoretical_max_savings
        }
        data['scenarios'].append(scenario_data)
    
    filepath = Path(__file__).parent / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n  ✓ Results saved to {filepath}")
    return filepath


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Artificial Delay Benchmark for Image Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python delay_benchmark.py --scenarios all --iterations 3
  python delay_benchmark.py --scenarios staggered extreme --iterations 5
  python delay_benchmark.py --custom 0,1000,2000,3000 --iterations 3
        """
    )
    
    parser.add_argument('--scenarios', nargs='+', 
                        choices=list(DELAY_SCENARIOS.keys()) + ['all'],
                        default=['staggered'],
                        help='Delay scenarios to run')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Iterations per mode (default: 3)')
    parser.add_argument('--custom', type=str,
                        help='Custom delays: Thumbnail,Transform,Filters,Contour (ms)')
    parser.add_argument('--no-cold', action='store_true',
                        help='Skip forcing cold starts')
    parser.add_argument('--output', type=str,
                        help='Output JSON filename')
    
    args = parser.parse_args()
    
    # Handle custom delays
    if args.custom:
        delays = [int(d) for d in args.custom.split(',')]
        if len(delays) != 4:
            print("Error: --custom requires 4 comma-separated values")
            return
        DELAY_SCENARIOS['custom'] = {
            'name': 'Custom Delays',
            'description': f'User-defined: {args.custom}',
            'delays': dict(zip(BRANCHES, delays))
        }
        scenarios = ['custom']
    elif 'all' in args.scenarios:
        scenarios = list(DELAY_SCENARIOS.keys())
    else:
        scenarios = args.scenarios
    
    # Print header
    print("=" * 80)
    print("  IMAGE PIPELINE - ARTIFICIAL DELAY BENCHMARK")
    print("  Demonstrating Future-Based Execution Benefits")
    print("=" * 80)
    print(f"\n  Test image: s3://{TEST_BUCKET}/{TEST_KEY}")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print(f"  Iterations per mode: {args.iterations}")
    print(f"  Force cold starts: {not args.no_cold}")
    
    # Run benchmarks
    benchmark = DelayBenchmark()
    results = []
    
    for scenario_key in scenarios:
        result = benchmark.run_scenario(
            scenario_key, 
            iterations=args.iterations,
            force_cold=not args.no_cold
        )
        print_scenario_summary(result)
        results.append(result)
    
    # Print overall summary
    if len(results) > 1:
        print_overall_summary(results)
    
    # Save results
    save_results(results, args.output)
    
    # Reset delays to zero
    print("\n  Resetting delays to zero...")
    benchmark.configure_delays({b: 0 for b in BRANCHES}, verbose=False)
    print("  ✓ Cleanup complete")


if __name__ == "__main__":
    main()
