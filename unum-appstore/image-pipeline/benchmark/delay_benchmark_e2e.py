#!/usr/bin/env python3
"""
Full E2E Delay Benchmark for Image Pipeline
============================================

This benchmark properly measures the FULL end-to-end latency by waiting for
the Publisher function to complete, not just the ImageLoader.

It polls CloudWatch logs to detect when Publisher finishes, giving accurate
E2E measurements that reflect the actual workflow completion time.

Usage:
    python delay_benchmark_e2e.py --scenarios staggered extreme --iterations 3
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
    'Thumbnail': 80,
    'Transform': 120,
    'Filters': 180,
    'Contour': 300,
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
    'moderate': {
        'name': 'Moderate Variance',
        'description': 'Small differences: 0, 500ms, 1s, 1.5s',
        'delays': {'Thumbnail': 0, 'Transform': 500, 'Filters': 1000, 'Contour': 1500}
    },
    # === NEW SCENARIOS ===
    'reversed': {
        'name': 'Reversed (Fastest Slowest)',
        'description': 'Thumbnail slowest, Contour fastest - reverses natural order',
        'delays': {'Thumbnail': 4000, 'Transform': 2000, 'Filters': 1000, 'Contour': 0}
    },
    'bimodal': {
        'name': 'Bimodal (2 Fast, 2 Slow)',
        'description': 'Two groups: fast (0s) and slow (3s)',
        'delays': {'Thumbnail': 0, 'Transform': 0, 'Filters': 3000, 'Contour': 3000}
    },
    'single_slow': {
        'name': 'Single Slow (Transform)',
        'description': 'Only Transform is slow - tests mid-branch delay',
        'delays': {'Thumbnail': 0, 'Transform': 4000, 'Filters': 0, 'Contour': 0}
    },
    'single_slow_filters': {
        'name': 'Single Slow (Filters)',
        'description': 'Only Filters is slow - tests different branch',
        'delays': {'Thumbnail': 0, 'Transform': 0, 'Filters': 4000, 'Contour': 0}
    },
    'thumbnail_slow': {
        'name': 'Thumbnail Slowest',
        'description': 'Naturally fastest branch becomes slowest',
        'delays': {'Thumbnail': 5000, 'Transform': 0, 'Filters': 0, 'Contour': 0}
    },
    'all_equal_delay': {
        'name': 'All Equal (2s each)',
        'description': 'All branches have 2s delay - no variance',
        'delays': {'Thumbnail': 2000, 'Transform': 2000, 'Filters': 2000, 'Contour': 2000}
    },
    'exponential': {
        'name': 'Exponential Growth',
        'description': 'Delays grow exponentially: 0, 500, 1500, 4000',
        'delays': {'Thumbnail': 0, 'Transform': 500, 'Filters': 1500, 'Contour': 4000}
    },
    'three_fast_one_slow': {
        'name': 'Three Fast, One Slow',
        'description': '3 branches at 0s, Contour at 6s - extreme outlier',
        'delays': {'Thumbnail': 0, 'Transform': 0, 'Filters': 0, 'Contour': 6000}
    },
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass 
class BenchmarkRun:
    """Single benchmark run result"""
    run_id: int
    scenario: str
    mode: str
    delays_config: Dict[str, int]
    e2e_latency_ms: float
    publisher_duration_ms: float = 0.0
    branch_durations: Dict[str, float] = field(default_factory=dict)
    invoker_branch: str = ''
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    """Aggregated results for a delay scenario"""
    scenario_name: str
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


# ============================================================================
# BENCHMARK CLASS
# ============================================================================

class E2EDelayBenchmark:
    """Benchmark that measures true E2E latency by waiting for Publisher"""
    
    def __init__(self):
        self.session = boto3.Session(profile_name=PROFILE, region_name=REGION)
        self.lambda_client = self.session.client('lambda')
        self.logs_client = self.session.client('logs')
        self.functions = self._load_function_arns()
        
    def _load_function_arns(self) -> Dict[str, str]:
        yaml_path = Path(__file__).parent.parent / 'function-arn.yaml'
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    
    def _get_function_env(self, func_arn: str) -> Dict[str, str]:
        try:
            config = self.lambda_client.get_function_configuration(FunctionName=func_arn)
            return config.get('Environment', {}).get('Variables', {})
        except Exception:
            return {}
    
    def configure_delays(self, delays: Dict[str, int]):
        """Set artificial delay for each branch function"""
        print("\n  Configuring delays:")
        
        for branch, delay_ms in delays.items():
            func_arn = self.functions.get(branch)
            if not func_arn:
                continue
                
            try:
                env = self._get_function_env(func_arn)
                env['ARTIFICIAL_DELAY_MS'] = str(delay_ms)
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={'Variables': env}
                )
                print(f"    {branch}: {delay_ms}ms")
            except Exception as e:
                print(f"    ⚠ Failed {branch}: {e}")
        
        time.sleep(3)
    
    def configure_mode(self, mode: str):
        """Configure Publisher for CLASSIC or FUTURE_BASED mode"""
        print(f"\n  Mode: {mode}")
        
        future_value = 'true' if mode == 'FUTURE_BASED' else 'false'
        
        try:
            env = self._get_function_env(self.functions['Publisher'])
            env['UNUM_FUTURE_BASED'] = future_value
            env.setdefault('CHECKPOINT', 'true')
            env.setdefault('DEBUG', 'true')
            env.setdefault('FAAS_PLATFORM', 'aws')
            env.setdefault('EAGER', 'true')
            env.setdefault('UNUM_INTERMEDIARY_DATASTORE_NAME', DYNAMODB_TABLE)
            env.setdefault('UNUM_INTERMEDIARY_DATASTORE_TYPE', 'dynamodb')
            
            self.lambda_client.update_function_configuration(
                FunctionName=self.functions['Publisher'],
                Environment={'Variables': env}
            )
            print(f"    ✓ Publisher: {mode}")
        except Exception as e:
            print(f"    ⚠ Failed: {e}")
        
        time.sleep(3)
    
    def wait_for_publisher(self, start_time: float, timeout: float = 120.0) -> Tuple[bool, float, str]:
        """
        Poll CloudWatch logs to detect when Publisher completes.
        Returns (success, duration_ms, log_excerpt)
        """
        publisher_arn = self.functions['Publisher']
        func_name = publisher_arn.split(':')[-1]
        log_group = f"{LOG_GROUP_PREFIX}{func_name}"
        
        poll_interval = 2.0
        start_poll = time.time()
        
        while (time.time() - start_poll) < timeout:
            try:
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=int(start_time * 1000),
                    endTime=int(time.time() * 1000) + 5000,
                    filterPattern='REPORT'
                )
                
                for event in response.get('events', []):
                    msg = event['message']
                    # Parse REPORT line for duration
                    duration_match = re.search(r'Duration:\s*([\d.]+)\s*ms', msg)
                    if duration_match:
                        duration = float(duration_match.group(1))
                        e2e = (event['timestamp'] / 1000) - start_time
                        return True, e2e * 1000, f"Publisher: {duration:.0f}ms"
                        
            except Exception as e:
                pass
            
            time.sleep(poll_interval)
        
        return False, 0, "Timeout"
    
    def get_branch_durations(self, start_time: float) -> Dict[str, float]:
        """Get duration of each branch from CloudWatch logs"""
        durations = {}
        
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
                    endTime=int(time.time() * 1000) + 60000,
                    filterPattern='REPORT'
                )
                
                for event in response.get('events', []):
                    msg = event['message']
                    duration_match = re.search(r'Duration:\s*([\d.]+)\s*ms', msg)
                    if duration_match:
                        durations[branch] = float(duration_match.group(1))
                        break
            except Exception:
                pass
        
        return durations
    
    def get_invoker(self, start_time: float) -> str:
        """Determine which branch invoked Publisher"""
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
                    endTime=int(time.time() * 1000) + 60000,
                )
                
                logs_text = ' '.join([e['message'] for e in response.get('events', [])])
                
                if 'invoking Publisher' in logs_text or 'Successfully claimed' in logs_text:
                    if 'already claimed' not in logs_text:
                        return branch
            except Exception:
                pass
        
        return 'unknown'
    
    def run_single(self, run_id: int, scenario: str, mode: str, 
                   delays: Dict[str, int]) -> BenchmarkRun:
        """Execute single benchmark run with proper E2E measurement"""
        
        # Invoke workflow
        payload = {
            "Data": {
                "Source": "http",
                "Value": {
                    "bucket": TEST_BUCKET,
                    "key": TEST_KEY
                }
            }
        }
        
        print(f"\n  Run {run_id}: ", end='', flush=True)
        
        try:
            start_time = time.time()
            
            # Invoke (async - returns quickly)
            self.lambda_client.invoke(
                FunctionName=self.functions['ImageLoader'],
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            print("invoked, waiting for Publisher...", end='', flush=True)
            
            # Wait for Publisher to complete
            success, e2e_ms, log_msg = self.wait_for_publisher(start_time, timeout=120)
            
            if not success:
                print(f" TIMEOUT")
                return BenchmarkRun(
                    run_id=run_id, scenario=scenario, mode=mode,
                    delays_config=delays, e2e_latency_ms=0, error="Timeout"
                )
            
            print(f" {e2e_ms:.0f}ms")
            
            # Get branch durations
            branch_durations = self.get_branch_durations(start_time)
            invoker = self.get_invoker(start_time)
            
            print(f"    Branches: {branch_durations}")
            print(f"    Invoker: {invoker}")
            
            return BenchmarkRun(
                run_id=run_id,
                scenario=scenario,
                mode=mode,
                delays_config=delays.copy(),
                e2e_latency_ms=e2e_ms,
                branch_durations=branch_durations,
                invoker_branch=invoker
            )
            
        except Exception as e:
            print(f" ERROR: {e}")
            return BenchmarkRun(
                run_id=run_id, scenario=scenario, mode=mode,
                delays_config=delays, e2e_latency_ms=0, error=str(e)
            )
    
    def run_scenario(self, scenario_key: str, iterations: int = 3) -> ScenarioResult:
        """Run complete benchmark for a delay scenario"""
        scenario = DELAY_SCENARIOS[scenario_key]
        delays = scenario['delays']
        
        print(f"\n{'='*70}")
        print(f"  SCENARIO: {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"{'='*70}")
        print(f"  Delays: {delays}")
        
        # Calculate expected times
        expected_classic = max(NATURAL_TIMES_MS[b] + delays[b] for b in BRANCHES)
        expected_future = min(NATURAL_TIMES_MS[b] + delays[b] for b in BRANCHES)
        print(f"  Expected CLASSIC: ~{expected_classic}ms (slowest branch)")
        print(f"  Expected FUTURE: ~{expected_future}ms (fastest branch)")
        
        result = ScenarioResult(
            scenario_name=scenario['name'],
            delays=delays
        )
        
        # Configure delays
        self.configure_delays(delays)
        
        # Run CLASSIC mode
        print(f"\n  --- CLASSIC MODE ---")
        self.configure_mode('CLASSIC')
        
        for i in range(iterations):
            run = self.run_single(i+1, scenario_key, 'CLASSIC', delays)
            result.classic_runs.append(run)
            time.sleep(5)  # Gap between runs
        
        # Run FUTURE_BASED mode
        print(f"\n  --- FUTURE_BASED MODE ---")
        self.configure_mode('FUTURE_BASED')
        
        for i in range(iterations):
            run = self.run_single(i+1, scenario_key, 'FUTURE_BASED', delays)
            result.future_runs.append(run)
            time.sleep(5)
        
        return result


def print_summary(results: List[ScenarioResult]):
    """Print benchmark summary"""
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS: CLASSIC vs FUTURE_BASED")
    print("=" * 80)
    
    print(f"\n  {'Scenario':<25} {'CLASSIC':>12} {'FUTURE':>12} {'Savings':>12} {'%':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    
    for result in results:
        c = result.classic_avg_latency
        f = result.future_avg_latency
        s = result.improvement_ms
        p = result.improvement_pct
        
        print(f"  {result.scenario_name:<25} {c:>10.0f}ms {f:>10.0f}ms {s:>10.0f}ms {p:>7.1f}%")
    
    # Invoker analysis
    print(f"\n  Invoker Analysis:")
    for result in results:
        classic_invokers = [r.invoker_branch for r in result.classic_runs if not r.error]
        future_invokers = [r.invoker_branch for r in result.future_runs if not r.error]
        print(f"    {result.scenario_name}:")
        print(f"      CLASSIC: {classic_invokers}")
        print(f"      FUTURE:  {future_invokers}")
    
    print("\n" + "=" * 80)
    print("  EXPECTED BEHAVIOR:")
    print("  - CLASSIC: Slowest branch (Contour) should invoke Publisher")
    print("  - FUTURE:  Fastest branch (Thumbnail) should invoke Publisher")
    print("=" * 80)


def save_results(results: List[ScenarioResult]):
    """Save results to JSON"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"delay_benchmark_e2e_{timestamp}.json"
    
    data = {
        'timestamp': timestamp,
        'scenarios': []
    }
    
    for result in results:
        scenario_data = {
            'name': result.scenario_name,
            'delays': result.delays,
            'classic': {
                'avg_latency_ms': result.classic_avg_latency,
                'runs': [
                    {
                        'e2e_ms': r.e2e_latency_ms,
                        'invoker': r.invoker_branch,
                        'branches': r.branch_durations,
                        'error': r.error
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
                        'branches': r.branch_durations,
                        'error': r.error
                    }
                    for r in result.future_runs
                ]
            },
            'improvement_ms': result.improvement_ms,
            'improvement_pct': result.improvement_pct
        }
        data['scenarios'].append(scenario_data)
    
    filepath = Path(__file__).parent / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n  ✓ Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Full E2E Delay Benchmark')
    parser.add_argument('--scenarios', nargs='+', 
                        choices=list(DELAY_SCENARIOS.keys()) + ['all'],
                        default=['staggered'])
    parser.add_argument('--iterations', type=int, default=3)
    
    args = parser.parse_args()
    
    if 'all' in args.scenarios:
        scenarios = list(DELAY_SCENARIOS.keys())
    else:
        scenarios = args.scenarios
    
    print("=" * 80)
    print("  E2E DELAY BENCHMARK - Full Pipeline Measurement")
    print("=" * 80)
    print(f"  Scenarios: {', '.join(scenarios)}")
    print(f"  Iterations: {args.iterations}")
    
    benchmark = E2EDelayBenchmark()
    results = []
    
    for scenario_key in scenarios:
        result = benchmark.run_scenario(scenario_key, args.iterations)
        results.append(result)
    
    print_summary(results)
    save_results(results)
    
    # Reset delays
    print("\n  Resetting delays to 0...")
    benchmark.configure_delays({b: 0 for b in BRANCHES})
    print("  ✓ Done")


if __name__ == "__main__":
    main()
