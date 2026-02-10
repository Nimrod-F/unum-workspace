#!/usr/bin/env python3
"""
Full Benchmark Suite - Tests Normal vs Streaming modes

This script:
1. Deploys normal mode, runs benchmarks
2. Deploys streaming mode, runs benchmarks  
3. Generates comparison charts

Usage:
    python full_benchmark.py
"""

import subprocess
import json
import os
import sys
import time
from datetime import datetime

WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, cwd=None):
    """Run a shell command and return output."""
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or WORKFLOW_DIR,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
    return result


def deploy_normal_mode():
    """Deploy workflow in normal mode."""
    print("\n[1] Deploying NORMAL mode...")
    
    # Run unum-cli build (without --streaming)
    run_command("python -m unum-cli build", cwd=WORKFLOW_DIR)
    
    # SAM build and deploy
    run_command("sam build")
    run_command("sam deploy --no-confirm-changeset --no-fail-on-empty-changeset")
    
    print("  Waiting 30s for deployment to stabilize...")
    time.sleep(30)


def deploy_streaming_mode():
    """Deploy workflow in streaming mode."""
    print("\n[3] Deploying STREAMING mode...")
    
    # Run unum-cli build with --streaming flag
    run_command("python -m unum-cli build --streaming", cwd=WORKFLOW_DIR)
    
    # SAM build and deploy
    run_command("sam build")
    run_command("sam deploy --no-confirm-changeset --no-fail-on-empty-changeset")
    
    print("  Waiting 30s for deployment to stabilize...")
    time.sleep(30)


def run_benchmark(mode, runs=5, cold_runs=2):
    """Run benchmark for a specific mode."""
    print(f"\n[Running benchmark for {mode.upper()} mode]")
    
    result = run_command(f"python benchmark.py --mode {mode} --runs {runs} --cold-runs {cold_runs}")
    
    # Find the latest results file
    result_files = [f for f in os.listdir(WORKFLOW_DIR) if f.startswith('benchmark_results_')]
    if result_files:
        latest = sorted(result_files)[-1]
        with open(os.path.join(WORKFLOW_DIR, latest)) as f:
            return json.load(f)
    
    return None


def generate_comparison_report(normal_results, streaming_results):
    """Generate a comparison report."""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "workflow": "streaming-demo",
        "description": "4-stage pipeline with 5 items per stage (0.5s each)",
        "expected": {
            "normal_e2e": "~10s (4 stages × 2.5s each, sequential)",
            "streaming_e2e": "~4-5s (parallel with pipeline overlap)"
        },
        "results": {
            "normal": normal_results.get("results", {}).get("current", {}).get("summary", {}),
            "streaming": streaming_results.get("results", {}).get("current", {}).get("summary", {})
        },
        "comparison": {}
    }
    
    # Calculate improvement
    normal_warm = report["results"]["normal"].get("warm", {}).get("avg_latency", 0)
    streaming_warm = report["results"]["streaming"].get("warm", {}).get("avg_latency", 0)
    
    if normal_warm and streaming_warm:
        improvement = ((normal_warm - streaming_warm) / normal_warm) * 100
        report["comparison"]["warm_improvement_percent"] = improvement
        report["comparison"]["speedup"] = normal_warm / streaming_warm
    
    normal_cold = report["results"]["normal"].get("cold", {}).get("avg_latency", 0)
    streaming_cold = report["results"]["streaming"].get("cold", {}).get("avg_latency", 0)
    
    if normal_cold and streaming_cold:
        cold_improvement = ((normal_cold - streaming_cold) / normal_cold) * 100
        report["comparison"]["cold_improvement_percent"] = cold_improvement
    
    # Save report
    report_file = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(WORKFLOW_DIR, report_file), 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nNormal Mode:")
    print(f"  Warm E2E: {normal_warm:.3f}s")
    print(f"  Cold E2E: {normal_cold:.3f}s")
    print(f"\nStreaming Mode:")
    print(f"  Warm E2E: {streaming_warm:.3f}s")
    print(f"  Cold E2E: {streaming_cold:.3f}s")
    print(f"\nImprovement:")
    print(f"  Warm: {report['comparison'].get('warm_improvement_percent', 0):.1f}% faster")
    print(f"  Cold: {report['comparison'].get('cold_improvement_percent', 0):.1f}% faster")
    print(f"  Speedup: {report['comparison'].get('speedup', 1):.2f}x")
    
    return report


def main():
    print("="*60)
    print("STREAMING DEMO - FULL BENCHMARK SUITE")
    print("="*60)
    
    # Deploy and benchmark normal mode
    deploy_normal_mode()
    normal_results = run_benchmark("current", runs=5, cold_runs=2)
    
    # Rename results file
    result_files = [f for f in os.listdir(WORKFLOW_DIR) if f.startswith('benchmark_results_')]
    if result_files:
        latest = sorted(result_files)[-1]
        os.rename(
            os.path.join(WORKFLOW_DIR, latest),
            os.path.join(WORKFLOW_DIR, "normal_benchmark_results.json")
        )
    
    # Deploy and benchmark streaming mode
    deploy_streaming_mode()
    streaming_results = run_benchmark("current", runs=5, cold_runs=2)
    
    # Rename results file
    result_files = [f for f in os.listdir(WORKFLOW_DIR) if f.startswith('benchmark_results_')]
    if result_files:
        latest = sorted(result_files)[-1]
        os.rename(
            os.path.join(WORKFLOW_DIR, latest),
            os.path.join(WORKFLOW_DIR, "streaming_benchmark_results.json")
        )
    
    # Load results and generate comparison
    with open(os.path.join(WORKFLOW_DIR, "normal_benchmark_results.json")) as f:
        normal_data = json.load(f)
    with open(os.path.join(WORKFLOW_DIR, "streaming_benchmark_results.json")) as f:
        streaming_data = json.load(f)
    
    generate_comparison_report(normal_data, streaming_data)
    
    print("\n[Complete!]")
    print("Run 'python generate_charts.py' to create visualization charts.")


if __name__ == "__main__":
    main()
