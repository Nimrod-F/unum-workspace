"""
Image Pipeline Benchmark - CLASSIC vs FUTURE Mode Comparison

Tests the real image processing workflow with actual PIL computation.
No artificial delays - all timing differences from real computation.
"""
import json
import time
import subprocess
import re
import sys
from datetime import datetime


# Configuration
STACK_NAME = "image-pipeline"
ENTRY_FUNCTION = "ImageLoader"
PUBLISHER_FUNCTION = "Publisher"
AWS_PROFILE = "research-profile"
AWS_REGION = "eu-central-1"
ITERATIONS = 3
COLD_START_WAIT = 180  # 3 minutes between iterations

# Test image - should be uploaded to S3 beforehand
TEST_BUCKET = "unum-benchmark-images"
TEST_KEY = "test-images/sample-1920x1080.jpg"


def run_aws_command(cmd: list) -> str:
    """Run AWS CLI command and return output"""
    full_cmd = ["aws", "--profile", AWS_PROFILE, "--region", AWS_REGION] + cmd
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"AWS CLI Error: {result.stderr}")
        return ""
    return result.stdout


def invoke_workflow(bucket: str, key: str) -> tuple:
    """Invoke the workflow and return (response, duration_ms)"""
    payload = json.dumps({"bucket": bucket, "key": key})
    
    start = time.time()
    output = run_aws_command([
        "lambda", "invoke",
        "--function-name", f"{STACK_NAME}-{ENTRY_FUNCTION}",
        "--payload", payload,
        "--cli-binary-format", "raw-in-base64-out",
        "/dev/stdout"
    ])
    duration = (time.time() - start) * 1000
    
    try:
        response = json.loads(output.split('\n')[0])
    except:
        response = {"raw": output[:500]}
    
    return response, duration


def get_publisher_logs(start_time: datetime) -> dict:
    """Get CloudWatch logs for the Publisher function"""
    time.sleep(15)  # Wait for logs
    
    log_group = f"/aws/lambda/{STACK_NAME}-{PUBLISHER_FUNCTION}"
    start_ms = int(start_time.timestamp() * 1000)
    
    output = run_aws_command([
        "logs", "filter-log-events",
        "--log-group-name", log_group,
        "--start-time", str(start_ms),
        "--filter-pattern", '?"REPORT" ?"Publisher" ?"FASTEST" ?"SLOWEST"'
    ])
    
    metrics = {
        'invoker': 'unknown',
        'fastest_branch': 'unknown',
        'slowest_branch': 'unknown',
        'variance_ms': 0,
        'billed_duration_ms': 0
    }
    
    if not output:
        return metrics
    
    try:
        logs = json.loads(output)
        for event in logs.get('events', []):
            msg = event.get('message', '')
            
            # Extract invoker (first function to trigger Publisher)
            if 'Thumbnail' in msg and 'FASTEST' in msg:
                metrics['fastest_branch'] = 'Thumbnail'
            if 'Contour' in msg and 'SLOWEST' in msg:
                metrics['slowest_branch'] = 'Contour'
            
            # Extract variance
            variance_match = re.search(r'VARIANCE:\s*(\d+)ms', msg)
            if variance_match:
                metrics['variance_ms'] = int(variance_match.group(1))
            
            # Extract REPORT metrics
            if 'REPORT' in msg:
                billed_match = re.search(r'Billed Duration:\s*(\d+)\s*ms', msg)
                if billed_match:
                    metrics['billed_duration_ms'] = int(billed_match.group(1))
    
    except Exception as e:
        print(f"Log parsing error: {e}")
    
    return metrics


def update_mode(future_mode: bool):
    """Update the Publisher function to use CLASSIC or FUTURE mode"""
    mode = 'FUTURE' if future_mode else 'CLASSIC'
    print(f"\n{'='*60}")
    print(f"Switching to {mode} mode...")
    
    future_value = 'true' if future_mode else 'false'
    
    output = run_aws_command([
        "lambda", "update-function-configuration",
        "--function-name", f"{STACK_NAME}-{PUBLISHER_FUNCTION}",
        "--environment", json.dumps({
            "Variables": {
                "UNUM_INTERMEDIATE_DATASTORE": "unum-intermediate-datastore",
                "EAGER": "true",
                "UNUM_FUTURE_BASED": future_value
            }
        })
    ])
    
    if output:
        print(f"  ✓ {PUBLISHER_FUNCTION} updated to {mode} mode")
    
    time.sleep(10)


def run_benchmark(mode: str, iterations: int) -> list:
    """Run benchmark iterations for given mode"""
    results = []
    
    for i in range(iterations):
        print(f"\n--- {mode} Mode - Iteration {i+1}/{iterations} ---")
        
        start_time = datetime.now()
        print(f"  Invoking workflow at {start_time.strftime('%H:%M:%S')}...")
        
        response, invoke_duration = invoke_workflow(TEST_BUCKET, TEST_KEY)
        print(f"  ✓ Entry function returned in {invoke_duration:.0f}ms")
        
        # Wait for workflow to complete
        print(f"  Waiting for workflow completion...")
        time.sleep(8)
        
        # Get publisher metrics
        print(f"  Fetching Publisher logs...")
        metrics = get_publisher_logs(start_time)
        
        result = {
            'iteration': i + 1,
            'mode': mode,
            'invoke_duration_ms': invoke_duration,
            'variance_ms': metrics['variance_ms'],
            'billed_duration_ms': metrics['billed_duration_ms'],
            'timestamp': start_time.isoformat()
        }
        
        results.append(result)
        print(f"  ✓ Variance: {result['variance_ms']}ms")
        print(f"  ✓ Billed Duration: {result['billed_duration_ms']}ms")
        
        if i < iterations - 1:
            print(f"\n  Waiting {COLD_START_WAIT}s for cold start...")
            time.sleep(COLD_START_WAIT)
    
    return results


def print_summary(classic_results: list, future_results: list):
    """Print benchmark summary"""
    print("\n" + "=" * 70)
    print("IMAGE PIPELINE BENCHMARK RESULTS")
    print("Real PIL Computation - No Artificial Delays")
    print("=" * 70)
    
    print("\nWorkflow Configuration:")
    print("  Branches:")
    print("    - Thumbnail:  resize (FASTEST)")
    print("    - Transform:  rotate/flip")
    print("    - Filters:    blur/sharpen")
    print("    - Contour:    edge detection (SLOWEST)")
    
    # Calculate averages
    classic_variance = sum(r['variance_ms'] for r in classic_results) / len(classic_results) if classic_results else 0
    future_variance = sum(r['variance_ms'] for r in future_results) / len(future_results) if future_results else 0
    
    print(f"\n  Average timing variance: {classic_variance:.0f}ms")
    print(f"  (Contour - Thumbnail = real computation difference)")
    
    print("\n" + "-" * 70)
    print("EXPECTED BEHAVIOR:")
    print("-" * 70)
    print("  CLASSIC: Contour (slowest) triggers Publisher")
    print("  FUTURE:  Thumbnail (fastest) triggers Publisher")
    print(f"  Expected improvement: ~{classic_variance:.0f}ms per execution")


def main():
    print("=" * 70)
    print("IMAGE PIPELINE BENCHMARK")
    print("Real Computation - CLASSIC vs FUTURE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Stack: {STACK_NAME}")
    print(f"  Test image: s3://{TEST_BUCKET}/{TEST_KEY}")
    print(f"  Iterations per mode: {ITERATIONS}")
    
    # Run benchmarks
    update_mode(future_mode=False)
    classic_results = run_benchmark("CLASSIC", ITERATIONS)
    
    update_mode(future_mode=True)
    future_results = run_benchmark("FUTURE", ITERATIONS)
    
    # Print summary
    print_summary(classic_results, future_results)
    
    # Save results
    all_results = {
        'benchmark': 'image-pipeline',
        'timestamp': datetime.now().isoformat(),
        'note': 'Real PIL computation - no artificial delays',
        'classic': classic_results,
        'future': future_results
    }
    
    results_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")


if __name__ == "__main__":
    main()
