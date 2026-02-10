#!/usr/bin/env python3
"""
Quick Benchmark - Compare Normal vs Streaming modes

This script:
1. Invokes the workflow
2. Waits for completion by checking Reporter logs
3. Calculates E2E latency from timestamps

Run this after deploying each mode:
- Normal: unum-cli build && unum-cli deploy
- Streaming: unum-cli build --streaming && unum-cli deploy
"""

import boto3
import json
import time
from datetime import datetime
import sys

REGION = "eu-central-1"
LAMBDA = boto3.client('lambda', region_name=REGION)
LOGS = boto3.client('logs', region_name=REGION)

# Function names (update if different)
GENERATOR = "streaming-demo-GeneratorFunction-kwKLWC4pQprl"
REPORTER = "streaming-demo-ReporterFunction-Wa39jU0EnrsQ"


def invoke_workflow():
    """Invoke the Generator and return start time."""
    payload = {"Data": {"Source": "http", "Value": {"input": f"bench_{time.time()}"}}}
    
    start = time.time()
    response = LAMBDA.invoke(
        FunctionName=GENERATOR,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    invoke_time = time.time() - start
    
    return start, invoke_time


def get_reporter_completion(start_time, timeout=60):
    """Wait for Reporter to complete and return its completion timestamp."""
    log_group = f"/aws/lambda/{REPORTER}"
    end_time = start_time + timeout
    
    while time.time() < end_time:
        try:
            # Get latest log stream
            streams = LOGS.describe_log_streams(
                logGroupName=log_group,
                orderBy='LastEventTime',
                descending=True,
                limit=3
            )
            
            for stream in streams.get('logStreams', []):
                events = LOGS.get_log_events(
                    logGroupName=log_group,
                    logStreamName=stream['logStreamName'],
                    startTime=int(start_time * 1000),
                    limit=100
                )
                
                for event in events.get('events', []):
                    msg = event.get('message', '')
                    if '[Reporter] Completed in' in msg:
                        return event['timestamp'] / 1000.0
        except Exception as e:
            pass
        
        time.sleep(1)
    
    return None


def run_benchmark(runs=3, mode="unknown"):
    """Run multiple benchmark iterations."""
    print(f"\n{'='*50}")
    print(f"BENCHMARK: {mode.upper()} MODE")
    print(f"{'='*50}")
    
    latencies = []
    
    for i in range(runs):
        print(f"\n  Run {i+1}/{runs}: ", end="", flush=True)
        
        start_time, invoke_time = invoke_workflow()
        
        # Wait for completion
        completion_time = get_reporter_completion(start_time, timeout=120)
        
        if completion_time:
            e2e = completion_time - start_time
            latencies.append(e2e)
            print(f"E2E = {e2e:.3f}s (invoke={invoke_time:.3f}s)")
        else:
            print("TIMEOUT - Reporter did not complete")
        
        # Wait between runs
        time.sleep(2)
    
    if latencies:
        avg = sum(latencies) / len(latencies)
        min_l = min(latencies)
        max_l = max(latencies)
        print(f"\n  Summary:")
        print(f"    Average E2E: {avg:.3f}s")
        print(f"    Min: {min_l:.3f}s, Max: {max_l:.3f}s")
        return avg
    
    return None


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    run_benchmark(runs, mode)
