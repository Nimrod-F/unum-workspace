#!/usr/bin/env python3
"""
Quick Benchmark Test - Single Run Per Mode

Tests the current deployment without switching modes.
Useful for verifying the benchmark infrastructure works.

Usage:
    python quick_test.py
    python quick_test.py --iterations 3
"""

import boto3
import json
import time
import re
import argparse
from datetime import datetime, timedelta


REGION = 'eu-west-1'
STACK_NAME = 'progressive-aggregator'


def get_functions():
    """Get Lambda function names from CloudFormation"""
    cf = boto3.client('cloudformation', region_name=REGION)
    response = cf.describe_stack_resources(StackName=STACK_NAME)
    
    functions = {}
    for resource in response['StackResources']:
        if resource['ResourceType'] == 'AWS::Lambda::Function':
            logical = resource['LogicalResourceId']
            physical = resource['PhysicalResourceId']
            if 'FanOut' in logical:
                functions['FanOut'] = physical
            elif 'Source' in logical:
                functions['Source'] = physical
            elif 'Aggregator' in logical:
                functions['Aggregator'] = physical
    
    return functions


def invoke_workflow(lambda_client, fanout_function):
    """Invoke workflow and return start time"""
    session_id = f"quick-test-{int(time.time() * 1000)}"
    
    payload = {
        "Data": {"Source": "http", "Value": {}},
        "Session": session_id
    }
    
    start = time.time()
    lambda_client.invoke(
        FunctionName=fanout_function,
        InvocationType='Event',
        Payload=json.dumps(payload)
    )
    
    return session_id, start


def wait_for_completion(logs_client, aggregator_function, start_time, timeout=60):
    """Wait for COMPLETED log message"""
    log_group = f'/aws/lambda/{aggregator_function}'
    start_ms = int(start_time * 1000)
    
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = logs_client.filter_log_events(
                logGroupName=log_group,
                startTime=start_ms,
                filterPattern='"[AGGREGATOR] COMPLETED"'
            )
            if response.get('events'):
                return True, time.time()
        except:
            pass
        time.sleep(0.5)
    
    return False, time.time()


def get_aggregator_metrics(logs_client, aggregator_function, start_time, end_time):
    """Extract key metrics from Aggregator logs"""
    log_group = f'/aws/lambda/{aggregator_function}'
    
    metrics = {
        'mode': 'UNKNOWN',
        'initially_ready': 0,
        'pre_resolved': 0,
        'total_wait_ms': 0,
    }
    
    try:
        response = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=int(start_time * 1000),
            endTime=int(end_time * 1000) + 60000
        )
        
        for event in response.get('events', []):
            msg = event['message']
            
            # Detect mode
            if 'FUTURE_BASED mode' in msg:
                metrics['mode'] = 'FUTURE_BASED'
            elif 'EAGER mode' in msg or 'LazyInput' in msg:
                metrics['mode'] = 'EAGER'
            elif 'CLASSIC' in msg:
                metrics['mode'] = 'CLASSIC'
            
            # Initial state
            ready_match = re.search(r'Initial state:\s*(\d+)\s*ready', msg)
            if ready_match:
                metrics['initially_ready'] = int(ready_match.group(1))
            
            # Pre-resolved
            pre_match = re.search(r'pre-resolved.*?(\d+)/\d+', msg)
            if pre_match:
                metrics['pre_resolved'] = int(pre_match.group(1))
            
            # Wait times
            wait_match = re.search(r'Received after waiting\s*(\d+)ms', msg)
            if wait_match:
                metrics['total_wait_ms'] += int(wait_match.group(1))
    
    except Exception as e:
        print(f"  Warning: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=1)
    args = parser.parse_args()
    
    print(f"\nQuick Benchmark Test")
    print(f"  Region: {REGION}")
    print(f"  Stack: {STACK_NAME}")
    print(f"  Iterations: {args.iterations}")
    
    # Setup clients
    lambda_client = boto3.client('lambda', region_name=REGION)
    logs_client = boto3.client('logs', region_name=REGION)
    
    # Get functions
    functions = get_functions()
    print(f"  Functions: {list(functions.keys())}")
    
    print("\n" + "=" * 60)
    
    for i in range(args.iterations):
        print(f"\n  Run {i+1}/{args.iterations}")
        print("-" * 40)
        
        # Invoke
        session_id, start_time = invoke_workflow(lambda_client, functions['FanOut'])
        print(f"  Session: {session_id}")
        print(f"  Waiting for completion...")
        
        # Wait
        success, end_time = wait_for_completion(
            logs_client, functions['Aggregator'], start_time
        )
        
        e2e_ms = (end_time - start_time) * 1000
        
        if success:
            print(f"  ✓ Completed in {e2e_ms:.0f}ms")
            
            # Get metrics
            time.sleep(2)  # Wait for log ingestion
            metrics = get_aggregator_metrics(
                logs_client, functions['Aggregator'], start_time, end_time
            )
            
            print(f"  Mode: {metrics['mode']}")
            print(f"  Initially Ready: {metrics['initially_ready']}/5")
            print(f"  Pre-Resolved (bg polling): {metrics['pre_resolved']}/5")
            print(f"  Total Wait Time: {metrics['total_wait_ms']}ms")
        else:
            print(f"  ✗ Timeout after {e2e_ms:.0f}ms")
        
        if i < args.iterations - 1:
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print("  Done!")


if __name__ == '__main__':
    main()
