"""
Simple test to verify Classic vs Future-Based fan-in execution
Directly queries CloudWatch logs after workflow execution
"""

import boto3
import json
import uuid
import time
import threading
from datetime import datetime, timedelta, timezone

# Configuration
REGION = 'eu-central-1'
PROFILE = 'research-profile'
STACK_NAME = 'unum-text-processing'

# Create boto3 session with profile
session = boto3.Session(profile_name=PROFILE, region_name=REGION)
lambda_client = session.client('lambda')
logs_client = session.client('logs')
cf_client = session.client('cloudformation')

def get_function_names():
    """Get the actual Lambda function names from CloudFormation"""
    response = cf_client.describe_stack_resources(StackName=STACK_NAME)
    functions = {}
    for resource in response['StackResources']:
        if resource['ResourceType'] == 'AWS::Lambda::Function':
            logical_id = resource['LogicalResourceId']
            physical_id = resource['PhysicalResourceId']
            if 'UserMention' in logical_id:
                functions['UserMention'] = physical_id
            elif 'FindUrl' in logical_id:
                functions['FindUrl'] = physical_id
            elif 'ShortenUrl' in logical_id:
                functions['ShortenUrl'] = physical_id
            elif 'CreatePost' in logical_id:
                functions['CreatePost'] = physical_id
            elif 'Publish' in logical_id:
                functions['Publish'] = physical_id
    return functions

def invoke_parallel(functions, test_input):
    """Invoke both start functions in parallel"""
    workflow_session = str(uuid.uuid4())
    
    user_mention_payload = {
        "Session": workflow_session,
        "Fan-out": {"Index": 0},
        "Data": {"Source": "http", "Value": test_input}
    }
    
    find_url_payload = {
        "Session": workflow_session,
        "Fan-out": {"Index": 1},
        "Data": {"Source": "http", "Value": test_input}
    }
    
    def invoke(name, func, payload):
        lambda_client.invoke(FunctionName=func, InvocationType='Event', Payload=json.dumps(payload))
        print(f"  {name} invoked")
    
    t1 = threading.Thread(target=invoke, args=('UserMention', functions['UserMention'], user_mention_payload))
    t2 = threading.Thread(target=invoke, args=('FindUrl', functions['FindUrl'], find_url_payload))
    
    start_time = datetime.now(timezone.utc)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    return workflow_session, start_time

def get_execution_times(functions, start_time, wait_seconds=25):
    """Get execution start times from CloudWatch logs"""
    print(f"\nWaiting {wait_seconds}s for workflow and logs...")
    time.sleep(wait_seconds)
    
    end_time = datetime.now(timezone.utc)
    results = {}
    
    for name, func in functions.items():
        log_group = f"/aws/lambda/{func}"
        try:
            response = logs_client.filter_log_events(
                logGroupName=log_group,
                startTime=int(start_time.timestamp() * 1000),
                endTime=int(end_time.timestamp() * 1000),
                filterPattern="START RequestId"
            )
            events = response.get('events', [])
            if events:
                # Get the earliest START event
                earliest = min(e['timestamp'] for e in events)
                results[name] = earliest
        except Exception as e:
            print(f"  Error getting logs for {name}: {e}")
    
    return results

def run_test(test_name, functions, test_input):
    """Run a single test and return timing results"""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    
    session_id, start_time = invoke_parallel(functions, test_input)
    print(f"Session: {session_id}")
    
    times = get_execution_times(functions, start_time)
    
    if not times:
        print("ERROR: No timing data retrieved")
        return None
    
    # Normalize to earliest time
    base = min(times.values())
    normalized = {k: v - base for k, v in times.items()}
    
    print("\nExecution Timeline (ms from first start):")
    for name in ['UserMention', 'FindUrl', 'ShortenUrl', 'CreatePost', 'Publish']:
        if name in normalized:
            print(f"  {name:15} +{normalized[name]:5}ms")
        else:
            print(f"  {name:15} (no data)")
    
    return normalized

def analyze_results(times):
    """Analyze if Future-Based or Classic mode"""
    if not times or 'CreatePost' not in times:
        return "UNKNOWN"
    
    # In Classic mode: CreatePost starts AFTER ShortenUrl
    # In Future-Based mode: CreatePost starts BEFORE or AROUND same time as ShortenUrl completes
    
    cp_start = times.get('CreatePost', 0)
    su_start = times.get('ShortenUrl', 0)
    um_start = times.get('UserMention', 0)
    
    # If CreatePost starts close to when UserMention would finish (within ~1000ms of branch start)
    # that indicates Future-Based mode
    
    print(f"\nAnalysis:")
    print(f"  UserMention started at: +{um_start}ms")
    print(f"  ShortenUrl started at: +{su_start}ms")
    print(f"  CreatePost started at: +{cp_start}ms")
    
    if su_start > 0 and cp_start < su_start + 500:
        # CreatePost started before or shortly after ShortenUrl even started
        print(f"\n  ✓ FUTURE-BASED MODE DETECTED!")
        print(f"    CreatePost started before ShortenUrl even began processing")
        return "FUTURE_BASED"
    elif su_start > 0 and cp_start > su_start + 1000:
        print(f"\n  → CLASSIC MODE DETECTED")
        print(f"    CreatePost waited for ShortenUrl to complete")
        return "CLASSIC"
    else:
        print(f"\n  ? Mode unclear from timing")
        return "UNKNOWN"

def main():
    print("="*60)
    print("Classic vs Future-Based Execution Test")
    print("="*60)
    
    functions = get_function_names()
    print(f"\nFunctions found: {list(functions.keys())}")
    
    test_input = "Hey @TechNews! Check out https://example.com/article"
    
    # Run test
    times = run_test("Execution Mode Detection", functions, test_input)
    mode = analyze_results(times)
    
    print(f"\n{'='*60}")
    print(f"Result: {mode}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
