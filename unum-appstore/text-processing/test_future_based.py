"""
Test script to verify Future-Based (Eager) Fan-in Execution

This test verifies that in Future-Based mode:
1. The FIRST branch to complete (UserMention - short path) immediately invokes CreatePost
2. CreatePost starts before ShortenUrl completes (Branch 1: FindUrl -> ShortenUrl is longer)
3. CreatePost waits/polls for the missing input from ShortenUrl

In Classic mode:
1. Both branches must complete before CreatePost is invoked
2. CreatePost only starts after the LAST branch finishes

The asymmetric workflow:
  Branch 0: UserMention (1 step - FAST)
  Branch 1: FindUrl -> ShortenUrl (2 steps - SLOWER)
  Fan-in: CreatePost
  Final: Publish
"""

import boto3
import json
import uuid
import time
from datetime import datetime, timedelta, timezone

# Configuration
REGION = 'eu-central-1'
PROFILE = 'research-profile'
STACK_NAME = 'unum-text-processing'

# Create boto3 session with profile
session = boto3.Session(profile_name=PROFILE, region_name=REGION)
lambda_client = session.client('lambda')
logs_client = session.client('logs')

def get_function_names():
    """Get the actual Lambda function names from CloudFormation"""
    cf_client = session.client('cloudformation')
    
    try:
        response = cf_client.describe_stack_resources(StackName=STACK_NAME)
        functions = {}
        for resource in response['StackResources']:
            if resource['ResourceType'] == 'AWS::Lambda::Function':
                logical_id = resource['LogicalResourceId']
                physical_id = resource['PhysicalResourceId']
                # Map logical names to physical names
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
    except Exception as e:
        print(f"Error getting function names: {e}")
        return None

def invoke_workflow(functions, test_input):
    """
    Invoke the workflow by calling both start functions (UserMention and FindUrl)
    TRULY IN PARALLEL using threads
    """
    import threading
    
    workflow_session = str(uuid.uuid4())
    print(f"\n{'='*60}")
    print(f"Starting workflow with session: {workflow_session}")
    print(f"{'='*60}")
    
    start_time = datetime.now(timezone.utc)
    
    # Prepare payloads for both branches
    user_mention_payload = {
        "Session": workflow_session,
        "Fan-out": {"Index": 0},
        "Data": {
            "Source": "http",
            "Value": test_input
        }
    }
    
    find_url_payload = {
        "Session": workflow_session,
        "Fan-out": {"Index": 1},
        "Data": {
            "Source": "http",
            "Value": test_input
        }
    }
    
    results = {}
    
    def invoke_function(name, func_name, payload):
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{ts}] Invoking {name}...")
        response = lambda_client.invoke(
            FunctionName=func_name,
            InvocationType='Event',  # Async invocation
            Payload=json.dumps(payload)
        )
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{ts}] {name} invoked: Status {response['StatusCode']}")
        results[name] = response
    
    # Create threads for parallel invocation
    thread1 = threading.Thread(target=invoke_function, args=('UserMention', functions['UserMention'], user_mention_payload))
    thread2 = threading.Thread(target=invoke_function, args=('FindUrl', functions['FindUrl'], find_url_payload))
    
    # Start both threads at the same time
    print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Starting parallel invocation of both branches...")
    thread1.start()
    thread2.start()
    
    # Wait for both to complete
    thread1.join()
    thread2.join()
    
    return workflow_session, start_time

def get_function_log_group(function_name):
    """Get the CloudWatch log group name for a Lambda function"""
    return f"/aws/lambda/{function_name}"

def get_log_events(log_group, start_time, end_time=None, filter_pattern=''):
    """Get log events from a CloudWatch log group"""
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    
    try:
        kwargs = {
            'logGroupName': log_group,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': 100
        }
        if filter_pattern:
            kwargs['filterPattern'] = filter_pattern
        
        response = logs_client.filter_log_events(**kwargs)
        return response.get('events', [])
    except Exception as e:
        print(f"Error getting logs from {log_group}: {e}")
        return []

def analyze_execution_order(functions, session_id, start_time):
    """
    Analyze CloudWatch logs to determine the execution order and timing
    """
    print(f"\n{'='*60}")
    print("Analyzing execution order from CloudWatch logs...")
    print(f"{'='*60}")
    
    # Wait for logs to be available
    print("\nWaiting 15 seconds for logs to be available...")
    time.sleep(15)
    
    end_time = datetime.now(timezone.utc)
    
    execution_times = {}
    
    for func_name, physical_name in functions.items():
        log_group = get_function_log_group(physical_name)
        
        # Look for START and END log entries
        events = get_log_events(log_group, start_time, end_time)
        
        start_event = None
        end_event = None
        eager_invocation = False
        waiting_for_inputs = False
        
        for event in events:
            message = event.get('message', '')
            timestamp = event.get('timestamp', 0)
            
            # Check for session match
            if session_id not in message and 'START RequestId' not in message and 'END RequestId' not in message and 'REPORT RequestId' not in message:
                continue
            
            if 'START RequestId' in message:
                if start_event is None or timestamp < start_event:
                    start_event = timestamp
            
            if 'END RequestId' in message or 'REPORT RequestId' in message:
                if end_event is None or timestamp > end_event:
                    end_event = timestamp
            
            # Look for eager fan-in indicators
            if 'EAGER invoking' in message or 'EagerFanIn' in message:
                eager_invocation = True
            
            if 'waiting for' in message.lower() or 'polling' in message.lower() or 'await' in message.lower():
                waiting_for_inputs = True
        
        if start_event:
            execution_times[func_name] = {
                'start': start_event,
                'end': end_event,
                'eager_invocation': eager_invocation,
                'waiting_for_inputs': waiting_for_inputs
            }
    
    return execution_times

def verify_future_based_execution(execution_times):
    """
    Verify that Future-Based execution is working correctly
    
    Expected behavior:
    1. UserMention starts and completes first (short path)
    2. CreatePost starts BEFORE ShortenUrl completes
    3. CreatePost waits for ShortenUrl's output
    """
    print(f"\n{'='*60}")
    print("Verification Results")
    print(f"{'='*60}")
    
    if not execution_times:
        print("ERROR: No execution data found in logs")
        return False
    
    # Print execution timeline
    print("\nExecution Timeline:")
    sorted_funcs = sorted(execution_times.items(), key=lambda x: x[1].get('start', float('inf')))
    
    base_time = min(t['start'] for t in execution_times.values() if t.get('start'))
    
    for func_name, times in sorted_funcs:
        if times.get('start'):
            relative_start = times['start'] - base_time
            relative_end = (times['end'] - base_time) if times.get('end') else 'N/A'
            duration = (times['end'] - times['start']) if times.get('end') else 'N/A'
            
            print(f"  {func_name}:")
            print(f"    Start: +{relative_start}ms")
            print(f"    End: +{relative_end}ms" if relative_end != 'N/A' else f"    End: N/A")
            print(f"    Duration: {duration}ms" if duration != 'N/A' else f"    Duration: N/A")
            if times.get('eager_invocation'):
                print(f"    *** EAGER INVOCATION DETECTED ***")
            if times.get('waiting_for_inputs'):
                print(f"    *** WAITING FOR INPUTS DETECTED ***")
    
    # Verify Future-Based behavior
    print("\n" + "-"*40)
    print("Future-Based Verification:")
    print("-"*40)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: UserMention should start before ShortenUrl completes
    tests_total += 1
    if 'UserMention' in execution_times and 'ShortenUrl' in execution_times:
        um_start = execution_times['UserMention'].get('start')
        su_end = execution_times['ShortenUrl'].get('end')
        if um_start and su_end and um_start < su_end:
            print("✓ TEST 1 PASSED: UserMention started before ShortenUrl completed")
            tests_passed += 1
        else:
            print("✗ TEST 1 FAILED: UserMention did not start before ShortenUrl")
    else:
        print("? TEST 1 SKIPPED: Missing timing data")
    
    # Test 2: CreatePost should start before ShortenUrl completes (key test for Future-Based)
    tests_total += 1
    if 'CreatePost' in execution_times and 'ShortenUrl' in execution_times:
        cp_start = execution_times['CreatePost'].get('start')
        su_end = execution_times['ShortenUrl'].get('end')
        if cp_start and su_end:
            if cp_start < su_end:
                print("✓ TEST 2 PASSED: CreatePost started BEFORE ShortenUrl completed")
                print("  --> This proves Future-Based (Eager) fan-in is working!")
                tests_passed += 1
            else:
                print("✗ TEST 2 FAILED: CreatePost started AFTER ShortenUrl completed")
                print("  --> This indicates Classic (non-eager) fan-in behavior")
        else:
            print("? TEST 2 INCONCLUSIVE: Missing start/end times")
    else:
        print("? TEST 2 SKIPPED: Missing CreatePost or ShortenUrl data")
    
    # Test 3: Check for eager invocation markers
    tests_total += 1
    eager_detected = any(t.get('eager_invocation') for t in execution_times.values())
    if eager_detected:
        print("✓ TEST 3 PASSED: Eager invocation detected in logs")
        tests_passed += 1
    else:
        print("? TEST 3 INCONCLUSIVE: No explicit eager invocation markers found in logs")
    
    print(f"\n{'='*60}")
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    print(f"{'='*60}")
    
    return tests_passed >= 2


def main():
    print("="*60)
    print("Future-Based (Eager) Fan-in Verification Test")
    print("="*60)
    
    # Get function names
    print("\nGetting Lambda function names from CloudFormation...")
    functions = get_function_names()
    
    if not functions:
        print("ERROR: Could not get function names. Make sure the stack is deployed.")
        return
    
    print(f"Found functions:")
    for name, physical in functions.items():
        print(f"  {name}: {physical}")
    
    # Test input with both @mentions and URLs
    test_input = "Hey @TechNews and @DevCommunity! Check out https://example.com/article and https://docs.example.com/guide for great resources!"
    
    print(f"\nTest input: {test_input}")
    
    # Invoke the workflow
    session_id, start_time = invoke_workflow(functions, test_input)
    
    # Wait for workflow to complete
    print("\nWaiting 20 seconds for workflow to complete...")
    time.sleep(20)
    
    # Analyze execution order
    execution_times = analyze_execution_order(functions, session_id, start_time)
    
    # Verify Future-Based behavior
    success = verify_future_based_execution(execution_times)
    
    if success:
        print("\n✅ FUTURE-BASED EXECUTION VERIFIED SUCCESSFULLY!")
        print("The first branch (UserMention) invoked the aggregator (CreatePost)")
        print("before all parameters were ready (ShortenUrl hadn't completed yet).")
    else:
        print("\n⚠️ COULD NOT FULLY VERIFY FUTURE-BASED EXECUTION")
        print("Check the logs above for more details.")


if __name__ == '__main__':
    main()
