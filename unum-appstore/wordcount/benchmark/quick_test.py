#!/usr/bin/env python3
"""Quick test to verify wordcount workflow execution"""

import boto3
import json
import time
import uuid

REGION = 'eu-central-1'
UNUMMAP0_ARN = 'arn:aws:lambda:eu-central-1:133480914851:function:unum-mapreduce-wordcount-dynamo-n-UnumMap0Function-lHKKKDwYqZJM'
SUMMARY_LOG_GROUP = '/aws/lambda/unum-mapreduce-wordcount-dynamo-ne-SummaryFunction-Jk5iNof4yqC3'

def main():
    client = boto3.client('lambda', region_name=REGION)
    logs = boto3.client('logs', region_name=REGION)
    
    session_id = f'test-{uuid.uuid4().hex[:8]}'
    payload = {'numMappers': 3, 'wordsPerMapper': 10, 'Session': session_id}
    
    print(f"Testing wordcount workflow")
    print(f"  Session: {session_id}")
    print(f"  Scale: 3 mappers × 10 words = 30 total")
    
    start_time = time.time()
    start_ms = int(start_time * 1000)
    
    print(f"\nInvoking UnumMap0...")
    resp = client.invoke(
        FunctionName=UNUMMAP0_ARN,
        InvocationType='Event',
        Payload=json.dumps(payload)
    )
    status = resp['StatusCode']
    print(f"  Status: {status}")
    
    print(f"\nWaiting for Summary completion (start_ms={start_ms})...")
    for i in range(60):  # 120 seconds max
        time.sleep(2)
        try:
            events = logs.filter_log_events(
                logGroupName=SUMMARY_LOG_GROUP,
                startTime=start_ms,
                filterPattern='REPORT RequestId'
            ).get('events', [])
            
            for event in events:
                ts = event.get('timestamp', 0)
                if ts >= start_ms - 1000:  # Allow 1s buffer
                    elapsed = time.time() - start_time
                    print(f"\n✓ Workflow completed in {elapsed:.2f}s")
                    print(f"  Event timestamp: {ts}")
                    print(f"  Message: {event['message'][:100]}...")
                    return True
        except Exception as e:
            if (i + 1) % 10 == 0:
                print(f"  Poll {i+1}: {e}")
        
        if (i + 1) % 10 == 0:
            print(f"  Poll {i+1}: waiting...")
    
    print("\n✗ Timeout after 120 seconds")
    return False

if __name__ == '__main__':
    main()
