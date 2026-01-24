#!/usr/bin/env python3
"""Test CloudWatch logs access"""
import boto3
import time

logs = boto3.client('logs', region_name='eu-west-1')
log_group = '/aws/lambda/unum-mapreduce-wordcount-dynamo-ne-SummaryFunction-Jk5iNof4yqC3'

# Test timestamp - 5 minutes ago
start_ms = int((time.time() - 300) * 1000)
print(f'Searching from {start_ms}')
print(f'Log group: {log_group}')

response = logs.filter_log_events(
    logGroupName=log_group,
    startTime=start_ms,
    filterPattern='REPORT RequestId'
)

events = response.get('events', [])
print(f'Found {len(events)} events')
for e in events[:3]:
    ts = e.get('timestamp', 0)
    msg = e.get('message', '')[:100]
    print(f'  ts={ts}, msg={msg}')
