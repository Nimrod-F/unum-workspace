#!/usr/bin/env python3
"""Quick test to verify asyncio fix and background polling work."""

import boto3
import time

logs = boto3.client('logs', region_name='eu-central-1')

print("Checking Aggregator logs for background polling...")
end_time = int(time.time() * 1000)
start_time = end_time - 120000  # Last 2 minutes (recent only)

query = '''fields @timestamp, @message
| filter @message like /inputs|BACKGROUND|resolved|ERROR|COMPLETED|RuntimeError|Accessing|INSTANT/
| sort @timestamp desc
| limit 40'''

r = logs.start_query(
    logGroupName='/aws/lambda/progressive-aggregator-AggregatorFunction-z0rG4dhEmMGR',
    startTime=start_time, 
    endTime=end_time, 
    queryString=query
)

time.sleep(5)
results = logs.get_query_results(queryId=r['queryId'])

print(f'\n=== Found {len(results["results"])} log entries ===\n')

has_error = False
has_completed = False
has_background = False
instant_count = 0

for e in results['results']:
    msg = next((f['value'] for f in e if f['field'] == '@message'), '')
    
    if 'ERROR' in msg or 'RuntimeError' in msg:
        has_error = True
        print(f'❌ {msg[:150]}')
    elif 'COMPLETED' in msg:
        has_completed = True
        print(f'✅ {msg[:150]}')
    elif 'BACKGROUND' in msg or 'background' in msg:
        has_background = True
        print(f'🔄 {msg[:150]}')
    elif 'INSTANT' in msg or 'Already resolved' in msg:
        instant_count += 1
        print(f'⚡ {msg[:150]}')
    elif 'Accessing' in msg or 'inputs[' in msg:
        print(f'📊 {msg[:150]}')

print('\n=== SUMMARY ===')
print(f'Completed: {"✅ YES" if has_completed else "❌ NO"}')
print(f'Errors: {"❌ YES" if has_error else "✅ NONE"}')
print(f'Background polling: {"✅ YES" if has_background else "❌ NO"}')
print(f'Instant resolves: {instant_count}')

print('\n' + '='*50)
print('TEST COMPLETE')
print('='*50)
