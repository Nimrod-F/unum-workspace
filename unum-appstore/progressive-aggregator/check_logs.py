#!/usr/bin/env python3
"""Check recent Aggregator logs."""

import boto3
import time

logs = boto3.client('logs', region_name='eu-central-1')
log_group = '/aws/lambda/progressive-aggregator-AggregatorFunction-z0rG4dhEmMGR'

query = '''
fields @timestamp, @message
| filter @message like /AGGREGATOR|inputs|Background|Already|Error|ERROR|event loop/
| sort @timestamp desc
| limit 50
'''

response = logs.start_query(
    logGroupName=log_group,
    startTime=int((time.time() - 300) * 1000),
    endTime=int(time.time() * 1000),
    queryString=query
)

print('Waiting for query...')
time.sleep(5)

results = logs.get_query_results(queryId=response['queryId'])
print(f'Found {len(results["results"])} log entries:\n')

for r in results['results'][:40]:
    ts = next((f['value'] for f in r if f['field'] == '@timestamp'), '')
    msg = next((f['value'] for f in r if f['field'] == '@message'), '')
    if msg:
        # Highlight key items
        prefix = '  '
        if 'ERROR' in msg or 'event loop' in msg.lower():
            prefix = '❌ '
        elif 'COMPLETED' in msg:
            prefix = '✅ '
        elif 'Already resolved' in msg or 'INSTANT' in msg:
            prefix = '⚡ '
        elif 'Background polling' in msg:
            prefix = '🔄 '
        print(f'{prefix}{ts[:19]} | {msg[:100]}')
