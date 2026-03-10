"""Quick debug: check last N log events for a Lambda function"""
import boto3, json, sys, time
from datetime import datetime, timezone, timedelta

client = boto3.client('logs', region_name='eu-central-1')

MONTECARLO_FUNCS = {
    "DataGenerator": "montecarlo-pipeline-DataGeneratorFunction-gvwkGVm8kP2e",
    "Transform": "montecarlo-pipeline-TransformFunction-Za86v7RmaX6r",
    "Simulate": "montecarlo-pipeline-SimulateFunction-6NCuJQBPFQmv",
    "Estimate": "montecarlo-pipeline-EstimateFunction-mhUITDfuH4DF",
    "Validate": "montecarlo-pipeline-ValidateFunction-QpakAoHLAHpH",
    "Aggregator": "montecarlo-pipeline-AggregatorFunction-Hh3xknBl44wt",
    "Reporter": "montecarlo-pipeline-ReporterFunction-PDvU2YRhKkqm",
}

WORDCOUNT_FUNCS = {
    "UnumMap0": "unum-mapreduce-wordcount-dynamo-n-UnumMap0Function-OkMfRUTfUjZ2",
    "Mapper": "unum-mapreduce-wordcount-dynamo-new-MapperFunction-9DvHW6XFVZhq",
    "Partition": "unum-mapreduce-wordcount-dynamo--PartitionFunction-5USijpA6yOq2",
    "Reducer": "unum-mapreduce-wordcount-dynamo-ne-ReducerFunction-MnrYs8421v6j",
    "Summary": "unum-mapreduce-wordcount-dynamo-ne-SummaryFunction-PnbfU7OR0fw2",
}

# Check last 5 minutes
start_time = int((datetime.now(timezone.utc) - timedelta(minutes=10)).timestamp() * 1000)

print("=" * 60)
print("MONTECARLO-PIPELINE LOG CHECK")
print("=" * 60)

for name, func in MONTECARLO_FUNCS.items():
    log_group = f"/aws/lambda/{func}"
    print(f"\n--- {name} ({func[:40]}) ---")
    try:
        response = client.filter_log_events(
            logGroupName=log_group,
            startTime=start_time,
            limit=20
        )
        events = response.get('events', [])
        if not events:
            print("  No log events found")
        else:
            print(f"  Found {len(events)} events. Last few:")
            for e in events[-5:]:
                msg = e['message'].strip()[:200]
                ts = datetime.fromtimestamp(e['timestamp']/1000, tz=timezone.utc).strftime('%H:%M:%S')
                print(f"  [{ts}] {msg}")
    except client.exceptions.ResourceNotFoundException:
        print("  Log group not found")
    except Exception as e:
        print(f"  Error: {e}")

print("\n\n" + "=" * 60)
print("WORDCOUNT LOG CHECK")
print("=" * 60)

for name, func in WORDCOUNT_FUNCS.items():
    log_group = f"/aws/lambda/{func}"
    print(f"\n--- {name} ({func[:40]}) ---")
    try:
        response = client.filter_log_events(
            logGroupName=log_group,
            startTime=start_time,
            limit=20
        )
        events = response.get('events', [])
        if not events:
            print("  No log events found")
        else:
            print(f"  Found {len(events)} events. Last few:")
            for e in events[-5:]:
                msg = e['message'].strip()[:200]
                ts = datetime.fromtimestamp(e['timestamp']/1000, tz=timezone.utc).strftime('%H:%M:%S')
                print(f"  [{ts}] {msg}")
    except client.exceptions.ResourceNotFoundException:
        print("  Log group not found")
    except Exception as e:
        print(f"  Error: {e}")
