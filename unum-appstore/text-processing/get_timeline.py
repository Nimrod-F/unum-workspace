"""Get execution timeline from CloudWatch logs"""
import boto3
from datetime import datetime, timedelta, timezone

REGION = 'eu-central-1'
PROFILE = 'research-profile'

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
logs = session.client('logs')

functions = {
    "UserMention": "/aws/lambda/unum-text-processing-UserMentionFunction-NKUbaGUlbsPU",
    "FindUrl": "/aws/lambda/unum-text-processing-FindUrlFunction-VzYepwSvKTQF",
    "ShortenUrl": "/aws/lambda/unum-text-processing-ShortenUrlFunction-qTo0E1T6b631",
    "CreatePost": "/aws/lambda/unum-text-processing-CreatePostFunction-KEV5tzoYutP2",
    "Publish": "/aws/lambda/unum-text-processing-PublishFunction-s9nRfhQLHTxq"
}

start_time = int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp() * 1000)

print("=== CLASSIC MODE Timeline ===\n")
timestamps = {}

for name, log_group in functions.items():
    try:
        response = logs.filter_log_events(
            logGroupName=log_group,
            startTime=start_time,
            filterPattern="START RequestId"
        )
        events = response.get('events', [])
        if events:
            # Get the most recent START event
            latest = max(e['timestamp'] for e in events)
            timestamps[name] = latest
    except Exception as e:
        print(f"Error for {name}: {e}")

if timestamps:
    base = min(timestamps.values())
    print("Function timings (ms from first function start):\n")
    for name in ["UserMention", "FindUrl", "ShortenUrl", "CreatePost", "Publish"]:
        if name in timestamps:
            diff = timestamps[name] - base
            role = ""
            if name in ["UserMention", "FindUrl"]:
                role = "(parallel start)"
            elif name == "ShortenUrl":
                role = "(Branch 1, step 2)"
            elif name == "CreatePost":
                role = "(FAN-IN aggregator)"
            elif name == "Publish":
                role = "(final)"
            print(f"  {name:15} +{diff:5}ms  {role}")
    
    print("\n" + "="*50)
    print("ANALYSIS:")
    print("="*50)
    
    um = timestamps.get("UserMention", 0)
    su = timestamps.get("ShortenUrl", 0) 
    cp = timestamps.get("CreatePost", 0)
    
    if um and su and cp:
        if cp > su:
            print("\n→ CLASSIC MODE: CreatePost started AFTER ShortenUrl")
            print(f"  CreatePost waited for all branches to complete")
            print(f"  (CreatePost: +{cp-base}ms, ShortenUrl: +{su-base}ms)")
        else:
            print("\n→ FUTURE-BASED MODE: CreatePost started BEFORE/WITH ShortenUrl")
            print(f"  CreatePost was invoked eagerly by first completing branch")
else:
    print("No timestamps found")
