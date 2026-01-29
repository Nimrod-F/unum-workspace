import boto3
import json
import uuid

session = str(uuid.uuid4())
print(f"Session: {session}")

client = boto3.client('lambda', region_name='eu-central-1')

# Invoke UserMention (branch 0)
user_payload = {
    "Session": session,
    "Fan-out": {"Index": 0},
    "Data": {
        "Source": "http",
        "Value": "Check out this article from @TechNews and @DevCommunity!"
    }
}
response1 = client.invoke(
    FunctionName='unum-text-processing-UserMentionFunction-eebvuSTMmMOf',
    Payload=json.dumps(user_payload)
)
print(f"UserMention Status: {response1['StatusCode']}")

# Invoke FindUrl (needs to trigger ShortenUrl which is branch 1)
find_payload = {
    "Session": session,
    "Fan-out": {"Index": 1},
    "Data": {
        "Source": "http",
        "Value": "Great resource at https://longurl.example.com/article and https://example.com/page"
    }
}
response2 = client.invoke(
    FunctionName='unum-text-processing-FindUrlFunction-BTyUtnuQdujg',
    Payload=json.dumps(find_payload)
)
print(f"FindUrl Status: {response2['StatusCode']}")
print(f"\nWorkflow started for session: {session}")
