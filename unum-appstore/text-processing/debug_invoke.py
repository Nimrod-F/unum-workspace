import boto3
import json

client = boto3.client('lambda', region_name='eu-west-1')

payload = {
    "Session": "test-debug-456",
    "Data": {
        "Source": "http",
        "Value": "Hello @TestUser and @AnotherUser"
    }
}

response = client.invoke(
    FunctionName='unum-text-processing-UserMentionFunction-eebvuSTMmMOf',
    Payload=json.dumps(payload)
)

result = json.loads(response['Payload'].read())
print(f"Status: {response['ResponseMetadata']['HTTPStatusCode']}")
print(f"Function Error: {response.get('FunctionError', 'None')}")
print(f"Result: {json.dumps(result, indent=2)}")
