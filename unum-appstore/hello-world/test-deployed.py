"""
Test script for deployed Hello-World Unum application
Tests the actual deployed Lambda functions through AWS
"""
import boto3
import json

# Initialize Lambda client
client = boto3.client('lambda', region_name='eu-central-1')
logs_client = boto3.client('logs', region_name='eu-central-1')

# Function names from function-arn.yaml
HELLO_FUNCTION = 'unum-hello-world-HelloFunction-4Qey0E4uzyno'
WORLD_FUNCTION = 'unum-hello-world-WorldFunction-h2wngJA2wKeD'

def get_recent_logs(function_name, limit=10):
    """Get recent CloudWatch logs for a function"""
    log_group = f"/aws/lambda/{function_name}"
    try:
        # Get the most recent log stream
        streams = logs_client.describe_log_streams(
            logGroupName=log_group,
            orderBy='LastEventTime',
            descending=True,
            limit=1
        )
        
        if streams.get('logStreams'):
            stream_name = streams['logStreams'][0]['logStreamName']
            events = logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=stream_name,
                limit=limit
            )
            return events.get('events', [])
    except Exception as e:
        print(f"Could not fetch logs: {e}")
    return []

def test_deployed_hello():
    """Test the deployed Hello function"""
    print("=" * 60)
    print("Testing deployed Hello function...")
    print("=" * 60)
    
    # Create payload in Unum format
    payload = {
        "Data": {
            "Source": "http",
            "Value": {}
        }
    }
    
    print(f"Function: {HELLO_FUNCTION}")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")
    
    try:
        response = client.invoke(
            FunctionName=HELLO_FUNCTION,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        status_code = response['StatusCode']
        print(f"Status Code: {status_code}")
        
        # Read the response payload
        result = json.loads(response['Payload'].read())
        
        if 'FunctionError' in response:
            print(f"❌ Function Error: {response['FunctionError']}")
            print(f"\nError Response:")
            print(json.dumps(result, indent=2))
            
            # Get error details
            if 'errorMessage' in result:
                print(f"\n❌ Error Message: {result['errorMessage']}")
            if 'errorType' in result:
                print(f"❌ Error Type: {result['errorType']}")
            if 'stackTrace' in result:
                print(f"\n❌ Stack Trace:")
                for line in result['stackTrace']:
                    print(f"   {line}")
            
            # Get CloudWatch logs
            print(f"\n📋 Recent CloudWatch Logs:")
            logs = get_recent_logs(HELLO_FUNCTION)
            for event in logs[-5:]:  # Last 5 log events
                print(f"   {event['message']}")
            
            return False
        
        # Success case
        print(f"\nResponse:")
        print(json.dumps(result, indent=2))
        
        # Unum returns [output, session_id, metadata]
        if isinstance(result, list) and len(result) > 0:
            print(f"\n✓ Output: {result[0]}")
            if len(result) > 1:
                print(f"✓ Session ID: {result[1]}")
            return True
        else:
            print(f"\n⚠ Unexpected response format")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_deployed_world():
    """Test the deployed World function directly"""
    print("\n" + "=" * 60)
    print("Testing deployed World function directly...")
    print("=" * 60)
    
    # World expects the output from Hello
    payload = {
        "Data": {
            "Source": "http",
            "Value": "Hello"  # Hello's output
        }
    }
    
    print(f"Function: {WORLD_FUNCTION}")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")
    
    try:
        response = client.invoke(
            FunctionName=WORLD_FUNCTION,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        status_code = response['StatusCode']
        print(f"Status Code: {status_code}")
        
        # Read the response payload
        result = json.loads(response['Payload'].read())
        
        if 'FunctionError' in response:
            print(f"❌ Function Error: {response['FunctionError']}")
            print(f"\nError Response:")
            print(json.dumps(result, indent=2))
            
            # Get error details
            if 'errorMessage' in result:
                print(f"\n❌ Error Message: {result['errorMessage']}")
            if 'errorType' in result:
                print(f"❌ Error Type: {result['errorType']}")
            if 'stackTrace' in result:
                print(f"\n❌ Stack Trace:")
                for line in result['stackTrace']:
                    print(f"   {line}")
            
            # Get CloudWatch logs
            print(f"\n📋 Recent CloudWatch Logs:")
            logs = get_recent_logs(WORLD_FUNCTION)
            for event in logs[-5:]:  # Last 5 log events
                print(f"   {event['message']}")
            
            return False
        
        # Success case
        print(f"\nResponse:")
        print(json.dumps(result, indent=2))
        
        if isinstance(result, list) and len(result) > 0:
            expected = "Hello world!"
            if result[0] == expected:
                print(f"\n✓ Correct output: {result[0]}")
                return True
            else:
                print(f"\n⚠ Expected '{expected}', got '{result[0]}'")
                return False
        else:
            print(f"\n⚠ Unexpected response format")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Deployed Hello-World Unum Application\n")
    
    # Test Hello (which should automatically invoke World)
    hello_success = test_deployed_hello()
    
    # Test World directly
    world_success = test_deployed_world()
    
    print("\n" + "=" * 60)
    if hello_success and world_success:
        print("✅ All deployed function tests passed!")
    else:
        print("❌ Some tests failed")
        print("\n💡 Tip: Check CloudWatch logs for more details:")
        print(f"   aws logs tail /aws/lambda/{HELLO_FUNCTION} --follow --region eu-central-1")
        print(f"   aws logs tail /aws/lambda/{WORLD_FUNCTION} --follow --region eu-central-1")
    print("=" * 60)