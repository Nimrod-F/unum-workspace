import boto3
import json
import time

lambda_client = boto3.client('lambda')
logs_client = boto3.client('logs')


def lambda_handler(event, context):
    """
    Synchronous API wrapper for Unum workflow
    Invokes Hello, waits briefly, then invokes World to get final result
    """
    
    # Parse input
    if 'body' in event:
        body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
    else:
        body = {}
    
    # Step 1: Invoke Hello function
    unum_payload = {
        "Data": {
            "Source": "http",
            "Value": body
        }
    }
    
    hello_response = lambda_client.invoke(
        FunctionName='unum-hello-world-HelloFunction-jVhjUNtjSiME',
        InvocationType='RequestResponse',
        Payload=json.dumps(unum_payload)
    )
    
    hello_result = json.loads(hello_response['Payload'].read())
    
    # Extract output and session from Hello
    if isinstance(hello_result, list) and len(hello_result) >= 2:
        hello_output = hello_result[0]
        session_id = hello_result[1]
        
        # Step 2: Wait briefly for World to be invoked
        time.sleep(0.5)
        
        # Step 3: World was invoked automatically, but we need to get its output
        # Option A: Invoke World again with same session (idempotent due to checkpointing)
        world_payload = {
            "Data": {
                "Source": "http",
                "Value": hello_output
            },
            "Session": session_id
        }
        
        world_response = lambda_client.invoke(
            FunctionName='unum-hello-world-WorldFunction-97sQbS3CpLey',
            InvocationType='RequestResponse',
            Payload=json.dumps(world_payload)
        )
        
        world_result = json.loads(world_response['Payload'].read())
        
        # Extract final output
        final_output = world_result[0] if isinstance(world_result, list) else world_result
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'result': final_output,
                'sessionId': session_id
            })
        }
    
    return {
        'statusCode': 500,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'error': 'Workflow execution failed'})
    }
