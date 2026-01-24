import boto3
import json
import time

lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')


def lambda_handler(event, context):
    """
    API Gateway handler that invokes Unum workflow and waits for result
    """
    
    # Parse input from API Gateway
    if 'body' in event:
        body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
    else:
        body = {}
    
    # Invoke the Hello function (entry point)
    unum_payload = {
        "Data": {
            "Source": "http",
            "Value": body
        }
    }
    
    response = lambda_client.invoke(
        FunctionName='unum-hello-world-HelloFunction-jVhjUNtjSiME',
        InvocationType='RequestResponse',
        Payload=json.dumps(unum_payload)
    )
    
    result = json.loads(response['Payload'].read())
    
    # Extract session ID from response
    # Result format: [output, session_id, metadata]
    if isinstance(result, list) and len(result) >= 2:
        session_id = result[1]
        
        # Poll for World function completion
        # In production, you'd check DynamoDB or use Step Functions
        # For now, just wait a bit and check CloudWatch or return session
        
        return {
            'statusCode': 202,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': 'Workflow started',
                'sessionId': session_id,
                'pollUrl': f'/status/{session_id}'
            })
        }
    
    return {
        'statusCode': 500,
        'body': json.dumps({'error': 'Unexpected response format'})
    }
