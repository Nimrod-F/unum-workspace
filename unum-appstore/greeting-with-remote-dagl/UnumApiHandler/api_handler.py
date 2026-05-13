import boto3
import json
import os
import time
import uuid

lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

WORKFLOW_NAME = os.environ.get('WORKFLOW_NAME', 'workflow')
START_FUNCTION_ARN = os.environ.get('START_FUNCTION_ARN', '')
DS_TABLE_NAME = os.environ.get('DS_TABLE_NAME', 'unum-intermediate-datastore')
END_FUNCTIONS = json.loads(os.environ.get('END_FUNCTIONS', '[]'))
TIMEOUT = int(os.environ.get('API_TIMEOUT', '25'))


def lambda_handler(event, context):
    """API Gateway handler: invokes workflow and returns final result."""
    # Parse HTTP body
    body = event.get('body', '{}')
    if event.get('isBase64Encoded', False):
        import base64
        body = base64.b64decode(body).decode('utf-8')

    try:
        input_data = json.loads(body) if isinstance(body, str) else body
    except (json.JSONDecodeError, TypeError):
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Invalid JSON in request body'})
        }

    # Create session ID for this request
    session_id = str(uuid.uuid4())

    # Build unum payload envelope
    payload = {
        'Data': {
            'Source': 'http',
            'Value': input_data
        },
        'Session': session_id
    }

    # Invoke the start function (async Event type to trigger the chain)
    try:
        lambda_client.invoke(
            FunctionName=START_FUNCTION_ARN,
            InvocationType='Event',
            Payload=json.dumps(payload).encode('utf-8')
        )
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': f'Failed to invoke workflow: {str(e)}'})
        }

    # Poll DynamoDB for the end function's checkpoint
    table = dynamodb.Table(DS_TABLE_NAME)
    end_keys = [f'{session_id}/{fn}-output' for fn in END_FUNCTIONS]

    start_time = time.time()
    results = {}

    while time.time() - start_time < TIMEOUT:
        for end_key in end_keys:
            if end_key in results:
                continue
            try:
                item = table.get_item(Key={'Name': end_key})
                if 'Item' in item:
                    raw = item['Item'].get('User', item['Item'].get('Value', None))
                    results[end_key] = raw
            except Exception:
                pass

        if len(results) == len(end_keys):
            break

        time.sleep(0.3)

    if not results:
        return {
            'statusCode': 504,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Workflow timeout',
                'session': session_id,
                'message': f'No result after {TIMEOUT}s. Workflow may still be running.'
            })
        }

    # Return the final result
    final_output = {}
    for fn in END_FUNCTIONS:
        key = f'{session_id}/{fn}-output'
        if key in results:
            raw = results[key]
            try:
                final_output[fn] = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                final_output[fn] = raw

    # If single end function, return its result directly
    if len(END_FUNCTIONS) == 1 and END_FUNCTIONS[0] in final_output:
        response_body = final_output[END_FUNCTIONS[0]]
    else:
        response_body = final_output

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(response_body)
    }
