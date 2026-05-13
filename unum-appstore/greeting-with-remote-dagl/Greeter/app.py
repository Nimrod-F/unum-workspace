import boto3
import json
import os

_client = boto3.client('lambda')
_REMOTE_ARN = os.environ.get('REMOTE_FUNCTION_ARN', 'arn:aws:lambda:eu-central-1:683003725669:function:standalone-greeter-GreeterFunction-glMvED1OQi7O')


def lambda_handler(event, context):
    """Proxy: invokes the remote function and returns its result."""
    response = _client.invoke(
        FunctionName=_REMOTE_ARN,
        InvocationType='RequestResponse',
        Payload=json.dumps(event),
    )
    payload = response['Payload'].read()
    return json.loads(payload)
