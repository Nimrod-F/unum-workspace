import boto3
import os

sqs_client = boto3.client("sqs")
dynamodb = boto3.resource("dynamodb")


def lambda_handler(event, context):
    if "sqs" in event:
        data = event["data"]
        ret = f'{data} world!'
        # write to sqs
        sqs_client.send_message(QueueUrl=event["sqs"], MessageBody=ret)
    else:
        data = event
    
    result = f'{data} world!'
    
    # Store result in DynamoDB for HTTP API access
    # Get session ID from environment or event context
    session_id = os.environ.get('UNUM_SESSION_ID', 'unknown')
    try:
        # Store in a results table (you'll need to create this)
        table = dynamodb.Table('unum-workflow-results')
        table.put_item(
            Item={
                'sessionId': session_id,
                'result': result,
                'status': 'completed',
                'timestamp': context.request_id
            }
        )
    except Exception as e:
        print(f"Error storing result: {e}")

    return result
