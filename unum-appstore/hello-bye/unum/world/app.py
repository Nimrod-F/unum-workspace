import boto3

sqs_client = boto3.client("sqs")


def lambda_handler(event, context):
    if "sqs" in event:
        data = event["data"]
        result = f'{data} world!'
        # Send final output to SQS
        sqs_client.send_message(QueueUrl=event["sqs"], MessageBody=result)
        return result
    else:
        return f'{event} world!'
