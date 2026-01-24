import boto3
import json
import os

sqs_client = boto3.client("sqs")


def lambda_handler(event, context):
	ret = {}
	for d in event:
		ret.update(d)

	# Check if SQS output is requested (passed via environment variable or in original payload)
	sqs_url = os.environ.get('SQS_OUTPUT_URL', '')
	
	if sqs_url:
		# Send final result to SQS
		try:
			message = json.dumps({
				'status': 'completed',
				'wordcount': ret,
				'session': os.environ.get('UNUM_SESSION_ID', 'unknown')
			})
			sqs_client.send_message(QueueUrl=sqs_url, MessageBody=message)
			print(f'[DEBUG] Sent result to SQS: {sqs_url}')
		except Exception as e:
			print(f'[ERROR] Failed to send to SQS: {e}')

	return ret