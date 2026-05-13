import boto3
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """@input {bucket: string, numReducer: integer}
    @output {bucket: string, partition: string}"""
    bucket = event[0]['bucket']
    response = s3_client.list_objects(Bucket=bucket, Prefix='', Delimiter='/')
    partitions = [{'bucket': bucket, 'partition': e['Prefix']} for e in response['CommonPrefixes']]
    return partitions
