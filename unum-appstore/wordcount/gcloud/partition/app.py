import boto3


# Credentials should be set via environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
s3_client = boto3.client('s3')


def lambda_handler(event, context):
    bucket = event[0]['bucket']
    numReducer = event[0]['numReducer']

    # # get all directories in the bucket that are "reducer" and return them in
    # # a sorted list

    response = s3_client.list_objects(
        Bucket=bucket,
        Prefix='',
        Delimiter='/'
    )

    partitions = [{'bucket': bucket, 'partition': e['Prefix']} for e in response['CommonPrefixes']]

    return partitions