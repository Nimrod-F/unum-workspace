import uuid
import hashlib
import os
import boto3
s3_client = boto3.client('s3')

numReducer = 3
destinationBucket = ""


def emitPerReducerSingle(word):
    reducerId = int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16) % numReducer
    fileUuid = uuid.uuid4()
    objKey = f'reducer{reducerId}/{word}/{fileUuid}'
    local_file_path = f'/tmp/{fileUuid}.tmp'
    with open(local_file_path, 'w'):
        os.utime(local_file_path, None)
    s3_client.upload_file(local_file_path, destinationBucket, objKey)


def readPerReducerSingle(bucket, partition):
    response = s3_client.list_objects(Bucket=bucket, Prefix=partition)
    data = [e['Key'] for e in response['Contents']]
    data = [e.split('/')[1:] for e in data]
    ret = {}
    for e in data:
        if e[0] in ret:
            ret[e[0]].append(1)
        else:
            ret[e[0]] = [1]
    return ret
