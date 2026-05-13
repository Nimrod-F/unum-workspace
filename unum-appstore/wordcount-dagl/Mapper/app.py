from user_map import user_map
import mapreduce as mr

def lambda_handler(event, context):
    """@input {text: string, destination: string}
    @output {bucket: string, numReducer: integer}"""
    text = event['text']
    bucket = event['destination']

    mr.destinationBucket = bucket

    user_map(text)

    return {
        "bucket": mr.destinationBucket,
        "numReducer": mr.numReducer
    }
