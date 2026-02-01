"""
Image Loader - Entry point that downloads image from S3 and fans out.

Downloads the image and passes raw bytes to all processing branches.
"""
import json
import time
import base64
import boto3


s3_client = boto3.client('s3')


def lambda_handler(event, context):
    """
    Download image from S3 and fan-out to processing branches.
    
    Input:
    {
        "bucket": "my-bucket",
        "key": "images/test.jpg"
    }
    
    Output: Image data for parallel processing branches
    """
    start_time = time.time()
    
    bucket = event.get('bucket')
    key = event.get('key')
    
    if not bucket or not key:
        raise ValueError("Missing 'bucket' or 'key' in event")
    
    print(f'[ImageLoader] Downloading s3://{bucket}/{key}')
    
    # Download image from S3
    download_start = time.time()
    response = s3_client.get_object(Bucket=bucket, Key=key)
    image_bytes = response['Body'].read()
    download_time = (time.time() - download_start) * 1000
    
    # Get image metadata
    content_type = response.get('ContentType', 'image/jpeg')
    image_size = len(image_bytes)
    
    print(f'[ImageLoader] Downloaded {image_size} bytes in {download_time:.0f}ms')
    
    # Encode as base64 for passing to other functions
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    total_time = (time.time() - start_time) * 1000
    
    result = {
        'bucket': bucket,
        'key': key,
        'image_b64': image_b64,
        'image_size_bytes': image_size,
        'content_type': content_type,
        'download_time_ms': int(download_time),
        'total_time_ms': int(total_time),
        'timestamp': time.time()
    }
    
    print(f'[ImageLoader] Complete in {total_time:.0f}ms')
    
    return result


if __name__ == '__main__':
    # Local test
    result = lambda_handler({
        'bucket': 'test-bucket',
        'key': 'test.jpg'
    }, None)
    print(f"Downloaded {result['image_size_bytes']} bytes")
