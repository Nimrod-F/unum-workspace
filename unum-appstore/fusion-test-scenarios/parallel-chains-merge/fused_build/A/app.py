def lambda_handler(event, context):
    """Function A: receive image, split into processing and metadata paths"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
