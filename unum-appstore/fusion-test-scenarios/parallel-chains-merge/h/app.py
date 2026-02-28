def lambda_handler(event, context):
    """Function H: combine processed image with enriched metadata"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
