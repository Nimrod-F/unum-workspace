def lambda_handler(event, context):
    """Function F: analyze image content with ML model"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
