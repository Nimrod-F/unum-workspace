def lambda_handler(event, context):
    """Function F: aggregate results from both processing paths"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
