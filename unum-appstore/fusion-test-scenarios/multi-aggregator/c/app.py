def lambda_handler(event, context):
    """Function C: search database beta"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
