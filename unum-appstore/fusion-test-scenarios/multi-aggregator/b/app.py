def lambda_handler(event, context):
    """Function B: search database alpha"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
