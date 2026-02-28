def lambda_handler(event, context):
    """Function C: format primary database results"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
