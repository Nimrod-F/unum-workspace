def lambda_handler(event, context):
    """Function D: quick cache lookup (lightweight, low memory)"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
