def lambda_handler(event, context):
    """Function A: receive search query, split into sub-queries"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
