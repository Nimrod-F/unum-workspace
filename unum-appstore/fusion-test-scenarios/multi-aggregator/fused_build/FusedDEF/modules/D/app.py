def lambda_handler(event, context):
    """Function D: merge and deduplicate search results"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
