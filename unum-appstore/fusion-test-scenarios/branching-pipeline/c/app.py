def lambda_handler(event, context):
    """Function C: fast path: finalize and prepare for aggregation"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
