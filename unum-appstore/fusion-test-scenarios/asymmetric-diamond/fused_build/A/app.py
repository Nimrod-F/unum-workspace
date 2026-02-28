def lambda_handler(event, context):
    """Function A: receive data and distribute to enrichment services"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
