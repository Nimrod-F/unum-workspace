def lambda_handler(event, context):
    """Function A: ingest raw data from source"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
