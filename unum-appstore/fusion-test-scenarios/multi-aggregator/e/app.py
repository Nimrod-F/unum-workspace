def lambda_handler(event, context):
    """Function E: rank merged results by relevance"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
