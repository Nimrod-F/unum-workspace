def lambda_handler(event, context):
    """Function A: receive request and split into fast/slow processing paths"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
