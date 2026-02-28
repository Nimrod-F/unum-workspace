def lambda_handler(event, context):
    """Function D: slow path: initial processing with heavier computation"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
