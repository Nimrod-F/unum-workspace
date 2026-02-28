def lambda_handler(event, context):
    """Function F: format output for target system"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
