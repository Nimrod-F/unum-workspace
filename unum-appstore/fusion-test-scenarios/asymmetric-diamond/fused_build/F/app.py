def lambda_handler(event, context):
    """Function F: parse and normalize external API response"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
