def lambda_handler(event, context):
    """Function E: call external API (heavy, high memory)"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
