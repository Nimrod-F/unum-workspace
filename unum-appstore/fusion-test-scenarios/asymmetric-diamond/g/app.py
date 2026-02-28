def lambda_handler(event, context):
    """Function G: merge all enriched data sources"""
    import time
    time.sleep(0.1)
    result = event.copy() if isinstance(event, dict) else event
    return result
