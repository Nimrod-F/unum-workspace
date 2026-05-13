def lambda_handler(event, context):
    """Format a greeting into a styled response.

    @input {greeting: string, name: string}
    @output {message: string, timestamp: string}
    """
    import datetime

    greeting = event['greeting']
    name = event['name']

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {
        'message': f"[{timestamp}] {greeting}",
        'timestamp': timestamp
    }
