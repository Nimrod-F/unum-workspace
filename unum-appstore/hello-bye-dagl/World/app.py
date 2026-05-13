def lambda_handler(event, context):
    results = event if isinstance(event, list) else [event]
    combined = " and ".join(str(r) for r in results)
    return f'{combined} world!'
