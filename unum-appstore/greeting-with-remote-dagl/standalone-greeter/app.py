def lambda_handler(event, context):
    """Generate a personalized greeting message.

    This is a standalone function deployed independently.
    It can be imported into unum workflows via @import.

    @input {name: string, style: string}
    @output {greeting: string, name: string}
    """
    name = event.get('name', 'World')
    style = event.get('style', 'formal')

    if style == 'casual':
        greeting = f"Hey {name}! What's up?"
    elif style == 'enthusiastic':
        greeting = f"WOW! So great to meet you, {name}!!!"
    else:
        greeting = f"Hello, {name}. Welcome."

    return {
        'greeting': greeting,
        'name': name
    }
