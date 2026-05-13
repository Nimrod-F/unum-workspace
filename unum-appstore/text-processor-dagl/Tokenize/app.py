import re
from collections import Counter


def lambda_handler(event, context):
    """Tokenize input text into words with frequency counts.

    @input {text: string, language: string}
    @output {words: array, count: integer, language: string}
    """
    text = event['text']
    language = event.get('language', 'en')

    # Simple tokenization: split on non-alphanumeric characters
    words = re.findall(r'\b\w+\b', text.lower())

    return {
        'words': words,
        'count': len(words),
        'language': language
    }
