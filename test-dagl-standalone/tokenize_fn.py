"""Tokenize function — splits text into words.

@input {text: string, language: string}
@output {words: array, count: integer, language: string}
"""

def handler(event, context):
    text = event.get('text', '')
    language = event.get('language', 'en')
    words = text.lower().split()
    return {
        'words': words,
        'count': len(words),
        'language': language
    }
