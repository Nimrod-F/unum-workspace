from collections import Counter


def lambda_handler(event, context):
    """Analyze tokenized text: compute word frequencies and statistics.

    @input {words: array, count: integer, language: string}
    @output {topWords: array, uniqueCount: integer, totalCount: integer}
    """
    words = event['words']
    total = event['count']

    freq = Counter(words)
    top_words = [{'word': w, 'freq': f} for w, f in freq.most_common(10)]

    return {
        'topWords': top_words,
        'uniqueCount': len(freq),
        'totalCount': total
    }
