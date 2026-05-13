"""Analyze function — word frequency analysis.

@input {words: array, count: integer, language: string}
@output {topWords: array, uniqueCount: integer, totalCount: integer}
"""

def handler(event, context):
    words = event.get('words', [])
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    
    top = sorted(freq.items(), key=lambda x: -x[1])
    return {
        'topWords': [{'word': w, 'freq': f} for w, f in top],
        'uniqueCount': len(freq),
        'totalCount': len(words)
    }
