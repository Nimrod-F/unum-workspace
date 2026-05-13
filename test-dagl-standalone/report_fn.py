"""Report function — generates text analysis summary.

@input {topWords: array, uniqueCount: integer, totalCount: integer}
@output {report: string, stats: object}
"""

def handler(event, context):
    top_words = event.get('topWords', [])
    unique = event.get('uniqueCount', 0)
    total = event.get('totalCount', 0)
    
    top_str = ', '.join(f"{w['word']}({w['freq']})" for w in top_words[:5])
    report = f"Text analysis: {total} words, {unique} unique. Top: {top_str}"
    
    return {
        'report': report,
        'stats': {
            'totalWords': total,
            'uniqueWords': unique,
            'topWords': top_words
        }
    }
