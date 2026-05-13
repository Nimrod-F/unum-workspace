def lambda_handler(event, context):
    """Generate a summary report from the analysis results.

    @input {topWords: array, uniqueCount: integer, totalCount: integer}
    @output {report: string, stats: object}
    """
    top_words = event['topWords']
    unique = event['uniqueCount']
    total = event['totalCount']

    top_list = ', '.join(f"{w['word']}({w['freq']})" for w in top_words[:5])
    report = f"Text analysis: {total} words, {unique} unique. Top: {top_list}"

    return {
        'report': report,
        'stats': {
            'totalWords': total,
            'uniqueWords': unique,
            'topWords': top_words
        }
    }
