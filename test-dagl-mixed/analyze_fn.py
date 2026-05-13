def handler(event, context):
    """Python version of Analyze — same interface as the Node.js one."""
    tokens = event.get("tokens", [])
    freq = {}
    for word in tokens:
        freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    return {
        "totalWords": len(tokens),
        "uniqueWords": len(sorted_words),
        "topWords": [{"word": w, "freq": c} for w, c in sorted_words[:5]],
    }
