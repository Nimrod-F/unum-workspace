import functions_framework
import json


@functions_framework.http
def handler(request):
    """Analyze function — counts word frequencies from tokens."""
    data = request.get_json(silent=True) or {}
    tokens = data.get("tokens", [])

    freq = {}
    for word in tokens:
        freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: -x[1])

    result = {
        "totalWords": len(tokens),
        "uniqueWords": len(sorted_words),
        "topWords": [{"word": w, "freq": c} for w, c in sorted_words[:5]],
    }
    return json.dumps(result), 200, {"Content-Type": "application/json"}
