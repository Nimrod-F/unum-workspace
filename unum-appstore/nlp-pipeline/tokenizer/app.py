"""
Tokenizer - Stage 1 of the NLP Pipeline

Performs real text tokenization, POS tagging, n-gram extraction,
and vocabulary statistics. All computation is genuine — no artificial delays.

Produces 5 independent output fields that downstream stages consume one-to-one.

Academic basis: Tokenization/POS-tagging is the first stage in virtually every
NLP pipeline described in serverless computing literature (SeBS, SAND, etc.)
"""
import json
import time
import re
import math
from unum_streaming import StreamingPublisher, set_streaming_output
from collections import Counter


# ─── Lightweight POS tagger (rule-based, no external deps) ───────────────

# Common English word → POS tag mappings (subset)
DETERMINERS = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your',
               'his', 'her', 'its', 'our', 'their', 'some', 'any', 'no', 'every',
               'each', 'all', 'both', 'few', 'many', 'much', 'several'}
PREPOSITIONS = {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about',
                'into', 'through', 'during', 'before', 'after', 'above', 'below',
                'between', 'under', 'over', 'up', 'down', 'out', 'off', 'against'}
CONJUNCTIONS = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'because', 'although',
                'while', 'if', 'when', 'unless', 'until', 'since', 'whereas'}
PRONOUNS = {'i', 'me', 'you', 'he', 'him', 'she', 'her', 'it', 'we', 'us', 'they',
            'them', 'myself', 'yourself', 'himself', 'herself', 'itself', 'who',
            'whom', 'which', 'what', 'whoever', 'whatever'}
AUXILIARIES = {'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
               'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
               'should', 'may', 'might', 'can', 'could', 'must'}
ADVERBS_COMMON = {'not', 'very', 'also', 'often', 'never', 'always', 'sometimes',
                  'usually', 'here', 'there', 'now', 'then', 'already', 'still',
                  'just', 'only', 'really', 'quite', 'almost', 'enough', 'too',
                  'well', 'however', 'therefore', 'moreover', 'furthermore'}


def pos_tag_word(word):
    """Rule-based POS tagging for a single word."""
    w = word.lower()
    if w in DETERMINERS:
        return 'DET'
    if w in PREPOSITIONS:
        return 'ADP'
    if w in CONJUNCTIONS:
        return 'CONJ'
    if w in PRONOUNS:
        return 'PRON'
    if w in AUXILIARIES:
        return 'AUX'
    if w in ADVERBS_COMMON:
        return 'ADV'
    if w.endswith('ly') and len(w) > 4:
        return 'ADV'
    if w.endswith(('ing', 'ed', 'en', 'ize', 'ise', 'ate', 'ify')):
        return 'VERB'
    if w.endswith(('tion', 'sion', 'ment', 'ness', 'ity', 'ence', 'ance', 'ism')):
        return 'NOUN'
    if w.endswith(('ous', 'ive', 'ful', 'less', 'able', 'ible', 'al', 'ial')):
        return 'ADJ'
    if w[0].isupper() and len(w) > 1:
        return 'PROPN'
    if w.isdigit():
        return 'NUM'
    return 'NOUN'  # Default to noun


def tokenize_text(text):
    """Split text into word tokens."""
    return re.findall(r"\b[a-zA-Z0-9']+\b", text)


def split_sentences(text):
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def extract_ngrams(tokens, n):
    """Extract n-grams from token list."""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def lambda_handler(event, context):
    """
    Stage 1: Tokenize input text and produce 5 independent analysis fields.

    Input: { "text": "...", "doc_id": "..." }
    Output: { "sentences", "pos_tags", "ngrams", "vocab_stats", "token_matrix" }
    """

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = event.get('Session', '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="TokenizerFunction",
        field_names=["sentences", "pos_tags", "ngrams", "vocab_stats", "token_matrix"]
    )
    start_time = time.time()

    text = event.get('text', '')
    doc_id = event.get('doc_id', 'unknown')

    print(f'[Tokenizer] Starting on doc {doc_id}, text length={len(text)} chars')

    # ── Field 1: sentences ─────────────────────────────────────────────
    t0 = time.time()
    sentences = split_sentences(text)
    sentence_tokens = []
    for sent in sentences:
        tokens = tokenize_text(sent)
        sentence_tokens.append({
            'text': sent,
            'tokens': tokens,
            'n_tokens': len(tokens)
        })
    field_1_time = int((time.time() - t0) * 1000)
    print(f'[Tokenizer] sentences: {len(sentences)} sentences in {field_1_time}ms')

    sentences = {
        'data': sentence_tokens,
        'count': len(sentence_tokens),
        'compute_ms': field_1_time,
        'doc_id': doc_id
    }
    _streaming_publisher.publish('sentences', sentences)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()

    # ── Field 2: pos_tags ──────────────────────────────────────────────
    t0 = time.time()
    all_tokens = tokenize_text(text)
    pos_tagged = []
    for token in all_tokens:
        tag = pos_tag_word(token)
        pos_tagged.append({'word': token, 'pos': tag})

    # POS distribution
    pos_counts = Counter(item['pos'] for item in pos_tagged)
    pos_distribution = dict(pos_counts.most_common())
    field_2_time = int((time.time() - t0) * 1000)
    print(f'[Tokenizer] pos_tags: {len(pos_tagged)} tagged tokens in {field_2_time}ms')

    pos_tags = {
        'tagged': pos_tagged,
        'distribution': pos_distribution,
        'total_tokens': len(pos_tagged),
        'compute_ms': field_2_time,
        'doc_id': doc_id
    }
    _streaming_publisher.publish('pos_tags', pos_tags)

    # ── Field 3: ngrams ────────────────────────────────────────────────
    t0 = time.time()
    lower_tokens = [t.lower() for t in all_tokens]
    bigrams = extract_ngrams(lower_tokens, 2)
    trigrams = extract_ngrams(lower_tokens, 3)
    fourgrams = extract_ngrams(lower_tokens, 4)

    bigram_freq = Counter(bigrams).most_common(100)
    trigram_freq = Counter(trigrams).most_common(80)
    fourgram_freq = Counter(fourgrams).most_common(50)
    field_3_time = int((time.time() - t0) * 1000)
    print(f'[Tokenizer] ngrams: {len(bigrams)}bi + {len(trigrams)}tri + {len(fourgrams)}four in {field_3_time}ms')

    ngrams = {
        'bigrams': bigram_freq,
        'trigrams': trigram_freq,
        'fourgrams': fourgram_freq,
        'compute_ms': field_3_time,
        'doc_id': doc_id
    }
    _streaming_publisher.publish('ngrams', ngrams)

    # ── Field 4: vocab_stats ───────────────────────────────────────────
    t0 = time.time()
    word_freq = Counter(lower_tokens)
    total_words = len(lower_tokens)
    vocab_size = len(word_freq)

    # Zipf's law analysis
    sorted_freq = sorted(word_freq.values(), reverse=True)
    zipf_pairs = []
    for rank, freq in enumerate(sorted_freq[:200], 1):
        expected = sorted_freq[0] / rank
        zipf_pairs.append({
            'rank': rank,
            'actual_freq': freq,
            'zipf_expected': round(expected, 2),
            'deviation': round(abs(freq - expected) / max(expected, 1), 4)
        })

    # Hapax legomena (words appearing once)
    hapax = [w for w, c in word_freq.items() if c == 1]

    # Type-token ratio
    ttr = vocab_size / max(total_words, 1)

    # Word length distribution
    length_dist = Counter(len(w) for w in lower_tokens)

    field_4_time = int((time.time() - t0) * 1000)
    print(f'[Tokenizer] vocab_stats: vocab={vocab_size}, TTR={ttr:.3f} in {field_4_time}ms')

    vocab_stats = {
        'total_words': total_words,
        'vocab_size': vocab_size,
        'ttr': round(ttr, 4),
        'hapax_count': len(hapax),
        'zipf_analysis': zipf_pairs[:50],
        'length_distribution': dict(length_dist),
        'compute_ms': field_4_time,
        'doc_id': doc_id
    }
    _streaming_publisher.publish('vocab_stats', vocab_stats)

    # ── Field 5: token_matrix ──────────────────────────────────────────
    t0 = time.time()
    # Build co-occurrence matrix (context window = 5)
    window_size = 5
    top_words = [w for w, _ in word_freq.most_common(150)]
    word_set = set(top_words)
    cooccurrence = {}

    for i, token in enumerate(lower_tokens):
        if token not in word_set:
            continue
        start = max(0, i - window_size)
        end = min(len(lower_tokens), i + window_size + 1)
        for j in range(start, end):
            if i != j and lower_tokens[j] in word_set:
                pair = tuple(sorted([token, lower_tokens[j]]))
                cooccurrence[pair] = cooccurrence.get(pair, 0) + 1

    # Top co-occurrences
    top_cooc = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:200]
    cooc_list = [{'word1': p[0], 'word2': p[1], 'count': c} for (p, c) in top_cooc]

    # Sentence-word indicator matrix (for TF-IDF later)
    sent_word_matrix = []
    for si, sent_info in enumerate(sentence_tokens[:100]):
        word_counts = Counter(t.lower() for t in sent_info['tokens'])
        sent_word_matrix.append({
            'sent_idx': si,
            'word_counts': dict(word_counts),
            'length': sent_info['n_tokens']
        })

    field_5_time = int((time.time() - t0) * 1000)
    print(f'[Tokenizer] token_matrix: {len(cooc_list)} co-occurrences in {field_5_time}ms')

    token_matrix = {
        'cooccurrences': cooc_list,
        'sent_word_matrix': sent_word_matrix,
        'top_words': top_words[:100],
        'compute_ms': field_5_time,
        'doc_id': doc_id
    }
    _streaming_publisher.publish('token_matrix', token_matrix)

    total_time = int((time.time() - start_time) * 1000)
    print(f'[Tokenizer] COMPLETE in {total_time}ms')

    return {
        'sentences': sentences,
        'pos_tags': pos_tags,
        'ngrams': ngrams,
        'vocab_stats': vocab_stats,
        'token_matrix': token_matrix
    }


# ─── Sample text for local testing ──────────────────────────────────────

SAMPLE_TEXT = """
Natural language processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language, in particular how to program computers to process and analyze large
amounts of natural language data. The result is a computer capable of understanding
the contents of documents, including the contextual nuances of the language within
them. Challenges in natural language processing frequently involve speech recognition,
natural language understanding, and natural language generation. Natural language
processing has its roots in computational linguistics and has a history of more than
50 years. In the 1950s, Alan Turing published an article titled Computing Machinery
and Intelligence which proposed what is now called the Turing test as a criterion of
intelligence. The Georgetown experiment in 1954 involved fully automatic translation
of more than sixty Russian sentences into English. The authors claimed that within
three or five years, machine translation would be a solved problem. However, real
progress was much slower, and after the ALPAC report in 1966, which found that
ten-year-long research had failed to fulfill the expectations, funding for machine
translation was dramatically reduced. Little further research in machine translation
was conducted until the late 1980s when the first statistical machine translation
systems were developed. Some notably successful natural language processing systems
developed in the 1960s were SHRDLU, a natural language system working in restricted
blocks worlds with restricted vocabularies, and ELIZA, a simulation of a Rogerian
psychotherapist, written by Joseph Weizenbaum. Using almost no information about
human thought or emotion, ELIZA sometimes provided a startlingly human-like
interaction. During the 1970s, many programmers began to write conceptual ontologies,
which structured real-world information into computer-understandable data. In the
1980s and early 1990s, most natural language processing systems were based on complex
sets of hand-written rules. Starting in the late 1980s, however, there was a
revolution in natural language processing with the introduction of machine learning
algorithms for language processing. This was due to both the steady increase in
computational power and the gradual lessening of the dominance of Chomskyan theories
of linguistics, whose theoretical underpinnings discouraged the sort of corpus
linguistics that underlies the machine-learning approach to language processing.
Modern deep learning techniques for NLP include word embedding, transformer models,
and large language models. Recurrent neural networks and long short-term memory
networks were popular in the 2010s, but have largely been replaced by transformer
architectures. The attention mechanism, introduced in the Transformer model by
Vaswani et al. in 2017, revolutionized NLP by enabling models to process sequences
in parallel rather than sequentially. BERT, GPT, and their successors have achieved
state-of-the-art results on virtually all NLP benchmarks, including question answering,
named entity recognition, sentiment analysis, and machine translation. Transfer
learning through pre-trained language models has become the dominant paradigm in NLP.
""" * 3  # Repeat to increase workload


if __name__ == '__main__':
    result = lambda_handler({'text': SAMPLE_TEXT, 'doc_id': 'test'}, None)
    for field, data in result.items():
        ms = data.get('compute_ms', '?')
        print(f'  {field}: {ms}ms')
