"""
Analyzer - Stage 2 of the NLP Pipeline

Performs named entity recognition, frequency distribution analysis,
dependency feature extraction, collocation detection, and readability scoring.
All computation is genuine — no artificial delays.

Each output field depends on exactly ONE input field from the Tokenizer:
  entities     ← sentences
  freq_dist    ← pos_tags
  dep_features ← ngrams
  collocations ← vocab_stats
  readability  ← token_matrix
"""
import json
import time
import re
import math
from unum_streaming import StreamingPublisher, set_streaming_output
from collections import Counter


# ─── Named Entity Recognition (rule-based) ──────────────────────────────

PERSON_TITLES = {'mr', 'mrs', 'ms', 'dr', 'prof', 'professor', 'president',
                 'sir', 'lord', 'lady', 'king', 'queen', 'prince'}
ORG_SUFFIXES = {'inc', 'corp', 'ltd', 'llc', 'co', 'company', 'corporation',
                'university', 'institute', 'foundation', 'association',
                'committee', 'commission', 'department', 'agency'}
KNOWN_ENTITIES = {
    'alan turing': 'PERSON', 'joseph weizenbaum': 'PERSON', 'noam chomsky': 'PERSON',
    'vaswani': 'PERSON', 'chomsky': 'PERSON', 'turing': 'PERSON',
    'google': 'ORG', 'microsoft': 'ORG', 'facebook': 'ORG', 'openai': 'ORG',
    'georgetown': 'ORG', 'alpac': 'ORG',
    'bert': 'TECH', 'gpt': 'TECH', 'eliza': 'TECH', 'shrdlu': 'TECH',
    'transformer': 'TECH',
    'english': 'LANG', 'russian': 'LANG', 'french': 'LANG',
}


def extract_entities(sentence_data):
    """Extract named entities from sentence tokens using rules and patterns."""
    entities = []
    text = sentence_data.get('text', '')
    tokens = sentence_data.get('tokens', [])

    text_lower = text.lower()
    for entity, etype in KNOWN_ENTITIES.items():
        if entity in text_lower:
            entities.append({'text': entity.title(), 'type': etype, 'method': 'dictionary'})

    cap_pattern = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    for match in cap_pattern:
        words = match.lower().split()
        if len(words) >= 2 and match.lower() not in KNOWN_ENTITIES:
            if words[0] in PERSON_TITLES:
                entities.append({'text': match, 'type': 'PERSON', 'method': 'pattern'})
            elif any(w in ORG_SUFFIXES for w in words):
                entities.append({'text': match, 'type': 'ORG', 'method': 'pattern'})
            else:
                entities.append({'text': match, 'type': 'ENTITY', 'method': 'pattern'})

    years = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', text)
    for y in years:
        entities.append({'text': y, 'type': 'DATE', 'method': 'pattern'})

    acronyms = re.findall(r'\b([A-Z]{2,6})\b', text)
    for acr in acronyms:
        if acr.lower() not in {'the', 'and', 'for'}:
            entities.append({'text': acr, 'type': 'ACRONYM', 'method': 'pattern'})

    return entities


def compute_entity_similarity_matrix(unique_entities, all_sentence_data):
    """
    Build entity co-occurrence matrix across sentences — O(E * S * T).
    Computes entity context vectors and pairwise cosine similarity.
    """
    # Build entity → sentence index
    entity_sentences = {}
    for entity in unique_entities:
        e_lower = entity['text'].lower()
        entity_sentences[e_lower] = set()

    for si, sent_info in enumerate(all_sentence_data):
        text_lower = sent_info.get('text', '').lower()
        for e_lower in entity_sentences:
            if e_lower in text_lower:
                entity_sentences[e_lower].add(si)

    # Build context vectors for each entity (bag-of-words in surrounding sentences)
    all_context_words = set()
    entity_context_counts = {}
    for entity in unique_entities:
        e_lower = entity['text'].lower()
        context_words = Counter()
        for si in entity_sentences.get(e_lower, set()):
            if si < len(all_sentence_data):
                for tok in all_sentence_data[si].get('tokens', []):
                    w = tok.lower()
                    if w != e_lower and len(w) > 2:
                        context_words[w] += 1
                        all_context_words.add(w)
        entity_context_counts[e_lower] = context_words

    # Build context vocabulary and entity vectors
    context_vocab = sorted(all_context_words)
    cv_idx = {w: i for i, w in enumerate(context_vocab)}
    dim = len(context_vocab)

    entity_vectors = {}
    for e_lower, ctx in entity_context_counts.items():
        vec = [0.0] * dim
        for w, c in ctx.items():
            if w in cv_idx:
                vec[cv_idx[w]] = c
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        entity_vectors[e_lower] = vec

    # Pairwise cosine similarity between entity vectors
    e_list = list(entity_vectors.keys())
    similarity_matrix = []
    for i in range(len(e_list)):
        for j in range(i + 1, len(e_list)):
            v_i = entity_vectors[e_list[i]]
            v_j = entity_vectors[e_list[j]]
            dot = sum(a * b for a, b in zip(v_i, v_j))
            if dot > 0.01:
                similarity_matrix.append({
                    'entity_a': e_list[i], 'entity_b': e_list[j],
                    'cosine_sim': round(dot, 4),
                })

    similarity_matrix.sort(key=lambda x: x['cosine_sim'], reverse=True)

    # Entity contexts (top words per entity)
    entity_contexts = {}
    for e_lower, ctx in entity_context_counts.items():
        entity_contexts[e_lower] = dict(ctx.most_common(30))

    return similarity_matrix[:100], entity_contexts


def compute_freq_distribution(pos_data):
    """Compute detailed frequency distributions from POS-tagged data.
    Builds full POS transition matrix and computes mutual information.
    """
    tagged = pos_data.get('tagged', [])
    distribution = pos_data.get('distribution', {})

    pos_words = {}
    for item in tagged:
        pos = item['pos']
        word = item['word'].lower()
        if pos not in pos_words:
            pos_words[pos] = Counter()
        pos_words[pos][word] += 1

    top_per_pos = {pos: counter.most_common(20) for pos, counter in pos_words.items()}

    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
    content_count = sum(distribution.get(p, 0) for p in content_pos)
    total = sum(distribution.values())
    lexical_density = content_count / max(total, 1)

    # Full POS bigram + trigram analysis
    pos_seq = [item['pos'] for item in tagged]
    pos_bigrams = Counter()
    for i in range(len(pos_seq) - 1):
        pos_bigrams[(pos_seq[i], pos_seq[i + 1])] += 1

    pos_trigrams = Counter()
    for i in range(len(pos_seq) - 2):
        pos_trigrams[(pos_seq[i], pos_seq[i + 1], pos_seq[i + 2])] += 1

    # Mutual information for POS bigrams
    pos_unigram = Counter(pos_seq)
    total_bigrams_count = sum(pos_bigrams.values())
    total_unigrams_count = sum(pos_unigram.values())
    mi_scores = []
    for (p1, p2), count in pos_bigrams.items():
        p_joint = count / max(total_bigrams_count, 1)
        p1_m = pos_unigram[p1] / max(total_unigrams_count, 1)
        p2_m = pos_unigram[p2] / max(total_unigrams_count, 1)
        if p1_m > 0 and p2_m > 0 and p_joint > 0:
            mi = math.log2(p_joint / (p1_m * p2_m))
            mi_scores.append({'from': p1, 'to': p2, 'mi': round(mi, 4), 'count': count})
    mi_scores.sort(key=lambda x: abs(x['mi']), reverse=True)

    # Pass word-POS pairs forward for downstream heavy computation
    word_pos_pairs = [(item['word'].lower(), item['pos']) for item in tagged]

    top_transitions = [
        {'from': p[0], 'to': p[1], 'count': c}
        for (p, c) in pos_bigrams.most_common(50)
    ]

    return {
        'top_per_pos': {k: v for k, v in top_per_pos.items()},
        'lexical_density': round(lexical_density, 4),
        'pos_transitions': top_transitions,
        'mi_scores': mi_scores[:60],
        'pos_trigrams': [{'seq': list(k), 'count': v} for k, v in pos_trigrams.most_common(80)],
        'word_pos_pairs': word_pos_pairs,
        'total_analyzed': total
    }


def extract_dep_features(ngram_data):
    """Extract dependency-like features from n-gram patterns.
    Computes PMI for all bigram pairs and builds a phrase graph.
    """
    bigrams = ngram_data.get('bigrams', [])
    trigrams = ngram_data.get('trigrams', [])
    fourgrams = ngram_data.get('fourgrams', [])

    prep_set = {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about'}
    verb_endings = ('ing', 'ed', 'en', 'ize', 'ise', 'ate')
    noun_endings = ('tion', 'sion', 'ment', 'ness', 'ity', 'ence', 'ance')

    # Compute unigram frequencies from bigrams
    unigram_freq = Counter()
    for bg, count in bigrams:
        for w in bg.split():
            unigram_freq[w] += count
    total_unigrams = sum(unigram_freq.values())
    total_bigrams = sum(c for _, c in bigrams)

    # PMI for every bigram pair
    pmi_scored = []
    for bg, count in bigrams:
        words = bg.split()
        if len(words) == 2:
            p_joint = count / max(total_bigrams, 1)
            p_w1 = unigram_freq.get(words[0], 1) / max(total_unigrams, 1)
            p_w2 = unigram_freq.get(words[1], 1) / max(total_unigrams, 1)
            if p_w1 > 0 and p_w2 > 0 and p_joint > 0:
                pmi = math.log2(p_joint / (p_w1 * p_w2))
                pmi_scored.append({
                    'bigram': bg, 'pmi': round(pmi, 4), 'freq': count,
                    'w1': words[0], 'w2': words[1]
                })
    pmi_scored.sort(key=lambda x: x['pmi'], reverse=True)

    verb_noun_pairs = []
    for item in pmi_scored:
        if item['w1'].endswith(verb_endings) and (item['w2'].endswith(noun_endings) or len(item['w2']) > 3):
            verb_noun_pairs.append({'verb': item['w1'], 'noun': item['w2'], 'freq': item['freq'], 'pmi': item['pmi']})

    prep_phrases = []
    for tg, count in trigrams:
        words = tg.split()
        if len(words) == 3 and words[1] in prep_set:
            prep_phrases.append({'head': words[0], 'prep': words[1], 'dep': words[2], 'freq': count})

    # Build phrase graph
    phrase_graph_edges = [
        {'source': item['w1'], 'target': item['w2'], 'weight': item['pmi'], 'freq': item['freq']}
        for item in pmi_scored[:200]
    ]

    # Graph metrics: degree distribution
    degree = Counter()
    for e in phrase_graph_edges:
        degree[e['source']] += 1
        degree[e['target']] += 1
    nodes = set()
    for e in phrase_graph_edges:
        nodes.add(e['source'])
        nodes.add(e['target'])

    # Fourgram chain patterns
    chain_patterns = [{'pattern': fg.split(), 'freq': count} for fg, count in fourgrams[:60] if len(fg.split()) == 4]

    bg_counts = [c for _, c in bigrams]
    total_bg = sum(bg_counts) if bg_counts else 1
    entropy = -sum((c / total_bg) * math.log2(c / total_bg) for c in bg_counts if c > 0) if bg_counts else 0

    return {
        'verb_noun_pairs': verb_noun_pairs[:50],
        'prep_phrases': prep_phrases[:50],
        'pmi_collocations': pmi_scored[:100],
        'phrase_graph_edges': phrase_graph_edges,
        'graph_hubs': [{'word': w, 'degree': d} for w, d in degree.most_common(30)],
        'chain_patterns': chain_patterns,
        'bigram_entropy': round(entropy, 4),
        'n_graph_nodes': len(nodes),
        'pattern_count': len(verb_noun_pairs) + len(prep_phrases)
    }


def detect_collocations(vocab_data):
    """Detect collocations. Performs Zipf-Mandelbrot fitting via grid search."""
    total_words = vocab_data.get('total_words', 1)
    zipf = vocab_data.get('zipf_analysis', [])
    length_dist = vocab_data.get('length_distribution', {})
    vocab_size = vocab_data.get('vocab_size', 0)
    hapax_count = vocab_data.get('hapax_count', 0)
    ttr = vocab_data.get('ttr', 0)

    freq_spectrum = Counter()
    for item in zipf:
        freq_spectrum[item['actual_freq']] += 1

    n = total_words
    simpson = sum(item['actual_freq'] * (item['actual_freq'] - 1) for item in zipf)
    simpson = simpson / max(n * (n - 1), 1)

    brunets_w = total_words ** (vocab_size ** -0.172) if vocab_size > 0 else 0
    honores_r = 100 * math.log(total_words) / max(1 - hapax_count / max(vocab_size, 1), 0.001) if total_words > 0 else 0

    total_chars = sum(int(length) * count for length, count in length_dist.items())
    avg_word_len = total_chars / max(total_words, 1)

    # Zipf-Mandelbrot fitting: f(r) = C / (r + b)^a — grid search
    best_a, best_b, best_error = 1.0, 0.0, float('inf')
    actual_freqs = [item['actual_freq'] for item in zipf if item['actual_freq'] > 0]
    if actual_freqs:
        c_est = actual_freqs[0]
        for a_10 in range(5, 25):
            a = a_10 / 10.0
            for b_10 in range(0, 30):
                b = b_10 / 10.0
                error = sum(
                    (actual - c_est / ((rank + b) ** a)) ** 2
                    for rank, actual in enumerate(actual_freqs[:200], 1)
                )
                if error < best_error:
                    best_error = error
                    best_a, best_b = a, b

    # Yule's K
    m2 = sum(freq ** 2 * count for freq, count in freq_spectrum.items())
    yules_k = 10000 * (m2 - total_words) / max(total_words ** 2, 1) if total_words > 0 else 0

    # Sichel's S
    dis_legomena = sum(1 for item in zipf if item['actual_freq'] == 2)
    sichels_s = dis_legomena / max(vocab_size, 1)

    # Word entropy
    word_entropy = -sum(
        (item['actual_freq'] / max(total_words, 1)) * math.log2(item['actual_freq'] / max(total_words, 1))
        for item in zipf if item['actual_freq'] > 0
    )

    return {
        'simpsons_diversity': round(simpson, 6),
        'brunets_w': round(brunets_w, 4),
        'honores_r': round(honores_r, 2),
        'yules_k': round(yules_k, 4),
        'sichels_s': round(sichels_s, 4),
        'word_entropy': round(word_entropy, 4),
        'hapax_ratio': round(hapax_count / max(vocab_size, 1), 4),
        'avg_word_length': round(avg_word_len, 2),
        'vocab_richness_ttr': round(ttr, 4),
        'zipf_mandelbrot': {'alpha': best_a, 'beta': best_b, 'mse': round(best_error, 2)},
        'frequency_spectrum': dict(freq_spectrum)
    }


def compute_readability(matrix_data):
    """Compute readability. Heavy: builds sentence similarity matrix (O(S^2 * V))."""
    sent_matrix = matrix_data.get('sent_word_matrix', [])
    cooccurrences = matrix_data.get('cooccurrences', [])
    top_words = matrix_data.get('top_words', [])

    sent_lengths = [s['length'] for s in sent_matrix]
    avg_sent_len = sum(sent_lengths) / max(len(sent_lengths), 1)
    if len(sent_lengths) > 1:
        variance = sum((l - avg_sent_len) ** 2 for l in sent_lengths) / (len(sent_lengths) - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0

    def count_syllables(word):
        word = word.lower()
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e') and not word.endswith('le'):
            count -= 1
        return max(count, 1)

    total_syllables = 0
    total_words_count = 0
    total_chars = 0
    for sent in sent_matrix:
        for word, count in sent['word_counts'].items():
            total_syllables += count_syllables(word) * count
            total_words_count += count
            total_chars += len(word) * count

    avg_syllables = total_syllables / max(total_words_count, 1)
    flesch = 206.835 - 1.015 * avg_sent_len - 84.6 * avg_syllables
    fk_grade = 0.39 * avg_sent_len + 11.8 * avg_syllables - 15.59
    chars_per_100 = (total_chars / max(total_words_count, 1)) * 100
    sents_per_100 = (len(sent_lengths) / max(total_words_count, 1)) * 100
    coleman_liau = 0.0588 * chars_per_100 - 0.296 * sents_per_100 - 15.8

    # ── Heavy: sentence-to-sentence cosine similarity matrix ───────────
    vocab_set = set(top_words[:100])
    vocab_list = sorted(vocab_set)
    vocab_idx = {w: i for i, w in enumerate(vocab_list)}
    dim = len(vocab_list)

    sent_vectors = []
    for sent in sent_matrix:
        vec = [0.0] * dim
        for word, count in sent['word_counts'].items():
            if word in vocab_idx:
                vec[vocab_idx[word]] = count
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        sent_vectors.append(vec)

    # Pairwise cosine similarity O(S^2 * V)
    n_sents = len(sent_vectors)
    cohesion_sum = 0.0
    n_pairs = 0
    adjacent_sim = []
    for i in range(n_sents):
        if i + 1 < n_sents:
            dot = sum(a * b for a, b in zip(sent_vectors[i], sent_vectors[i + 1]))
            adjacent_sim.append(round(dot, 4))
            cohesion_sum += dot
            n_pairs += 1
        for j in range(i + 2, min(i + 6, n_sents)):
            dot = sum(a * b for a, b in zip(sent_vectors[i], sent_vectors[j]))
            cohesion_sum += dot
            n_pairs += 1

    avg_cohesion = cohesion_sum / max(n_pairs, 1)

    n_cooc = len(cooccurrences)
    n_nodes = len(top_words)
    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
    network_density = n_cooc / max(max_edges, 1)

    return {
        'flesch_reading_ease': round(flesch, 2),
        'flesch_kincaid_grade': round(fk_grade, 2),
        'coleman_liau_index': round(coleman_liau, 2),
        'avg_sentence_length': round(avg_sent_len, 2),
        'sentence_length_std': round(std_dev, 2),
        'avg_syllables_per_word': round(avg_syllables, 2),
        'cooccurrence_density': round(network_density, 6),
        'cohesion_score': round(avg_cohesion, 4),
        'adjacent_similarities': adjacent_sim,
        'n_sentences': len(sent_lengths),
        'sent_vectors': sent_vectors,
        'vocab_index': vocab_list
    }


def lambda_handler(event, context):
    """Stage 2: Analyze tokenized text — NER, frequency, dependencies, readability."""

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = event.get('Session', '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="AnalyzerFunction",
        field_names=["entities", "freq_dist", "dep_features", "collocations", "readability"]
    )
    start_time = time.time()
    doc_id = event.get('sentences', {}).get('doc_id', 'unknown')
    print(f'[Analyzer] Starting on doc {doc_id}')

    # ── Field 1: entities ← sentences
    t0 = time.time()
    sentences_data = event.get('sentences', {})
    all_sentence_data = sentences_data.get('data', [])
    all_entities = []
    entity_types = Counter()
    for sent_info in all_sentence_data:
        ents = extract_entities(sent_info)
        all_entities.extend(ents)
        for e in ents:
            entity_types[e['type']] += 1
    seen = set()
    unique_entities = []
    for e in all_entities:
        key = (e['text'].lower(), e['type'])
        if key not in seen:
            seen.add(key)
            unique_entities.append(e)
    sim_matrix, entity_contexts = compute_entity_similarity_matrix(unique_entities, all_sentence_data)
    field_1_time = int((time.time() - t0) * 1000)
    print(f'[Analyzer] entities: {len(unique_entities)} unique, {len(sim_matrix)} pairs in {field_1_time}ms')
    entities_out = {
        'unique_entities': unique_entities,
        'type_counts': dict(entity_types),
        'total_found': len(all_entities),
        'similarity_matrix': sim_matrix,
        'entity_contexts': entity_contexts,
        'compute_ms': field_1_time,
        'doc_id': doc_id
    }
    _streaming_publisher.publish('entities', entities_out)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()

    # ── Field 2: freq_dist ← pos_tags
    t0 = time.time()
    freq_dist = compute_freq_distribution(event.get('pos_tags', {}))
    field_2_time = int((time.time() - t0) * 1000)
    print(f'[Analyzer] freq_dist: density={freq_dist["lexical_density"]} in {field_2_time}ms')
    freq_dist_out = {**freq_dist, 'compute_ms': field_2_time, 'doc_id': doc_id}
    _streaming_publisher.publish('freq_dist', freq_dist_out)

    # ── Field 3: dep_features ← ngrams
    t0 = time.time()
    dep_features = extract_dep_features(event.get('ngrams', {}))
    field_3_time = int((time.time() - t0) * 1000)
    print(f'[Analyzer] dep_features: {dep_features["pattern_count"]} patterns, {dep_features["n_graph_nodes"]} nodes in {field_3_time}ms')
    dep_features_out = {**dep_features, 'compute_ms': field_3_time, 'doc_id': doc_id}
    _streaming_publisher.publish('dep_features', dep_features_out)

    # ── Field 4: collocations ← vocab_stats
    t0 = time.time()
    collocations = detect_collocations(event.get('vocab_stats', {}))
    field_4_time = int((time.time() - t0) * 1000)
    print(f'[Analyzer] collocations: diversity={collocations["simpsons_diversity"]} in {field_4_time}ms')
    collocations_out = {**collocations, 'compute_ms': field_4_time, 'doc_id': doc_id}
    _streaming_publisher.publish('collocations', collocations_out)

    # ── Field 5: readability ← token_matrix
    t0 = time.time()
    readability = compute_readability(event.get('token_matrix', {}))
    field_5_time = int((time.time() - t0) * 1000)
    print(f'[Analyzer] readability: Flesch={readability["flesch_reading_ease"]}, cohesion={readability["cohesion_score"]} in {field_5_time}ms')
    readability_out = {**readability, 'compute_ms': field_5_time, 'doc_id': doc_id}
    _streaming_publisher.publish('readability', readability_out)

    total_time = int((time.time() - start_time) * 1000)
    print(f'[Analyzer] COMPLETE in {total_time}ms')

    return {
        'entities': entities_out,
        'freq_dist': freq_dist_out,
        'dep_features': dep_features_out,
        'collocations': collocations_out,
        'readability': readability_out
    }
