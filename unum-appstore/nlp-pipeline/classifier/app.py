"""
Classifier - Stage 3 of the NLP Pipeline

Performs TF-IDF vectorization, sentiment analysis, topic classification,
text feature extraction, and document classification.
All computation is genuine — no artificial delays.

Each output field depends on exactly ONE input field from the Analyzer:
  sentiment      ← entities
  tfidf_vectors  ← freq_dist
  topics         ← dep_features
  text_features  ← collocations
  classification ← readability
"""
import json
import time
import math
import re
from unum_streaming import StreamingPublisher, set_streaming_output
from collections import Counter


# ─── Sentiment Lexicon ───────────────────────────────────────────────────

POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
    'brilliant', 'outstanding', 'superb', 'remarkable', 'impressive',
    'beautiful', 'perfect', 'love', 'best', 'happy', 'success', 'successful',
    'powerful', 'innovative', 'revolutionary', 'achieved', 'progress',
    'improved', 'efficient', 'effective', 'popular', 'advanced', 'capable',
    'enabled', 'significant', 'important', 'notable', 'breakthrough'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'failure',
    'failed', 'hate', 'ugly', 'boring', 'disappointing', 'difficult',
    'problem', 'problems', 'complex', 'complicated', 'slow', 'slower',
    'limited', 'restricted', 'reduced', 'lack', 'lacking', 'declined'
}

INTENSIFIERS = {'very', 'extremely', 'highly', 'incredibly', 'remarkably',
                'particularly', 'especially', 'most', 'quite'}
NEGATORS = {'not', 'no', 'never', 'neither', 'nor', 'hardly', 'barely', 'scarcely'}

TOPIC_KEYWORDS = {
    'technology': {'computer', 'software', 'algorithm', 'machine', 'learning',
                   'processing', 'system', 'model', 'data', 'neural', 'network',
                   'artificial', 'intelligence', 'computational', 'digital'},
    'science': {'research', 'experiment', 'theory', 'hypothesis', 'analysis',
                'study', 'scientific', 'observation', 'evidence', 'method'},
    'language': {'language', 'linguistic', 'word', 'sentence', 'grammar',
                 'translation', 'speech', 'text', 'vocabulary', 'syntax'},
    'history': {'history', 'historical', 'century', 'year', 'period', 'era',
                'ancient', 'modern', 'tradition', 'development', 'evolution'},
    'mathematics': {'number', 'equation', 'formula', 'statistical', 'probability',
                    'function', 'variable', 'matrix', 'vector', 'calculation'},
}


def analyze_sentiment(entity_data):
    """Compute sentiment scores using entity context vectors.
    Heavy: iterates over all entity context word bags to compute
    sentiment polarity and builds entity-sentiment co-occurrence matrix.
    """
    entities = entity_data.get('unique_entities', [])
    type_counts = entity_data.get('type_counts', {})
    sim_matrix = entity_data.get('similarity_matrix', [])
    entity_contexts = entity_data.get('entity_contexts', {})

    # Score sentiment for each entity's context window
    entity_sentiments = []
    for entity in entities:
        e_lower = entity['text'].lower()
        context = entity_contexts.get(e_lower, {})

        pos_score = sum(count for word, count in context.items() if word in POSITIVE_WORDS)
        neg_score = sum(count for word, count in context.items() if word in NEGATIVE_WORDS)
        intensified = sum(count for word, count in context.items() if word in INTENSIFIERS)
        negated = sum(count for word, count in context.items() if word in NEGATORS)

        # Adjusted sentiment
        raw = pos_score - neg_score
        adjusted = raw * (1 + 0.5 * intensified) * (-1 if negated > 0 else 1)
        total_context = sum(context.values())

        entity_sentiments.append({
            'entity': entity['text'],
            'type': entity['type'],
            'pos_score': pos_score,
            'neg_score': neg_score,
            'net': raw,
            'adjusted': round(adjusted, 2),
            'context_size': total_context,
            'polarity': round(raw / max(total_context, 1), 4)
        })

    # Build entity-sentiment interaction matrix: for each pair of similar
    # entities, compute combined sentiment profile
    sentiment_interactions = []
    for pair in sim_matrix:
        ea = pair['entity_a']
        eb = pair['entity_b']
        sent_a = next((s for s in entity_sentiments if s['entity'].lower() == ea), None)
        sent_b = next((s for s in entity_sentiments if s['entity'].lower() == eb), None)
        if sent_a and sent_b:
            combined = sent_a['adjusted'] + sent_b['adjusted']
            agreement = 1 if (sent_a['adjusted'] >= 0) == (sent_b['adjusted'] >= 0) else -1
            sentiment_interactions.append({
                'entity_a': ea, 'entity_b': eb,
                'sim': pair['cosine_sim'],
                'combined_sentiment': round(combined, 2),
                'agreement': agreement
            })

    # Overall metrics
    total_entities = len(entities)
    tech_ratio = type_counts.get('TECH', 0) / max(total_entities, 1)
    person_ratio = type_counts.get('PERSON', 0) / max(total_entities, 1)
    opinion_entities = sum(1 for es in entity_sentiments if es['net'] != 0)
    subjectivity = opinion_entities / max(total_entities, 1)

    # Compute sentiment distribution across entity types
    type_sentiment = {}
    for es in entity_sentiments:
        t = es['type']
        if t not in type_sentiment:
            type_sentiment[t] = {'pos': 0, 'neg': 0, 'neutral': 0, 'total_adjusted': 0}
        if es['net'] > 0:
            type_sentiment[t]['pos'] += 1
        elif es['net'] < 0:
            type_sentiment[t]['neg'] += 1
        else:
            type_sentiment[t]['neutral'] += 1
        type_sentiment[t]['total_adjusted'] += es['adjusted']

    return {
        'entity_sentiments': entity_sentiments,
        'sentiment_interactions': sentiment_interactions[:50],
        'type_sentiment': type_sentiment,
        'tech_entity_ratio': round(tech_ratio, 4),
        'person_entity_ratio': round(person_ratio, 4),
        'subjectivity': round(subjectivity, 4),
        'n_entities_analyzed': total_entities
    }


def compute_tfidf(freq_data):
    """Compute TF-IDF vectors with full word-POS cross-tabulation.
    Heavy: processes all word-POS pairs, builds weighted TF-IDF matrix,
    and computes pairwise term similarity using cosine distance.
    """
    top_per_pos = freq_data.get('top_per_pos', {})
    transitions = freq_data.get('pos_transitions', [])
    mi_scores = freq_data.get('mi_scores', [])
    word_pos_pairs = freq_data.get('word_pos_pairs', [])
    total = freq_data.get('total_analyzed', 1)

    # Build term frequencies across POS categories
    pos_weight = {
        'NOUN': 1.5, 'VERB': 1.8, 'ADJ': 2.0, 'ADV': 2.2,
        'PROPN': 2.5, 'DET': 0.5, 'ADP': 0.5, 'CONJ': 0.3,
        'PRON': 0.4, 'AUX': 0.3, 'NUM': 1.0
    }

    all_terms = {}
    for pos, words in top_per_pos.items():
        for word_pair in words:
            if isinstance(word_pair, (list, tuple)) and len(word_pair) == 2:
                word, freq = word_pair
            else:
                continue
            tf = freq / max(total, 1)
            weight = pos_weight.get(pos, 1.0)
            tfidf = tf * weight
            if word not in all_terms or all_terms[word]['tfidf'] < tfidf:
                all_terms[word] = {'word': word, 'tf': round(tf, 6), 'pos': pos,
                                   'tfidf': round(tfidf, 6), 'freq': freq}

    sorted_terms = sorted(all_terms.values(), key=lambda x: x['tfidf'], reverse=True)

    # Build word co-occurrence from word-POS pairs for term similarity
    # Sliding window = 5 over the word-POS stream
    word_context = Counter()
    word_freq_total = Counter()
    words_only = [wp[0] for wp in word_pos_pairs]
    window = 5
    top_term_set = set(t['word'] for t in sorted_terms[:100])

    for i, w in enumerate(words_only):
        if w not in top_term_set:
            continue
        word_freq_total[w] += 1
        for j in range(max(0, i - window), min(len(words_only), i + window + 1)):
            if i != j and words_only[j] in top_term_set:
                pair = tuple(sorted([w, words_only[j]]))
                word_context[pair] += 1

    # Build term vectors from co-occurrence context
    term_list = sorted(top_term_set)
    term_idx = {t: i for i, t in enumerate(term_list)}
    dim = len(term_list)
    term_vectors = {}
    for t in term_list:
        vec = [0.0] * dim
        for pair, count in word_context.items():
            if pair[0] == t and pair[1] in term_idx:
                vec[term_idx[pair[1]]] = count
            elif pair[1] == t and pair[0] in term_idx:
                vec[term_idx[pair[0]]] = count
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        term_vectors[t] = vec

    # Pairwise term similarity (O(T^2 * V))
    term_similarities = []
    tl = list(term_vectors.keys())
    for i in range(len(tl)):
        for j in range(i + 1, len(tl)):
            dot = sum(a * b for a, b in zip(term_vectors[tl[i]], term_vectors[tl[j]]))
            if dot > 0.1:
                term_similarities.append({
                    'term_a': tl[i], 'term_b': tl[j],
                    'cosine': round(dot, 4)
                })
    term_similarities.sort(key=lambda x: x['cosine'], reverse=True)

    vector_norm = math.sqrt(sum(t['tfidf'] ** 2 for t in sorted_terms)) if sorted_terms else 0

    # Transition entropy
    trans_counts = [t['count'] for t in transitions]
    trans_total = sum(trans_counts) if trans_counts else 1
    trans_entropy = -sum(
        (c / trans_total) * math.log2(c / trans_total) for c in trans_counts if c > 0
    ) if trans_counts else 0

    return {
        'top_terms': sorted_terms[:100],
        'term_similarities': term_similarities[:100],
        'vector_dim': len(sorted_terms),
        'vector_norm': round(vector_norm, 6),
        'transition_entropy': round(trans_entropy, 4),
        'lexical_density': freq_data.get('lexical_density', 0),
        'mi_scores': mi_scores
    }


def classify_topics(dep_data):
    """Classify document topics from dependency features.
    Heavy: runs iterative PageRank-style scoring on the phrase graph.
    """
    verb_noun_pairs = dep_data.get('verb_noun_pairs', [])
    prep_phrases = dep_data.get('prep_phrases', [])
    pmi_collocations = dep_data.get('pmi_collocations', [])
    phrase_graph_edges = dep_data.get('phrase_graph_edges', [])
    graph_hubs = dep_data.get('graph_hubs', [])

    # Collect all content words
    content_words = Counter()
    for vn in verb_noun_pairs:
        content_words[vn['verb']] += vn['freq']
        content_words[vn['noun']] += vn['freq']
    for pp in prep_phrases:
        content_words[pp['head']] += pp['freq']
        content_words[pp['dep']] += pp['freq']
    for col in pmi_collocations:
        content_words[col['w1']] += col['freq']
        content_words[col['w2']] += col['freq']

    # Score each topic
    topic_scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        matched = []
        for word in keywords:
            if word in content_words:
                score += content_words[word]
                matched.append(word)
        total_score_max = sum(content_words.values())
        topic_scores[topic] = {
            'score': score,
            'matched_keywords': matched,
            'coverage': len(matched) / len(keywords),
            'normalized': round(score / max(total_score_max, 1), 4)
        }

    # ── PageRank on phrase graph ──────────────────────────────────────
    nodes = set()
    adjacency = {}
    for edge in phrase_graph_edges:
        s, t = edge['source'], edge['target']
        nodes.add(s)
        nodes.add(t)
        if s not in adjacency:
            adjacency[s] = []
        adjacency[s].append(t)
        if t not in adjacency:
            adjacency[t] = []
        adjacency[t].append(s)

    # Initialize PageRank scores
    n_nodes = len(nodes)
    if n_nodes > 0:
        pr = {node: 1.0 / n_nodes for node in nodes}
        damping = 0.85
        # Iterate 30 rounds of PageRank
        for _ in range(30):
            new_pr = {}
            for node in nodes:
                rank_sum = 0.0
                for neighbor in adjacency.get(node, []):
                    out_degree = len(adjacency.get(neighbor, []))
                    if out_degree > 0:
                        rank_sum += pr[neighbor] / out_degree
                new_pr[node] = (1 - damping) / n_nodes + damping * rank_sum
            pr = new_pr

        top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:50]
    else:
        top_pr = []
        pr = {}

    # Map PageRank scores to topics
    topic_pr = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        topic_pr[topic] = sum(pr.get(w, 0) for w in keywords)

    primary = max(topic_scores.items(), key=lambda x: x[1]['score'])

    return {
        'topic_scores': topic_scores,
        'primary_topic': primary[0],
        'primary_confidence': primary[1].get('normalized', 0),
        'pagerank_top': [{'word': w, 'score': round(s, 6)} for w, s in top_pr],
        'topic_pagerank': {t: round(s, 6) for t, s in topic_pr.items()},
        'n_graph_nodes': n_nodes,
        'structural_entropy': dep_data.get('bigram_entropy', 0),
        'n_dep_patterns': len(verb_noun_pairs) + len(prep_phrases)
    }


def extract_text_features(collocation_data):
    """Extract high-level text features from collocation analysis.
    Heavy: performs bootstrap resampling to estimate confidence intervals
    for vocabulary richness metrics.
    """
    diversity = collocation_data.get('simpsons_diversity', 0)
    brunets = collocation_data.get('brunets_w', 0)
    honores = collocation_data.get('honores_r', 0)
    yules_k = collocation_data.get('yules_k', 0)
    sichels_s = collocation_data.get('sichels_s', 0)
    word_entropy = collocation_data.get('word_entropy', 0)
    hapax_ratio = collocation_data.get('hapax_ratio', 0)
    avg_word_len = collocation_data.get('avg_word_length', 0)
    ttr = collocation_data.get('vocab_richness_ttr', 0)
    zipf_params = collocation_data.get('zipf_mandelbrot', {})
    freq_spectrum = collocation_data.get('frequency_spectrum', {})

    features = {
        'lexical_richness': round(1 - diversity, 4),
        'vocabulary_growth': round(min(brunets / 100, 1.0), 4),
        'hapax_proportion': round(hapax_ratio, 4),
        'word_complexity': round(min(avg_word_len / 10, 1.0), 4),
        'type_token_ratio': round(ttr, 4),
    }

    formality = (avg_word_len / 8) * 0.4 + (1 - diversity) * 0.3 + ttr * 0.3
    features['formality_score'] = round(min(max(formality, 0), 1.0), 4)

    if ttr > 0.6 and avg_word_len > 5:
        genre = 'academic'
    elif ttr > 0.5 and avg_word_len > 4.5:
        genre = 'technical'
    elif ttr > 0.4:
        genre = 'journalistic'
    else:
        genre = 'conversational'
    features['estimated_genre'] = genre
    features['honores_r'] = round(honores, 2)
    features['brunets_w'] = round(brunets, 4)
    features['yules_k'] = round(yules_k, 4)
    features['sichels_s'] = round(sichels_s, 4)
    features['word_entropy'] = round(word_entropy, 4)
    features['zipf_alpha'] = zipf_params.get('alpha', 1.0)
    features['zipf_beta'] = zipf_params.get('beta', 0.0)

    # ── Bootstrap resampling for confidence intervals ─────────────────
    # Resample from frequency spectrum to estimate metric variance
    import random
    rng = random.Random(42)

    # Reconstruct frequency list from spectrum
    freq_list = []
    for freq_str, count in freq_spectrum.items():
        freq_list.extend([int(freq_str)] * count)

    n_bootstrap = 200
    ttr_samples = []
    hapax_samples = []
    simpson_samples = []

    if freq_list:
        n = len(freq_list)
        total_tokens = sum(freq_list)
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = [freq_list[rng.randint(0, n - 1)] for _ in range(n)]
            sample_total = sum(sample)
            sample_vocab = len(set(sample))

            ttr_s = sample_vocab / max(sample_total, 1)
            ttr_samples.append(ttr_s)

            hapax_s = sum(1 for f in sample if f == 1) / max(sample_vocab, 1)
            hapax_samples.append(hapax_s)

            simp_s = sum(f * (f - 1) for f in sample) / max(sample_total * (sample_total - 1), 1)
            simpson_samples.append(simp_s)

    def ci_95(samples):
        if not samples:
            return {'mean': 0, 'lower': 0, 'upper': 0}
        s = sorted(samples)
        n = len(s)
        return {
            'mean': round(sum(s) / n, 6),
            'lower': round(s[int(n * 0.025)], 6),
            'upper': round(s[int(n * 0.975)], 6)
        }

    features['bootstrap_ci'] = {
        'ttr': ci_95(ttr_samples),
        'hapax_ratio': ci_95(hapax_samples),
        'simpson': ci_95(simpson_samples),
        'n_resamples': n_bootstrap
    }

    return features


def build_classification(readability_data):
    """Build final classification from readability metrics.
    Heavy: uses sentence vectors to run k-means clustering (Lloyd's algorithm)
    to identify discourse segments.
    """
    flesch = readability_data.get('flesch_reading_ease', 50)
    fk_grade = readability_data.get('flesch_kincaid_grade', 8)
    coleman = readability_data.get('coleman_liau_index', 8)
    avg_sent_len = readability_data.get('avg_sentence_length', 15)
    std_dev = readability_data.get('sentence_length_std', 5)
    avg_syllables = readability_data.get('avg_syllables_per_word', 1.5)
    cohesion = readability_data.get('cohesion_score', 0)
    sent_vectors = readability_data.get('sent_vectors', [])
    vocab_index = readability_data.get('vocab_index', [])

    # Reading level
    if fk_grade <= 6:
        level = 'elementary'
    elif fk_grade <= 9:
        level = 'middle_school'
    elif fk_grade <= 12:
        level = 'high_school'
    elif fk_grade <= 16:
        level = 'undergraduate'
    else:
        level = 'graduate'

    complexity_score = min(1.0, (fk_grade / 20) * 0.4 + (avg_syllables / 3) * 0.3 + (std_dev / 15) * 0.3)
    variety = 1.0 - math.exp(-std_dev / 5) if std_dev > 0 else 0

    # ── K-Means clustering on sentence vectors (Lloyd's algorithm) ────
    import random
    rng = random.Random(42)

    k = min(5, max(2, len(sent_vectors) // 10))
    dim = len(sent_vectors[0]) if sent_vectors else 0
    clusters = []

    if dim > 0 and len(sent_vectors) >= k:
        # Initialize centroids randomly
        indices = list(range(len(sent_vectors)))
        rng.shuffle(indices)
        centroids = [list(sent_vectors[indices[i]]) for i in range(k)]

        # Run 20 iterations of Lloyd's algorithm
        assignments = [0] * len(sent_vectors)
        for iteration in range(20):
            # Assign each sentence to nearest centroid
            for si, sv in enumerate(sent_vectors):
                best_c = 0
                best_dist = float('inf')
                for ci, centroid in enumerate(centroids):
                    dist = sum((a - b) ** 2 for a, b in zip(sv, centroid))
                    if dist < best_dist:
                        best_dist = dist
                        best_c = ci
                assignments[si] = best_c

            # Update centroids
            for ci in range(k):
                members = [sent_vectors[si] for si in range(len(sent_vectors)) if assignments[si] == ci]
                if members:
                    centroids[ci] = [
                        sum(m[d] for m in members) / len(members)
                        for d in range(dim)
                    ]

        # Compute cluster quality
        for ci in range(k):
            member_indices = [si for si in range(len(sent_vectors)) if assignments[si] == ci]
            if member_indices:
                # Intra-cluster distance
                intra_dist = 0
                for si in member_indices:
                    intra_dist += sum(
                        (sent_vectors[si][d] - centroids[ci][d]) ** 2
                        for d in range(dim)
                    )
                intra_dist /= len(member_indices)

                clusters.append({
                    'cluster_id': ci,
                    'size': len(member_indices),
                    'avg_intra_distance': round(math.sqrt(intra_dist), 4),
                    'sentence_indices': member_indices[:20]  # sample
                })

    return {
        'reading_level': level,
        'grade_level': round(fk_grade, 1),
        'complexity_score': round(complexity_score, 4),
        'sentence_variety': round(variety, 4),
        'flesch_score': round(flesch, 2),
        'coleman_liau': round(coleman, 2),
        'cohesion_density': round(cohesion, 4),
        'avg_sentence_words': round(avg_sent_len, 1),
        'n_sentences': readability_data.get('n_sentences', 0),
        'discourse_clusters': clusters,
        'n_clusters': len(clusters)
    }


def lambda_handler(event, context):
    """Stage 3: Classify text — sentiment, TF-IDF, topics, features, classification."""

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = event.get('Session', '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="ClassifierFunction",
        field_names=["sentiment", "tfidf_vectors", "topics", "text_features", "classification"]
    )
    start_time = time.time()
    doc_id = event.get('entities', {}).get('doc_id', 'unknown')
    print(f'[Classifier] Starting on doc {doc_id}')

    # ── Field 1: sentiment ← entities
    t0 = time.time()
    sentiment = analyze_sentiment(event.get('entities', {}))
    field_1_time = int((time.time() - t0) * 1000)
    print(f'[Classifier] sentiment: subjectivity={sentiment["subjectivity"]} in {field_1_time}ms')
    sentiment_out = {**sentiment, 'compute_ms': field_1_time, 'doc_id': doc_id}
    _streaming_publisher.publish('sentiment', sentiment_out)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()

    # ── Field 2: tfidf_vectors ← freq_dist
    t0 = time.time()
    tfidf = compute_tfidf(event.get('freq_dist', {}))
    field_2_time = int((time.time() - t0) * 1000)
    print(f'[Classifier] tfidf: dim={tfidf["vector_dim"]}, {len(tfidf["term_similarities"])} sim-pairs in {field_2_time}ms')
    tfidf_vectors = {**tfidf, 'compute_ms': field_2_time, 'doc_id': doc_id}
    _streaming_publisher.publish('tfidf_vectors', tfidf_vectors)

    # ── Field 3: topics ← dep_features
    t0 = time.time()
    topics = classify_topics(event.get('dep_features', {}))
    field_3_time = int((time.time() - t0) * 1000)
    print(f'[Classifier] topics: primary={topics["primary_topic"]}, pr_nodes={topics["n_graph_nodes"]} in {field_3_time}ms')
    topics_out = {**topics, 'compute_ms': field_3_time, 'doc_id': doc_id}
    _streaming_publisher.publish('topics', topics_out)

    # ── Field 4: text_features ← collocations
    t0 = time.time()
    text_features = extract_text_features(event.get('collocations', {}))
    field_4_time = int((time.time() - t0) * 1000)
    print(f'[Classifier] text_features: genre={text_features["estimated_genre"]} in {field_4_time}ms')
    text_features_out = {**text_features, 'compute_ms': field_4_time, 'doc_id': doc_id}
    _streaming_publisher.publish('text_features', text_features_out)

    # ── Field 5: classification ← readability
    t0 = time.time()
    classification = build_classification(event.get('readability', {}))
    field_5_time = int((time.time() - t0) * 1000)
    print(f'[Classifier] classification: level={classification["reading_level"]}, clusters={classification["n_clusters"]} in {field_5_time}ms')
    classification_out = {**classification, 'compute_ms': field_5_time, 'doc_id': doc_id}
    _streaming_publisher.publish('classification', classification_out)

    total_time = int((time.time() - start_time) * 1000)
    print(f'[Classifier] COMPLETE in {total_time}ms')

    return {
        'sentiment': sentiment_out,
        'tfidf_vectors': tfidf_vectors,
        'topics': topics_out,
        'text_features': text_features_out,
        'classification': classification_out
    }
