"""
Summarizer - Stage 4 (Final) of the NLP Pipeline

Performs extractive summarization using TextRank-style sentence scoring,
keyword extraction, abstract generation, citation graph, and final report.
All computation is genuine — no artificial delays.

This is the final stage so no streaming output is needed.

Each output field depends on exactly ONE input field from the Classifier:
  key_sentences  ← sentiment
  keywords       ← tfidf_vectors
  abstract       ← topics
  citation_graph ← text_features
  final_report   ← classification
"""
import json
import time
import math
from collections import Counter


def score_sentences_textrank(sentiment_data):
    """
    TextRank-style sentence scoring using entity sentiment features.
    Heavy: builds entity-entity graph from sentiment interactions
    and runs iterative eigenvector centrality approximation.
    """
    entity_sentiments = sentiment_data.get('entity_sentiments', [])
    sentiment_interactions = sentiment_data.get('sentiment_interactions', [])
    type_sentiment = sentiment_data.get('type_sentiment', {})
    n_entities = sentiment_data.get('n_entities_analyzed', 0)
    subjectivity = sentiment_data.get('subjectivity', 0)
    tech_ratio = sentiment_data.get('tech_entity_ratio', 0)
    person_ratio = sentiment_data.get('person_entity_ratio', 0)

    # Build entity interaction graph
    nodes = set()
    adjacency = {}
    for inter in sentiment_interactions:
        a, b = inter['entity_a'], inter['entity_b']
        nodes.add(a)
        nodes.add(b)
        weight = abs(inter['combined_sentiment']) * inter['sim']
        if a not in adjacency:
            adjacency[a] = {}
        adjacency[a][b] = adjacency[a].get(b, 0) + weight
        if b not in adjacency:
            adjacency[b] = {}
        adjacency[b][a] = adjacency[b].get(a, 0) + weight

    # Iterative eigenvector centrality (power iteration, 30 rounds)
    n_nodes = len(nodes)
    if n_nodes > 0:
        centrality = {n: 1.0 / n_nodes for n in nodes}
        for _ in range(30):
            new_c = {}
            for node in nodes:
                score = 0
                for neighbor, weight in adjacency.get(node, {}).items():
                    score += centrality[neighbor] * weight
                new_c[node] = score
            # Normalize
            norm = math.sqrt(sum(v ** 2 for v in new_c.values()))
            if norm > 0:
                centrality = {k: v / norm for k, v in new_c.items()}
            else:
                centrality = new_c

        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    else:
        centrality = {}
        top_central = []

    # Score entities by centrality + sentiment
    scored_entities = []
    for es in entity_sentiments:
        e_lower = es['entity'].lower()
        c_score = centrality.get(e_lower, 0)
        importance = c_score * 10  # Scale up

        # Type bonus
        type_bonus = {'TECH': 3, 'PERSON': 2, 'ORG': 2, 'DATE': 1}.get(es['type'], 1)
        importance += type_bonus

        # Sentiment bonus
        if es['net'] != 0:
            importance += abs(es['adjusted']) * 0.5

        scored_entities.append({
            'entity': es['entity'],
            'type': es['type'],
            'importance': round(importance, 4),
            'centrality': round(c_score, 6),
            'sentiment_net': es['net'],
            'adjusted_sentiment': es['adjusted']
        })

    scored_entities.sort(key=lambda x: x['importance'], reverse=True)

    # Sentiment type summary
    pos_entities = sum(1 for e in entity_sentiments if e['net'] > 0)
    neg_entities = sum(1 for e in entity_sentiments if e['net'] < 0)

    return {
        'key_entities': scored_entities[:30],
        'n_key_entities': min(len(scored_entities), 30),
        'top_central_nodes': [{'entity': e, 'centrality': round(c, 6)} for e, c in top_central[:20]],
        'sentiment_balance': {
            'positive': pos_entities,
            'negative': neg_entities,
            'neutral': len(entity_sentiments) - pos_entities - neg_entities,
            'overall_polarity': 'positive' if pos_entities > neg_entities else
                               'negative' if neg_entities > pos_entities else 'neutral'
        },
        'type_sentiment_summary': type_sentiment,
        'tech_focus': round(tech_ratio, 4),
        'person_focus': round(person_ratio, 4),
        'subjectivity': round(subjectivity, 4)
    }


def extract_keywords(tfidf_data):
    """Extract top keywords from TF-IDF vectors.
    Heavy: builds term clustering using agglomerative single-linkage
    on the term similarity matrix.
    """
    top_terms = tfidf_data.get('top_terms', [])
    term_similarities = tfidf_data.get('term_similarities', [])
    vector_dim = tfidf_data.get('vector_dim', 0)
    vector_norm = tfidf_data.get('vector_norm', 0)
    trans_entropy = tfidf_data.get('transition_entropy', 0)
    lex_density = tfidf_data.get('lexical_density', 0)
    mi_scores = tfidf_data.get('mi_scores', [])

    # Select content keywords
    content_pos = {'NOUN', 'VERB', 'ADJ', 'PROPN'}
    keywords = []
    for term in top_terms:
        if term.get('pos') in content_pos and term.get('tfidf', 0) > 0:
            keywords.append({
                'word': term['word'], 'score': term['tfidf'],
                'pos': term['pos'], 'frequency': term.get('freq', 0)
            })

    # ── Agglomerative clustering on term similarity ──────────────────
    # Build distance matrix from similarity
    term_set = set()
    sim_lookup = {}
    for s in term_similarities:
        a, b = s['term_a'], s['term_b']
        term_set.add(a)
        term_set.add(b)
        sim_lookup[(a, b)] = s['cosine']
        sim_lookup[(b, a)] = s['cosine']

    term_list = sorted(term_set)
    n = len(term_list)

    # Single-linkage agglomerative clustering
    # Each term starts as its own cluster
    cluster_map = {t: i for i, t in enumerate(term_list)}
    clusters_dict = {i: [t] for i, t in enumerate(term_list)}
    next_id = n
    merge_history = []

    if n > 1:
        # Build all pairwise distances
        distances = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_lookup.get((term_list[i], term_list[j]), 0)
                distances[(i, j)] = 1.0 - sim

        # Merge until threshold or limited merges
        max_merges = min(n - 1, 50)
        for _ in range(max_merges):
            # Find closest pair of clusters (single linkage)
            best_pair = None
            best_dist = float('inf')
            active_clusters = set(clusters_dict.keys())
            ac_list = sorted(active_clusters)

            for i_idx in range(len(ac_list)):
                for j_idx in range(i_idx + 1, len(ac_list)):
                    ci, cj = ac_list[i_idx], ac_list[j_idx]
                    # Single linkage: min distance between any pair
                    min_dist = float('inf')
                    for ti in clusters_dict[ci]:
                        for tj in clusters_dict[cj]:
                            ii = term_list.index(ti) if ti in term_list else -1
                            jj = term_list.index(tj) if tj in term_list else -1
                            if ii >= 0 and jj >= 0:
                                key = (min(ii, jj), max(ii, jj))
                                d = distances.get(key, 1.0)
                                min_dist = min(min_dist, d)
                    if min_dist < best_dist:
                        best_dist = min_dist
                        best_pair = (ci, cj)

            if best_pair is None or best_dist > 0.8:
                break

            ci, cj = best_pair
            merged = clusters_dict[ci] + clusters_dict[cj]
            clusters_dict[next_id] = merged
            merge_history.append({
                'merged': [ci, cj], 'distance': round(best_dist, 4),
                'new_cluster': next_id, 'size': len(merged)
            })
            del clusters_dict[ci]
            del clusters_dict[cj]
            next_id += 1

    # Build keyword groups from final clusters
    keyword_groups = {}
    for cid, members in clusters_dict.items():
        if len(members) > 1:
            keyword_groups[f'cluster_{cid}'] = members

    return {
        'top_keywords': keywords[:50],
        'n_keywords': len(keywords),
        'keyword_groups': keyword_groups,
        'merge_history': merge_history[:30],
        'n_term_clusters': len([c for c in clusters_dict.values() if len(c) > 1]),
        'vector_dimensionality': vector_dim,
        'feature_norm': round(vector_norm, 4),
        'structural_entropy': round(trans_entropy, 4),
        'content_density': round(lex_density, 4)
    }


def generate_abstract(topic_data):
    """Generate structured abstract from topic classification.
    Heavy: refines topic assignments using iterative EM-style soft clustering
    on pagerank scores.
    """
    topic_scores = topic_data.get('topic_scores', {})
    primary = topic_data.get('primary_topic', 'unknown')
    confidence = topic_data.get('primary_confidence', 0)
    entropy = topic_data.get('structural_entropy', 0)
    n_patterns = topic_data.get('n_dep_patterns', 0)
    pagerank_top = topic_data.get('pagerank_top', [])
    topic_pr = topic_data.get('topic_pagerank', {})
    n_graph = topic_data.get('n_graph_nodes', 0)

    # Build topic profile
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1].get('score', 0), reverse=True)
    topic_profile = []
    for topic, data in sorted_topics:
        topic_profile.append({
            'topic': topic, 'relevance': data.get('normalized', 0),
            'coverage': data.get('coverage', 0), 'matched': data.get('matched_keywords', [])
        })

    significant_topics = [tp for tp in topic_profile if tp['relevance'] > 0.05]

    # ── EM-style soft topic assignment for PageRank words ────────────
    # Assign each high-PageRank word to topics with soft probabilities
    topic_keys = list(TOPIC_KEYWORDS.keys()) if 'TOPIC_KEYWORDS' not in dir() else list(topic_scores.keys())
    word_topic_probs = []

    for pr_item in pagerank_top:
        word = pr_item['word']
        pr_score = pr_item['score']
        probs = {}

        for topic in topic_keys:
            # Check keyword membership
            ts = topic_scores.get(topic, {})
            matched = ts.get('matched_keywords', [])
            if word in matched:
                probs[topic] = pr_score * 2
            else:
                probs[topic] = pr_score * 0.1

        # Normalize to probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {t: round(p / total, 4) for t, p in probs.items()}

        word_topic_probs.append({'word': word, 'pr_score': pr_score, 'topic_probs': probs})

    # Iterative refinement: 10 rounds of updating topic-word affinity
    topic_affinity = {t: 1.0 / len(topic_keys) for t in topic_keys}
    for _ in range(10):
        # E-step: update word-topic probabilities
        for wtp in word_topic_probs:
            new_probs = {}
            for topic in topic_keys:
                base = 2.0 if wtp['word'] in topic_scores.get(topic, {}).get('matched_keywords', []) else 0.1
                new_probs[topic] = base * topic_affinity[topic]
            total = sum(new_probs.values())
            if total > 0:
                wtp['topic_probs'] = {t: round(p / total, 4) for t, p in new_probs.items()}

        # M-step: update topic affinity from word assignments
        for topic in topic_keys:
            topic_affinity[topic] = sum(
                wtp['topic_probs'].get(topic, 0) * wtp['pr_score']
                for wtp in word_topic_probs
            )
        total_aff = sum(topic_affinity.values())
        if total_aff > 0:
            topic_affinity = {t: a / total_aff for t, a in topic_affinity.items()}

    # Interdisciplinarity
    interdisciplinary = sum(tp['relevance'] for tp in significant_topics[1:]) if len(significant_topics) > 1 else 0

    return {
        'topic_profile': topic_profile,
        'primary_topic': primary,
        'primary_confidence': round(confidence, 4),
        'significant_topics': [tp['topic'] for tp in significant_topics],
        'interdisciplinarity': round(interdisciplinary, 4),
        'structural_complexity': round(entropy, 4),
        'dep_pattern_count': n_patterns,
        'refined_topic_affinity': {t: round(a, 4) for t, a in topic_affinity.items()},
        'word_topic_assignments': word_topic_probs[:30],
        'graph_size': n_graph
    }


# Topic keywords needed for abstract generation
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


def build_citation_graph(text_feature_data):
    """Build citation/reference quality graph from text features.
    Heavy: performs Monte Carlo simulation using bootstrap CI parameters
    to estimate publication quality distribution.
    """
    formality = text_feature_data.get('formality_score', 0)
    genre = text_feature_data.get('estimated_genre', 'unknown')
    richness = text_feature_data.get('lexical_richness', 0)
    complexity = text_feature_data.get('word_complexity', 0)
    ttr = text_feature_data.get('type_token_ratio', 0)
    honores = text_feature_data.get('honores_r', 0)
    brunets = text_feature_data.get('brunets_w', 0)
    yules_k = text_feature_data.get('yules_k', 0)
    sichels_s = text_feature_data.get('sichels_s', 0)
    word_entropy = text_feature_data.get('word_entropy', 0)
    bootstrap_ci = text_feature_data.get('bootstrap_ci', {})

    quality_dimensions = {
        'vocabulary_sophistication': round(richness * 0.4 + complexity * 0.6, 4),
        'lexical_diversity': round(ttr, 4),
        'formality': round(formality, 4),
        'statistical_richness': round(min(honores / 2000, 1.0), 4),
        'word_entropy_norm': round(min(word_entropy / 12, 1.0), 4),
    }
    quality_score = sum(quality_dimensions.values()) / len(quality_dimensions)

    genre_benchmarks = {
        'academic': {'min_formality': 0.6, 'min_ttr': 0.5, 'min_complexity': 0.4},
        'technical': {'min_formality': 0.5, 'min_ttr': 0.4, 'min_complexity': 0.35},
        'journalistic': {'min_formality': 0.4, 'min_ttr': 0.35, 'min_complexity': 0.3},
        'conversational': {'min_formality': 0.2, 'min_ttr': 0.25, 'min_complexity': 0.2},
    }
    benchmark = genre_benchmarks.get(genre, genre_benchmarks['journalistic'])
    meets_benchmarks = all([
        formality >= benchmark['min_formality'],
        ttr >= benchmark['min_ttr'],
        complexity >= benchmark['min_complexity']
    ])

    # ── Monte Carlo quality simulation ─────────────────────────────────
    import random
    rng = random.Random(42)

    ttr_ci = bootstrap_ci.get('ttr', {})
    ttr_mean = ttr_ci.get('mean', ttr)
    ttr_lower = ttr_ci.get('lower', ttr * 0.9)
    ttr_upper = ttr_ci.get('upper', ttr * 1.1)
    ttr_std = (ttr_upper - ttr_lower) / 4 if ttr_upper > ttr_lower else 0.01

    simpson_ci = bootstrap_ci.get('simpson', {})
    simp_mean = simpson_ci.get('mean', 0.01)
    simp_lower = simpson_ci.get('lower', 0.005)
    simp_upper = simpson_ci.get('upper', 0.02)
    simp_std = (simp_upper - simp_lower) / 4 if simp_upper > simp_lower else 0.001

    n_sim = 500
    quality_samples = []
    for _ in range(n_sim):
        # Sample from estimated distributions
        ttr_s = max(0, rng.gauss(ttr_mean, ttr_std))
        simp_s = max(0, rng.gauss(simp_mean, simp_std))
        form_s = max(0, rng.gauss(formality, 0.05))

        q = (ttr_s * 0.3 + (1 - simp_s) * 0.3 + form_s * 0.2 + complexity * 0.2)
        quality_samples.append(q)

    quality_samples.sort()
    n = len(quality_samples)
    quality_distribution = {
        'mean': round(sum(quality_samples) / n, 4),
        'median': round(quality_samples[n // 2], 4),
        'p5': round(quality_samples[int(n * 0.05)], 4),
        'p95': round(quality_samples[int(n * 0.95)], 4),
        'std': round(math.sqrt(sum((q - sum(quality_samples) / n) ** 2 for q in quality_samples) / n), 4)
    }

    return {
        'quality_dimensions': quality_dimensions,
        'overall_quality': round(quality_score, 4),
        'genre': genre,
        'meets_genre_benchmarks': meets_benchmarks,
        'genre_benchmarks': benchmark,
        'quality_distribution': quality_distribution,
        'n_simulations': n_sim,
        'brunets_w': round(brunets, 4),
        'honores_r': round(honores, 2),
        'yules_k': round(yules_k, 4)
    }


def compile_report(classification_data):
    """Compile final classification report.
    Heavy: performs permutation testing to assess significance of
    discourse cluster assignments.
    """
    level = classification_data.get('reading_level', 'unknown')
    grade = classification_data.get('grade_level', 0)
    complexity = classification_data.get('complexity_score', 0)
    variety = classification_data.get('sentence_variety', 0)
    flesch = classification_data.get('flesch_score', 0)
    coleman = classification_data.get('coleman_liau', 0)
    cohesion = classification_data.get('cohesion_density', 0)
    avg_sent = classification_data.get('avg_sentence_words', 0)
    n_sents = classification_data.get('n_sentences', 0)
    clusters = classification_data.get('discourse_clusters', [])

    # Audience recommendations
    audience_map = {
        'elementary': ('general_public', False),
        'middle_school': ('general_public', False),
        'high_school': ('informed_reader', False),
        'undergraduate': ('educated_reader', True),
        'graduate': ('specialist', True),
    }
    audience, simplification_needed = audience_map.get(level, ('specialist', True))

    # Writing style
    if variety > 0.7 and avg_sent > 15:
        style = 'complex_varied'
    elif variety > 0.5:
        style = 'balanced'
    elif avg_sent > 20:
        style = 'dense'
    else:
        style = 'simple'

    # ── Permutation test for cluster significance ─────────────────────
    import random
    rng = random.Random(42)

    cluster_significance = {}
    if clusters and len(clusters) >= 2:
        # Observed: ratio of within-cluster to total variance
        all_sizes = [c['size'] for c in clusters]
        total_sentences = sum(all_sizes)
        observed_intra = sum(c['avg_intra_distance'] * c['size'] for c in clusters) / max(total_sentences, 1)

        # Permutation test: shuffle cluster assignments 200 times
        n_perms = 200
        perm_intras = []
        for _ in range(n_perms):
            # Random partition into same-sized clusters
            shuffled_sizes = list(all_sizes)
            rng.shuffle(shuffled_sizes)
            # Simulate random intra-cluster distances
            perm_intra = sum(
                rng.uniform(0, 1) * s for s in shuffled_sizes
            ) / max(total_sentences, 1)
            perm_intras.append(perm_intra)

        perm_intras.sort()
        # P-value: fraction of permutations with lower intra-cluster distance
        p_value = sum(1 for pi in perm_intras if pi <= observed_intra) / n_perms

        cluster_significance = {
            'observed_intra': round(observed_intra, 4),
            'perm_mean': round(sum(perm_intras) / n_perms, 4),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
            'n_permutations': n_perms
        }

    return {
        'reading_level': level,
        'target_audience': audience,
        'writing_style': style,
        'simplification_needed': simplification_needed,
        'grade_level': round(grade, 1),
        'complexity': round(complexity, 4),
        'sentence_variety': round(variety, 4),
        'flesch_score': round(flesch, 2),
        'coleman_liau': round(coleman, 2),
        'cohesion': round(cohesion, 4),
        'n_sentences': n_sents,
        'n_discourse_segments': len(clusters),
        'cluster_significance': cluster_significance
    }


def lambda_handler(event, context):
    """Stage 4 (final): Summarize — key sentences, keywords, abstract, quality, report."""
    start_time = time.time()
    doc_id = event.get('sentiment', {}).get('doc_id', 'unknown')
    print(f'[Summarizer] Starting on doc {doc_id}')

    # ── Field 1: key_sentences ← sentiment
    t0 = time.time()
    key_sentences = score_sentences_textrank(event.get('sentiment', {}))
    field_1_time = int((time.time() - t0) * 1000)
    print(f'[Summarizer] key_sentences: {key_sentences["n_key_entities"]} key entities in {field_1_time}ms')
    key_sentences_out = {**key_sentences, 'compute_ms': field_1_time, 'doc_id': doc_id}

    # ── Field 2: keywords ← tfidf_vectors
    t0 = time.time()
    keywords = extract_keywords(event.get('tfidf_vectors', {}))
    field_2_time = int((time.time() - t0) * 1000)
    print(f'[Summarizer] keywords: {keywords["n_keywords"]} extracted, {keywords["n_term_clusters"]} clusters in {field_2_time}ms')
    keywords_out = {**keywords, 'compute_ms': field_2_time, 'doc_id': doc_id}

    # ── Field 3: abstract ← topics
    t0 = time.time()
    abstract = generate_abstract(event.get('topics', {}))
    field_3_time = int((time.time() - t0) * 1000)
    print(f'[Summarizer] abstract: primary={abstract["primary_topic"]} in {field_3_time}ms')
    abstract_out = {**abstract, 'compute_ms': field_3_time, 'doc_id': doc_id}

    # ── Field 4: citation_graph ← text_features
    t0 = time.time()
    citation = build_citation_graph(event.get('text_features', {}))
    field_4_time = int((time.time() - t0) * 1000)
    print(f'[Summarizer] citation_graph: quality={citation["overall_quality"]} in {field_4_time}ms')
    citation_graph = {**citation, 'compute_ms': field_4_time, 'doc_id': doc_id}

    # ── Field 5: final_report ← classification
    t0 = time.time()
    report = compile_report(event.get('classification', {}))
    field_5_time = int((time.time() - t0) * 1000)
    print(f'[Summarizer] final_report: level={report["reading_level"]} in {field_5_time}ms')
    final_report = {**report, 'compute_ms': field_5_time, 'doc_id': doc_id}

    total_time = int((time.time() - start_time) * 1000)
    print(f'[Summarizer] COMPLETE in {total_time}ms')

    return {
        'key_sentences': key_sentences_out,
        'keywords': keywords_out,
        'abstract': abstract_out,
        'citation_graph': citation_graph,
        'final_report': final_report
    }
