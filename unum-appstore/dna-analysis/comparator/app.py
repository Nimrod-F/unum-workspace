"""
Comparator - Stage 3 of DNA Sequence Analysis Pipeline

Performs sequence alignment scoring, motif detection, palindrome finding,
repeat comparison, and conservation analysis.

All computation is genuine — no artificial delays.

Each output field depends on exactly ONE input field from the Analyzer:
  alignment_scores ← gc_profile
  motifs           ← codon_usage
  palindromes      ← cpg_islands
  repeat_analysis  ← repeats
  conservation     ← complexity_map
"""
import json
import time
import math
from unum_streaming import StreamingPublisher, set_streaming_output
from collections import Counter


KNOWN_MOTIFS = {
    'TATA': 'TATA_box', 'CAAT': 'CAAT_box', 'AATAAA': 'poly_A_signal',
    'GCCGCC': 'Kozak_like', 'CACGTG': 'E_box', 'CCAAT': 'CCAAT_box',
    'TGAC': 'AP1_site', 'GATA': 'GATA_motif',
}


def compute_alignment_scores(gc_data):
    """Compute alignment-like scores between k-mer profiles.
    Heavy: processes ALL 3-mer and 4-mer GC profiles (64 + 256 entries),
    computes pairwise k-mer distance matrix for 4-mers by obs/exp ratio,
    and runs spectral-like analysis of k-mer enrichment patterns.
    """
    gc_per_3mer = gc_data.get('gc_per_3mer', {})
    gc_per_4mer = gc_data.get('gc_per_4mer', {})
    kmer_4_clusters = gc_data.get('kmer_4_clusters', [])
    chi2_3 = gc_data.get('chi2_3mer', 0)
    chi2_4 = gc_data.get('chi2_4mer', 0)
    gc_bias = gc_data.get('gc_bias', 0)
    entropy_ratio = gc_data.get('entropy_ratio', 0)
    diversity_3 = gc_data.get('diversity_3mer', 0)
    diversity_4 = gc_data.get('diversity_4mer', 0)

    # Log-odds scores for all 4-mers
    total_4 = sum(d['count'] for d in gc_per_4mer.values())
    alignment_scores = []
    for kmer, data in gc_per_4mer.items():
        obs_freq = data['count'] / max(total_4, 1)
        expected = 1 / 256
        if obs_freq > 0 and expected > 0:
            log_odds = math.log2(obs_freq / expected)
        else:
            log_odds = 0
        alignment_scores.append({
            'kmer': kmer, 'count': data['count'], 'gc': data['gc_fraction'],
            'log_odds': round(log_odds, 4), 'chi2': data['chi2'],
            'enriched': log_odds > 1.0, 'depleted': log_odds < -1.0
        })
    alignment_scores.sort(key=lambda x: abs(x['log_odds']), reverse=True)

    # ── Pairwise 4-mer distance matrix based on obs/exp ratio ────────
    # Build feature vectors: each 4-mer is a point in (gc, obs_exp) space
    kmer_list = sorted(gc_per_4mer.keys())
    n = len(kmer_list)
    features = {}
    for kmer in kmer_list:
        d = gc_per_4mer[kmer]
        features[kmer] = [d['gc_fraction'], d['obs_exp_ratio']]

    # Compute distance matrix and find most similar/dissimilar pairs
    similar_pairs = []
    dissimilar_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ki, kj = kmer_list[i], kmer_list[j]
            fi, fj = features[ki], features[kj]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(fi, fj)))
            if dist < 0.1:
                similar_pairs.append({'kmer_a': ki, 'kmer_b': kj, 'distance': round(dist, 4)})
            elif dist > 1.5:
                dissimilar_pairs.append({'kmer_a': ki, 'kmer_b': kj, 'distance': round(dist, 4)})

    similar_pairs.sort(key=lambda x: x['distance'])
    dissimilar_pairs.sort(key=lambda x: x['distance'], reverse=True)

    # 3-mer enrichment profile
    enrichment_3mer = []
    total_3 = sum(d['count'] for d in gc_per_3mer.values())
    for kmer, data in gc_per_3mer.items():
        obs = data['count'] / max(total_3, 1)
        exp = 1 / 64
        lo = math.log2(obs / exp) if obs > 0 else 0
        enrichment_3mer.append({'kmer': kmer, 'log_odds': round(lo, 4), 'gc': data['gc_fraction']})
    enrichment_3mer.sort(key=lambda x: abs(x['log_odds']), reverse=True)

    enriched = sum(1 for a in alignment_scores if a['enriched'])
    depleted = sum(1 for a in alignment_scores if a['depleted'])
    complexity_index = entropy_ratio * diversity_3 * diversity_4

    return {
        'scores': alignment_scores[:100],
        'enrichment_3mer': enrichment_3mer,
        'similar_4mer_pairs': similar_pairs[:50],
        'dissimilar_4mer_pairs': dissimilar_pairs[:50],
        'n_enriched': enriched,
        'n_depleted': depleted,
        'chi2_3mer': chi2_3,
        'chi2_4mer': chi2_4,
        'gc_bias': round(gc_bias, 4),
        'complexity_index': round(complexity_index, 4),
        'entropy_ratio': round(entropy_ratio, 4)
    }


def find_motifs(codon_data):
    """Search for known regulatory motifs using dinucleotide context.
    Heavy: computes expected motif frequencies from dinucleotide model,
    runs Monte Carlo simulation to estimate motif p-values.
    """
    dinuc_abundance = codon_data.get('dinuc_abundance', {})
    cpg_ratio = codon_data.get('cpg_ratio', 0)
    gc_mean = codon_data.get('gc_mean', 0)
    gc_std = codon_data.get('gc_std', 0)
    gc_skew = codon_data.get('gc_skew', [])
    gc_model = codon_data.get('gc_sinusoidal_model', {})

    # Predict motif presence from dinucleotide composition
    motif_predictions = []
    for motif, name in KNOWN_MOTIFS.items():
        score = 0
        n_dinucs = 0
        for i in range(len(motif) - 1):
            dinuc = motif[i:i + 2]
            if dinuc in dinuc_abundance:
                score += dinuc_abundance[dinuc].get('ratio', 1.0)
                n_dinucs += 1
        avg_score = score / max(n_dinucs, 1)
        motif_predictions.append({
            'motif': motif, 'name': name, 'dinuc_score': round(avg_score, 4),
            'likely_present': avg_score > 0.8,
            'gc_content': sum(1 for b in motif if b in 'GC') / len(motif)
        })

    # Dinucleotide step parameters
    step_params = {}
    for dinuc, data in dinuc_abundance.items():
        ratio = data.get('ratio', 1.0)
        bias = 'overrepresented' if ratio > 1.3 else 'underrepresented' if ratio < 0.7 else 'neutral'
        step_params[dinuc] = {'ratio': round(ratio, 4), 'bias': bias}

    # ── Monte Carlo motif p-value estimation ─────────────────────────
    # Simulate random sequences using dinucleotide model and compute
    # expected motif scores
    import random
    rng = random.Random(42)

    n_simulations = 300
    for motif_pred in motif_predictions:
        motif_str = motif_pred['motif']
        observed_score = motif_pred['dinuc_score']
        sim_scores = []

        for _ in range(n_simulations):
            # Randomly permute dinucleotide ratios
            score = 0
            n_d = 0
            for i in range(len(motif_str) - 1):
                # Random dinucleotide ratio from distribution
                rand_ratio = rng.gauss(1.0, 0.3)
                score += max(0, rand_ratio)
                n_d += 1
            sim_score = score / max(n_d, 1)
            sim_scores.append(sim_score)

        # P-value: fraction of simulations >= observed
        p_value = sum(1 for s in sim_scores if s >= observed_score) / n_simulations
        motif_pred['p_value'] = round(p_value, 4)
        motif_pred['significant'] = p_value < 0.05

    # GC skew analysis: find cumulative skew to detect replication origin
    cumulative_skew = []
    cumsum = 0
    for sk in gc_skew:
        cumsum += sk['skew']
        cumulative_skew.append({
            'position': sk['position'],
            'cumulative': round(cumsum, 4)
        })

    # Find min/max of cumulative skew (potential ori/ter positions)
    if cumulative_skew:
        min_cs = min(cumulative_skew, key=lambda x: x['cumulative'])
        max_cs = max(cumulative_skew, key=lambda x: x['cumulative'])
    else:
        min_cs = max_cs = {'position': 0, 'cumulative': 0}

    return {
        'motif_predictions': motif_predictions,
        'n_likely_motifs': sum(1 for m in motif_predictions if m['likely_present']),
        'n_significant_motifs': sum(1 for m in motif_predictions if m.get('significant', False)),
        'dinuc_step_params': step_params,
        'cpg_ratio': round(cpg_ratio, 4),
        'cpg_suppressed': cpg_ratio < 0.8,
        'gc_mean': round(gc_mean, 4),
        'gc_variation': round(gc_std, 4),
        'gc_model': gc_model,
        'cumulative_skew': cumulative_skew[:200],
        'predicted_ori': min_cs,
        'predicted_ter': max_cs
    }


def find_palindromes(cpg_data):
    """Find palindrome-like features from ORF cluster data.
    Heavy: analyzes ALL ORF clusters for symmetry, builds ORF overlap graph,
    computes codon frequency chi-squared test, and runs bootstrap on codon
    usage to establish confidence intervals.
    """
    import random
    rng = random.Random(42)

    clusters = cpg_data.get('orf_clusters', [])
    frame_dist = cpg_data.get('frame_distribution', {})
    coding_density = cpg_data.get('coding_density', 0)
    total_orfs = cpg_data.get('total_orfs', 0)
    codon_freq = cpg_data.get('codon_freq', {})
    prot_hist = cpg_data.get('protein_length_hist', {})

    # Symmetry analysis for each cluster
    symmetry_scores = []
    for cluster in clusters:
        n_orfs = cluster.get('n_orfs', 0)
        span = cluster.get('span', 0)
        density = cluster.get('density', 0)
        avg_len = cluster.get('avg_orf_len', 0)

        density_score = min(density / 5.0, 1.0)
        symmetry = 1.0 if n_orfs % 2 == 0 else 0.7
        regularity = 1.0 - abs(avg_len - 300) / 1000  # 300bp is 'typical'

        symmetry_scores.append({
            'cluster_start': cluster.get('start', 0),
            'cluster_end': cluster.get('end', 0),
            'n_orfs': n_orfs, 'span': span,
            'density_score': round(density_score, 4),
            'symmetry_score': round(symmetry * density_score, 4),
            'regularity': round(max(0, regularity), 4)
        })

    # ── Cluster overlap graph ────────────────────────────────────────
    overlap_edges = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            end_i = clusters[i].get('end', 0)
            start_j = clusters[j].get('start', 0)
            gap = start_j - end_i
            if gap < 1000:
                overlap_edges.append({
                    'cluster_a': i, 'cluster_b': j,
                    'gap': gap, 'overlapping': gap < 0
                })

    # ── Codon frequency chi-squared test ─────────────────────────────
    aa_total = sum(codon_freq.values())
    n_aa_types = len(codon_freq)
    expected_per_aa = aa_total / max(n_aa_types, 1)
    chi2_codon = sum(
        (count - expected_per_aa) ** 2 / max(expected_per_aa, 1)
        for count in codon_freq.values()
    )

    # ── Bootstrap codon usage confidence intervals ───────────────────
    # Resample codon frequencies N times to get CI on chi-squared
    codon_list = []
    for aa, count in codon_freq.items():
        codon_list.extend([aa] * count)

    n_bootstrap = 200
    bootstrap_chi2 = []
    if codon_list:
        # Use a smaller sample size to keep computation reasonable
        sample_size = min(len(codon_list), 500)
        for _ in range(n_bootstrap):
            sample = [codon_list[rng.randint(0, len(codon_list) - 1)]
                      for _ in range(sample_size)]
            sample_counts = Counter(sample)
            sample_total = sum(sample_counts.values())
            exp = sample_total / max(n_aa_types, 1)
            bchi2 = sum((c - exp) ** 2 / max(exp, 1) for c in sample_counts.values())
            bootstrap_chi2.append(bchi2)

    bootstrap_chi2.sort()

    # ── Pairwise codon co-occurrence analysis ─────────────────────────
    # Build a codon-codon co-occurrence matrix from clusters
    aa_keys = sorted(codon_freq.keys())
    n_keys = min(len(aa_keys), 20)
    cooccurrence = [[0] * n_keys for _ in range(n_keys)]
    for cluster in clusters:
        orfs_in_cluster = cluster.get('orf_list', cluster.get('n_orfs', 0))
        if isinstance(orfs_in_cluster, int):
            # Simulate ORFs from codon frequencies
            for ki in range(n_keys):
                for kj in range(ki + 1, n_keys):
                    co = codon_freq.get(aa_keys[ki], 0) * codon_freq.get(aa_keys[kj], 0)
                    cooccurrence[ki][kj] += co
                    cooccurrence[kj][ki] += co

    # Compute PMI-like scores from co-occurrence
    total_co = sum(sum(row) for row in cooccurrence) or 1
    pmi_scores = []
    for ki in range(n_keys):
        row_sum = sum(cooccurrence[ki]) or 1
        for kj in range(ki + 1, n_keys):
            col_sum = sum(cooccurrence[r][kj] for r in range(n_keys)) or 1
            joint = cooccurrence[ki][kj] / total_co
            marginal = (row_sum / total_co) * (col_sum / total_co)
            pmi = math.log(joint / max(marginal, 1e-10)) if joint > 0 else 0
            pmi_scores.append({
                'aa_a': aa_keys[ki], 'aa_b': aa_keys[kj],
                'pmi': round(pmi, 4)
            })

    # Frame bias analysis
    total_frame = sum(frame_dist.values()) if frame_dist else 1
    frame_entropy = -sum(
        (v / total_frame) * math.log2(v / total_frame) for v in frame_dist.values() if v > 0
    ) if frame_dist else 0
    max_frame_entropy = math.log2(3) if total_frame > 0 else 0
    frame_balance = frame_entropy / max(max_frame_entropy, 1)

    return {
        'symmetry_scores': symmetry_scores,
        'n_symmetric_clusters': sum(1 for s in symmetry_scores if s['symmetry_score'] > 0.5),
        'cluster_overlaps': overlap_edges[:30],
        'codon_chi2': round(chi2_codon, 2),
        'codon_freq': codon_freq,
        'codon_pmi_scores': pmi_scores[:50],
        'protein_length_hist': prot_hist,
        'frame_entropy': round(frame_entropy, 4),
        'frame_balance': round(frame_balance, 4),
        'coding_density': round(coding_density, 4),
        'total_orfs': total_orfs,
        'frame_distribution': frame_dist
    }


def analyze_repeats(repeat_data):
    """Analyze repeat patterns and genome organization.
    Heavy: uses merge history and isochore data to reconstruct
    genome domain boundaries, runs permutation test for heterogeneity
    significance, and computes domain transition probabilities.
    """
    import random
    rng = random.Random(42)

    low_complexity = repeat_data.get('low_complexity_regions', 0)
    high_gc = repeat_data.get('high_gc_regions', 0)
    low_gc = repeat_data.get('low_gc_regions', 0)
    avg_complexity = repeat_data.get('avg_complexity', 0)
    gc_heterogeneity = repeat_data.get('gc_heterogeneity', 0)
    isochores = repeat_data.get('isochores', [])
    merge_history = repeat_data.get('merge_history', [])
    n_segments = repeat_data.get('n_segments_analyzed', 0)

    # Genome organization classification
    if gc_heterogeneity > 0.15:
        organization = 'heterogeneous'
    elif gc_heterogeneity > 0.08:
        organization = 'moderately_heterogeneous'
    else:
        organization = 'homogeneous'

    # Isochore analysis
    n_isochores = len(isochores)
    isochore_coverage = sum(iso.get('n_segments', 0) for iso in isochores) / max(n_segments, 1)

    repeat_score = (low_complexity / max(n_segments, 1)) * 100

    domain_types = {
        'gc_rich': high_gc, 'at_rich': low_gc,
        'balanced': n_segments - high_gc - low_gc,
        'low_complexity': low_complexity
    }

    # ── Dendrogram analysis from merge history ──────────────────────
    merge_distances = [m['distance'] for m in merge_history]
    if len(merge_distances) > 1:
        sorted_dists = sorted(merge_distances)
        gaps = []
        for i in range(1, len(sorted_dists)):
            gaps.append({
                'level': i,
                'distance': sorted_dists[i],
                'gap': round(sorted_dists[i] - sorted_dists[i - 1], 4)
            })
        gaps.sort(key=lambda x: x['gap'], reverse=True)
        optimal_threshold = gaps[0]['distance'] if gaps else 0.1
        optimal_clusters = sum(1 for d in sorted_dists if d > optimal_threshold) + 1
    else:
        optimal_threshold = 0.1
        optimal_clusters = n_segments
        gaps = []

    # Isochore GC distribution
    iso_gc_values = [iso['avg_gc'] for iso in isochores]
    if iso_gc_values:
        iso_gc_mean = sum(iso_gc_values) / len(iso_gc_values)
        iso_gc_range = max(iso_gc_values) - min(iso_gc_values) if len(iso_gc_values) > 1 else 0
    else:
        iso_gc_mean = 0
        iso_gc_range = 0

    # ── Permutation test for GC heterogeneity significance ──────────
    # Generate null distribution by shuffling segment assignments
    n_perms = 500
    perm_hets = []
    for _ in range(n_perms):
        # Simulate GC values under null hypothesis (uniform)
        sim_gc = [rng.gauss(iso_gc_mean if iso_gc_mean > 0 else 0.5, 0.05)
                  for _ in range(max(n_segments, 10))]
        sim_mean = sum(sim_gc) / len(sim_gc)
        sim_het = math.sqrt(sum((g - sim_mean) ** 2 for g in sim_gc) / len(sim_gc))
        perm_hets.append(sim_het)

    perm_hets.sort()
    het_pvalue = sum(1 for h in perm_hets if h >= gc_heterogeneity) / n_perms

    # ── Domain transition matrix ─────────────────────────────────────
    # Build transition probabilities between domain types
    domain_labels = ['gc_rich', 'at_rich', 'balanced', 'low_complexity']
    n_labels = len(domain_labels)
    trans_counts = [[0] * n_labels for _ in range(n_labels)]

    # Simulate domain sequence from isochore assignments
    segment_domains = []
    for iso in isochores:
        gc = iso.get('avg_gc', 0.5)
        n_segs = iso.get('n_segments', 1)
        if gc > 0.55:
            label_idx = 0
        elif gc < 0.45:
            label_idx = 1
        else:
            label_idx = 2
        segment_domains.extend([label_idx] * n_segs)

    for i in range(1, len(segment_domains)):
        trans_counts[segment_domains[i - 1]][segment_domains[i]] += 1

    # Normalize to probabilities
    trans_probs = {}
    for i in range(n_labels):
        row_sum = sum(trans_counts[i]) or 1
        for j in range(n_labels):
            key = f'{domain_labels[i]}_to_{domain_labels[j]}'
            trans_probs[key] = round(trans_counts[i][j] / row_sum, 4)

    return {
        'organization': organization,
        'gc_heterogeneity': round(gc_heterogeneity, 4),
        'het_pvalue': round(het_pvalue, 4),
        'het_significant': het_pvalue < 0.05,
        'n_isochores': n_isochores,
        'isochore_coverage': round(isochore_coverage, 4),
        'isochore_gc_mean': round(iso_gc_mean, 4),
        'isochore_gc_range': round(iso_gc_range, 4),
        'repeat_score': round(repeat_score, 2),
        'domain_types': domain_types,
        'domain_transitions': trans_probs,
        'avg_complexity': round(avg_complexity, 4),
        'optimal_clusters': optimal_clusters,
        'optimal_threshold': round(optimal_threshold, 4),
        'dendrogram_gaps': gaps[:10]
    }


def assess_conservation(complexity_data):
    """Assess sequence conservation from entropy/complexity patterns.
    Heavy: processes ALL positional entropy values and ALL trinucleotide
    transition entropies, runs multi-scale sliding window analysis,
    computes pairwise position similarity, and builds conservation landscape.
    """
    position_entropy = complexity_data.get('position_entropy', [])
    smoothed_entropy = complexity_data.get('smoothed_entropy', [])
    avg_entropy = complexity_data.get('avg_entropy', 0)
    entropy_range = complexity_data.get('entropy_range', 0)
    trinuc_entropy = complexity_data.get('trinuc_entropy', 0)
    trinuc_trans_entropy = complexity_data.get('trinuc_trans_entropy', {})
    context_ratio = complexity_data.get('context_bias_ratio', 0)
    max_context = complexity_data.get('max_context', {})
    n_transitions = complexity_data.get('n_transitions', 0)

    # Conservation scoring per position
    max_possible = 2.0
    conserved_positions = []
    variable_positions = []
    conservation_values = []
    for pe in position_entropy:
        norm_entropy = pe['entropy'] / max(max_possible, 1)
        conservation = 1.0 - norm_entropy
        conservation_values.append(conservation)
        if conservation > 0.3:
            conserved_positions.append({
                'position': pe['position'], 'conservation': round(conservation, 4),
                'dominant': pe['dominant']
            })
        elif conservation < 0.1:
            variable_positions.append({
                'position': pe['position'], 'conservation': round(conservation, 4)
            })

    # ── Multi-scale sliding window conservation ──────────────────────
    # Compute conservation scores at multiple window sizes
    window_sizes = [5, 10, 20, 50]
    multi_scale_profiles = {}
    for ws in window_sizes:
        profile = []
        for i in range(len(conservation_values) - ws + 1):
            window = conservation_values[i:i + ws]
            w_mean = sum(window) / ws
            w_var = sum((v - w_mean) ** 2 for v in window) / ws
            profile.append({
                'position': i, 'mean_conservation': round(w_mean, 4),
                'variance': round(w_var, 6)
            })
        multi_scale_profiles[f'window_{ws}'] = len(profile)

    # ── Pairwise position entropy similarity (sample 200 positions) ──
    n_pos = len(conservation_values)
    step = max(1, n_pos // 200)
    sampled = conservation_values[::step][:200]
    n_s = len(sampled)
    similarity_sum = 0
    similarity_count = 0
    for i in range(n_s):
        for j in range(i + 1, n_s):
            sim = 1.0 - abs(sampled[i] - sampled[j])
            similarity_sum += sim
            similarity_count += 1
    avg_similarity = similarity_sum / max(similarity_count, 1)

    # Smoothed conservation regions
    conservation_regions = []
    if smoothed_entropy:
        region_start = None
        for se in smoothed_entropy:
            cons = 1.0 - se['smoothed_entropy'] / max(max_possible, 1)
            if cons > 0.2:
                if region_start is None:
                    region_start = se['position']
            else:
                if region_start is not None:
                    conservation_regions.append({
                        'start': region_start, 'end': se['position'],
                        'length': se['position'] - region_start
                    })
                    region_start = None
        if region_start is not None:
            conservation_regions.append({
                'start': region_start,
                'end': smoothed_entropy[-1]['position'],
                'length': smoothed_entropy[-1]['position'] - region_start
            })

    # ── Trinucleotide context dependency analysis ────────────────────
    context_dep_score = 0
    if trinuc_trans_entropy:
        entropies = list(trinuc_trans_entropy.values())
        mean_e = sum(entropies) / len(entropies)
        var_e = sum((e - mean_e) ** 2 for e in entropies) / max(len(entropies) - 1, 1)
        context_dep_score = math.sqrt(var_e)

        # Pairwise trinucleotide entropy distance matrix
        trinuc_keys = sorted(trinuc_trans_entropy.keys())
        n_tri = len(trinuc_keys)
        trinuc_dist_sum = 0
        trinuc_pairs = 0
        for i in range(n_tri):
            for j in range(i + 1, n_tri):
                d = abs(trinuc_trans_entropy[trinuc_keys[i]] - trinuc_trans_entropy[trinuc_keys[j]])
                trinuc_dist_sum += d
                trinuc_pairs += 1
        avg_trinuc_dist = trinuc_dist_sum / max(trinuc_pairs, 1)

        sorted_trinucs = sorted(trinuc_trans_entropy.items(), key=lambda x: x[1])
        most_dependent = sorted_trinucs[:10]
        least_dependent = sorted_trinucs[-10:]
    else:
        most_dependent = []
        least_dependent = []
        avg_trinuc_dist = 0

    # Overall conservation score
    n_conserved = len(conserved_positions)
    n_variable = len(variable_positions)
    n_total = len(position_entropy)
    conservation_score = n_conserved / max(n_total, 1)

    # Genome type classification
    if conservation_score > 0.3:
        genome_type = 'structured'
    elif conservation_score > 0.1:
        genome_type = 'moderately_structured'
    else:
        genome_type = 'random_like'

    return {
        'conservation_score': round(conservation_score, 4),
        'n_conserved_positions': n_conserved,
        'n_variable_positions': n_variable,
        'conservation_regions': conservation_regions[:30],
        'context_dependency': round(context_dep_score, 4),
        'most_context_dependent': [{'trinuc': t, 'entropy': e} for t, e in most_dependent],
        'least_context_dependent': [{'trinuc': t, 'entropy': e} for t, e in least_dependent],
        'genome_type': genome_type,
        'avg_positional_entropy': round(avg_entropy, 4),
        'avg_position_similarity': round(avg_similarity, 4),
        'avg_trinuc_distance': round(avg_trinuc_dist, 4),
        'multi_scale_profiles': multi_scale_profiles,
        'entropy_range': round(entropy_range, 4),
        'trinuc_transition_entropy': round(trinuc_entropy, 4),
        'n_transitions': n_transitions,
        'dominant_context': dict(max_context)
    }


def lambda_handler(event, context):
    """Stage 3: Compare and analyze DNA features."""


    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = event.get('Session', '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="ComparatorFunction",
        field_names=["alignment_scores", "motifs", "palindromes", "repeat_analysis", "conservation"]
    )
    start_time = time.time()
    seq_id = event.get('gc_profile', {}).get('seq_id', 'unknown')
    print(f'[Comparator] Starting on {seq_id}')

    # ── Field 1: alignment_scores ← gc_profile
    t0 = time.time()
    alignment = compute_alignment_scores(event.get('gc_profile', {}))
    field_1_time = int((time.time() - t0) * 1000)
    print(f'[Comparator] alignment: {alignment["n_enriched"]} enriched, {len(alignment["similar_4mer_pairs"])} similar pairs in {field_1_time}ms')
    alignment_scores = {**alignment, 'compute_ms': field_1_time, 'seq_id': seq_id}
    _streaming_publisher.publish('alignment_scores', alignment_scores)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()
    # ── Field 2: motifs ← codon_usage
    t0 = time.time()
    motifs = find_motifs(event.get('codon_usage', {}))
    field_2_time = int((time.time() - t0) * 1000)
    print(f'[Comparator] motifs: {motifs["n_likely_motifs"]} likely, {motifs["n_significant_motifs"]} significant in {field_2_time}ms')
    motifs_out = {**motifs, 'compute_ms': field_2_time, 'seq_id': seq_id}
    _streaming_publisher.publish('motifs', motifs_out)

    # ── Field 3: palindromes ← cpg_islands
    t0 = time.time()
    palindromes = find_palindromes(event.get('cpg_islands', {}))
    field_3_time = int((time.time() - t0) * 1000)
    print(f'[Comparator] palindromes: frame_balance={palindromes["frame_balance"]}, codon_chi2={palindromes["codon_chi2"]} in {field_3_time}ms')
    palindromes_out = {**palindromes, 'compute_ms': field_3_time, 'seq_id': seq_id}
    _streaming_publisher.publish('palindromes', palindromes_out)

    # ── Field 4: repeat_analysis ← repeats
    t0 = time.time()
    repeat_analysis = analyze_repeats(event.get('repeats', {}))
    field_4_time = int((time.time() - t0) * 1000)
    print(f'[Comparator] repeats: org={repeat_analysis["organization"]}, opt_k={repeat_analysis["optimal_clusters"]} in {field_4_time}ms')
    repeat_analysis_out = {**repeat_analysis, 'compute_ms': field_4_time, 'seq_id': seq_id}
    _streaming_publisher.publish('repeat_analysis', repeat_analysis_out)

    # ── Field 5: conservation ← complexity_map
    t0 = time.time()
    conservation = assess_conservation(event.get('complexity_map', {}))
    field_5_time = int((time.time() - t0) * 1000)
    print(f'[Comparator] conservation: type={conservation["genome_type"]}, deps={conservation["context_dependency"]} in {field_5_time}ms')
    conservation_out = {**conservation, 'compute_ms': field_5_time, 'seq_id': seq_id}
    _streaming_publisher.publish('conservation', conservation_out)

    total_time = int((time.time() - start_time) * 1000)
    print(f'[Comparator] COMPLETE in {total_time}ms')

    return {
        'alignment_scores': alignment_scores,
        'motifs': motifs_out,
        'palindromes': palindromes_out,
        'repeat_analysis': repeat_analysis_out,
        'conservation': conservation_out
    }
