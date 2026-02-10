"""
Analyzer - Stage 2 of DNA Sequence Analysis Pipeline

Performs GC content profiling, codon usage analysis, CpG island detection,
repeat finding, and sequence complexity scoring.

All computation is genuine — no artificial delays.

Each output field depends on exactly ONE input field from the Reader:
  gc_profile      ← kmers
  codon_usage     ← base_composition
  cpg_islands     ← open_reading_frames
  repeats         ← segments
  complexity_map  ← nucleotide_freq
"""
import json
import time
import math
from unum_streaming import StreamingPublisher, set_streaming_output
from collections import Counter


def compute_gc_profile(kmer_data):
    """Compute GC content profile from k-mer distribution.
    Heavy: processes ALL 3-mer and 4-mer counts, builds observed/expected
    ratio matrix, runs chi-squared test for each k-mer, and clusters
    k-mers by GC content using iterative k-means.
    """
    kmer_3_all = kmer_data.get('kmer_3_all', {})
    kmer_4_all = kmer_data.get('kmer_4_all', {})
    kmer_5_top = kmer_data.get('kmer_5_top', [])
    entropy_3 = kmer_data.get('entropy_3', 0)
    diversity_3 = kmer_data.get('diversity_3', 0)
    diversity_4 = kmer_data.get('diversity_4', 0)

    # GC content per 3-mer
    gc_per_3mer = {}
    total_3 = sum(kmer_3_all.values())
    for kmer, count in kmer_3_all.items():
        gc_frac = sum(1 for b in kmer if b in 'GC') / len(kmer)
        expected = total_3 / 64  # Uniform expectation
        obs_exp = count / max(expected, 1)
        # Chi-squared contribution
        chi2 = (count - expected) ** 2 / max(expected, 1)
        gc_per_3mer[kmer] = {
            'count': count, 'gc_fraction': round(gc_frac, 3),
            'obs_exp_ratio': round(obs_exp, 4), 'chi2': round(chi2, 4)
        }

    # GC content per 4-mer with full analysis
    gc_per_4mer = {}
    total_4 = sum(kmer_4_all.values())
    for kmer, count in kmer_4_all.items():
        gc_frac = sum(1 for b in kmer if b in 'GC') / len(kmer)
        expected = total_4 / 256
        obs_exp = count / max(expected, 1)
        chi2 = (count - expected) ** 2 / max(expected, 1)
        gc_per_4mer[kmer] = {
            'count': count, 'gc_fraction': round(gc_frac, 3),
            'obs_exp_ratio': round(obs_exp, 4), 'chi2': round(chi2, 4)
        }

    # Total chi-squared statistic
    chi2_3 = sum(v['chi2'] for v in gc_per_3mer.values())
    chi2_4 = sum(v['chi2'] for v in gc_per_4mer.values())

    # GC-rich vs AT-rich balance
    gc_rich_count = sum(d['count'] for d in gc_per_3mer.values() if d['gc_fraction'] > 0.6)
    at_rich_count = sum(d['count'] for d in gc_per_3mer.values() if d['gc_fraction'] < 0.4)
    gc_bias = (gc_rich_count - at_rich_count) / max(total_3, 1)

    # ── K-means clustering of 4-mers by GC + obs/exp ────────────────
    import random
    rng = random.Random(42)

    # Feature vectors: (gc_fraction, obs_exp_ratio_normalized)
    kmer_features = []
    kmer_names = []
    for kmer, data in gc_per_4mer.items():
        kmer_features.append([data['gc_fraction'], data['obs_exp_ratio']])
        kmer_names.append(kmer)

    # Normalize features
    if kmer_features:
        max_oe = max(f[1] for f in kmer_features)
        for f in kmer_features:
            f[1] = f[1] / max(max_oe, 1)

    k = 4
    n = len(kmer_features)
    clusters_result = []
    if n >= k:
        # Initialize centroids
        indices = list(range(n))
        rng.shuffle(indices)
        centroids = [list(kmer_features[indices[i]]) for i in range(k)]
        assignments = [0] * n

        for _ in range(20):
            # Assign
            for i in range(n):
                best_c = 0
                best_d = float('inf')
                for ci in range(k):
                    d = sum((a - b) ** 2 for a, b in zip(kmer_features[i], centroids[ci]))
                    if d < best_d:
                        best_d = d
                        best_c = ci
                assignments[i] = best_c
            # Update centroids
            for ci in range(k):
                members = [kmer_features[i] for i in range(n) if assignments[i] == ci]
                if members:
                    centroids[ci] = [sum(m[d] for m in members) / len(members) for d in range(2)]

        # Build cluster profiles
        for ci in range(k):
            member_kmers = [kmer_names[i] for i in range(n) if assignments[i] == ci]
            member_gc = [kmer_features[i][0] for i in range(n) if assignments[i] == ci]
            if member_gc:
                clusters_result.append({
                    'cluster': ci,
                    'size': len(member_kmers),
                    'avg_gc': round(sum(member_gc) / len(member_gc), 4),
                    'centroid': [round(c, 4) for c in centroids[ci]],
                    'sample_kmers': member_kmers[:10]
                })

    # 5-mer analysis
    kmer_5_gc = []
    for kmer, count in kmer_5_top:
        gc = sum(1 for b in kmer if b in 'GC') / len(kmer)
        kmer_5_gc.append({'kmer': kmer, 'count': count, 'gc': round(gc, 3)})

    max_entropy = math.log2(64)
    entropy_ratio = entropy_3 / max(max_entropy, 1)

    return {
        'gc_per_3mer': gc_per_3mer,
        'gc_per_4mer': gc_per_4mer,
        'chi2_3mer': round(chi2_3, 2),
        'chi2_4mer': round(chi2_4, 2),
        'gc_bias': round(gc_bias, 4),
        'gc_rich_fraction': round(gc_rich_count / max(total_3, 1), 4),
        'at_rich_fraction': round(at_rich_count / max(total_3, 1), 4),
        'kmer_4_clusters': clusters_result,
        'kmer_5_gc_profile': kmer_5_gc[:100],
        'entropy_ratio': round(entropy_ratio, 4),
        'diversity_3mer': round(diversity_3, 4),
        'diversity_4mer': round(diversity_4, 4)
    }


def analyze_codon_usage(base_data):
    """Analyze codon usage from base composition data.
    Heavy: processes ALL GC windows to fit a sinusoidal model via
    least-squares grid search, and computes dinucleotide relative abundance.
    """
    dinuc_freqs = base_data.get('dinuc_freqs', {})
    base_freqs = base_data.get('base_freqs', {})
    gc_windows = base_data.get('gc_windows', [])
    total_bases = base_data.get('total_bases', 0)

    # Relative dinucleotide abundance (observed/expected)
    dinuc_abundance = {}
    for dinuc, freq in dinuc_freqs.items():
        if len(dinuc) == 2:
            expected = base_freqs.get(dinuc[0], 0.25) * base_freqs.get(dinuc[1], 0.25)
            ratio = freq / max(expected, 1e-10)
            dinuc_abundance[dinuc] = {
                'observed': round(freq, 6), 'expected': round(expected, 6),
                'ratio': round(ratio, 4)
            }

    # CpG observed/expected ratio
    cpg_obs = dinuc_freqs.get('CG', 0)
    cpg_exp = base_freqs.get('C', 0.25) * base_freqs.get('G', 0.25)
    cpg_ratio = cpg_obs / max(cpg_exp, 1e-10)

    # GC content variation along sequence
    gc_values = [w['gc'] for w in gc_windows]
    if gc_values:
        gc_mean = sum(gc_values) / len(gc_values)
        gc_var = sum((g - gc_mean) ** 2 for g in gc_values) / max(len(gc_values) - 1, 1)
        gc_std = math.sqrt(gc_var)
    else:
        gc_mean, gc_std = 0, 0

    # GC skew per window: (G-C)/(G+C)
    # Requires re-examining window data; approximate from gc and base freqs
    gc_skew_values = []
    g_frac_global = base_freqs.get('G', 0.25)
    c_frac_global = base_freqs.get('C', 0.25)
    gc_skew_global = (g_frac_global - c_frac_global) / max(g_frac_global + c_frac_global, 1e-10)

    for w in gc_windows:
        # Approximate per-window skew from deviation around global
        deviation = w['gc'] - gc_mean
        skew_approx = gc_skew_global + deviation * 0.5
        gc_skew_values.append({'position': w['position'], 'skew': round(skew_approx, 4)})

    # ── Sinusoidal model fitting for GC variation (grid search) ──────
    # Model: gc(pos) = A * sin(2π * pos / period + phase) + offset
    # Grid search over period and phase
    best_error = float('inf')
    best_params = {'A': 0, 'period': 1000, 'phase': 0, 'offset': gc_mean}
    positions = [w['position'] for w in gc_windows]

    if len(gc_values) > 10:
        max_pos = max(positions) if positions else 1
        for period_k in range(1, 21):  # 20 period candidates
            period = max_pos / max(period_k, 1)
            if period < 1:
                continue
            for phase_k in range(0, 12):  # 12 phase candidates
                phase = phase_k * math.pi / 6
                for amp_k in range(1, 11):  # 10 amplitude candidates
                    amp = gc_std * amp_k / 5
                    error = 0
                    for i, pos in enumerate(positions):
                        predicted = amp * math.sin(2 * math.pi * pos / period + phase) + gc_mean
                        error += (gc_values[i] - predicted) ** 2
                    if error < best_error:
                        best_error = error
                        best_params = {
                            'A': round(amp, 6), 'period': round(period, 1),
                            'phase': round(phase, 4), 'offset': round(gc_mean, 6)
                        }

    return {
        'dinuc_abundance': dinuc_abundance,
        'cpg_ratio': round(cpg_ratio, 4),
        'gc_mean': round(gc_mean, 4),
        'gc_std': round(gc_std, 4),
        'gc_skew': gc_skew_values[:200],
        'gc_sinusoidal_model': best_params,
        'gc_model_mse': round(best_error / max(len(gc_values), 1), 8),
        'total_bases': total_bases,
        'n_windows': len(gc_windows)
    }


def detect_cpg_islands(orf_data):
    """Detect CpG island-like features from ORF data.
    Heavy: processes ALL ORFs, builds ORF distance matrix,
    runs DBSCAN-like density clustering, and computes codon frequency tables.
    """
    orfs = orf_data.get('orfs', [])
    total_orfs = orf_data.get('total_orfs', 0)
    coding_density = orf_data.get('coding_density', 0)

    # Build ORF distance matrix and cluster by proximity
    orf_positions = sorted(orfs, key=lambda x: x['start'])

    # DBSCAN-like clustering: eps=500, min_pts=2
    eps = 500
    visited = [False] * len(orf_positions)
    clusters = []

    for i in range(len(orf_positions)):
        if visited[i]:
            continue
        # Find all ORFs within eps distance
        neighbors = []
        for j in range(len(orf_positions)):
            if i != j and not visited[j]:
                dist = abs(orf_positions[i]['start'] - orf_positions[j]['start'])
                if dist <= eps:
                    neighbors.append(j)

        if len(neighbors) >= 1:  # min_pts = 2 (including self)
            cluster = [i]
            visited[i] = True
            queue = list(neighbors)
            while queue:
                q = queue.pop(0)
                if not visited[q]:
                    visited[q] = True
                    cluster.append(q)
                    # Expand cluster
                    for j in range(len(orf_positions)):
                        if not visited[j]:
                            d = abs(orf_positions[q]['start'] - orf_positions[j]['start'])
                            if d <= eps and j not in queue:
                                queue.append(j)

            cluster_orfs = [orf_positions[idx] for idx in cluster]
            cluster_start = min(o['start'] for o in cluster_orfs)
            cluster_end = max(o['end'] for o in cluster_orfs)
            clusters.append({
                'start': cluster_start,
                'end': cluster_end,
                'n_orfs': len(cluster_orfs),
                'span': cluster_end - cluster_start,
                'density': len(cluster_orfs) / max((cluster_end - cluster_start) / 1000, 0.1),
                'avg_orf_len': round(sum(o['length'] for o in cluster_orfs) / len(cluster_orfs), 1)
            })
        else:
            visited[i] = True

    # Frame distribution
    frame_dist = Counter(o['frame'] for o in orfs)

    # Codon usage table from protein previews
    codon_freq = Counter()
    for orf in orfs:
        preview = orf.get('protein_preview', '')
        for aa in preview:
            codon_freq[aa] += 1

    # Protein length distribution analysis
    prot_lengths = [o['protein_length'] for o in orfs]
    if prot_lengths:
        avg_prot = sum(prot_lengths) / len(prot_lengths)
        max_prot = max(prot_lengths)
        # Length histogram
        bins = [0, 30, 50, 75, 100, 150, 200, 500]
        length_hist = Counter()
        for pl in prot_lengths:
            for bi in range(len(bins) - 1):
                if bins[bi] <= pl < bins[bi + 1]:
                    length_hist[f'{bins[bi]}-{bins[bi+1]}'] += 1
                    break
            else:
                length_hist[f'{bins[-1]}+'] += 1
    else:
        avg_prot, max_prot = 0, 0
        length_hist = {}

    return {
        'orf_clusters': clusters[:30],
        'n_clusters': len(clusters),
        'frame_distribution': dict(frame_dist),
        'codon_freq': dict(codon_freq),
        'protein_length_hist': dict(length_hist),
        'coding_density': round(coding_density, 4),
        'avg_protein_length': round(avg_prot, 1),
        'max_protein_length': max_prot,
        'total_orfs': total_orfs
    }


def find_repeats(segment_data):
    """Find repetitive elements in sequence segments.
    Heavy: computes pairwise segment similarity using composition vectors,
    runs hierarchical clustering, and identifies isochore boundaries.
    """
    segments = segment_data.get('segment_stats', [])
    total_length = segment_data.get('total_length', 0)

    complexities = [s['complexity'] for s in segments]
    gc_contents = [s['gc_content'] for s in segments]

    low_complexity = [s for s in segments if s['complexity'] < 0.85]
    high_gc = [s for s in segments if s['gc_content'] > 0.6]
    low_gc = [s for s in segments if s['gc_content'] < 0.4]

    if complexities:
        avg_complexity = sum(complexities) / len(complexities)
        complexity_var = sum((c - avg_complexity) ** 2 for c in complexities) / max(len(complexities) - 1, 1)
    else:
        avg_complexity, complexity_var = 0, 0

    if gc_contents:
        gc_mean = sum(gc_contents) / len(gc_contents)
        gc_range = max(gc_contents) - min(gc_contents)
    else:
        gc_mean, gc_range = 0, 0

    # ── Pairwise segment similarity (composition distance) ──────────
    n = len(segments)
    seg_distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            gc_diff = abs(segments[i]['gc_content'] - segments[j]['gc_content'])
            comp_diff = abs(segments[i]['complexity'] - segments[j]['complexity'])
            dist = math.sqrt(gc_diff ** 2 + comp_diff ** 2)
            seg_distances[(i, j)] = dist

    # ── Single-linkage clustering on segments ───────────────────────
    cluster_map = {i: i for i in range(n)}
    clusters_dict = {i: [i] for i in range(n)}
    merge_history = []
    next_id = n

    max_merges = min(n - 1, 50)
    for _ in range(max_merges):
        active = set(clusters_dict.keys())
        ac_list = sorted(active)
        best_pair = None
        best_dist = float('inf')
        for ai in range(len(ac_list)):
            for bi in range(ai + 1, len(ac_list)):
                ci, cj = ac_list[ai], ac_list[bi]
                # Single linkage
                min_d = float('inf')
                for si in clusters_dict[ci]:
                    for sj in clusters_dict[cj]:
                        key = (min(si, sj), max(si, sj))
                        d = seg_distances.get(key, 1.0)
                        min_d = min(min_d, d)
                if min_d < best_dist:
                    best_dist = min_d
                    best_pair = (ci, cj)

        if best_pair is None or best_dist > 0.3:
            break

        ci, cj = best_pair
        merged = clusters_dict[ci] + clusters_dict[cj]
        clusters_dict[next_id] = merged
        merge_history.append({'merged': [ci, cj], 'distance': round(best_dist, 4), 'size': len(merged)})
        del clusters_dict[ci]
        del clusters_dict[cj]
        next_id += 1

    # Isochore identification from clusters
    isochores = []
    for cid, members in clusters_dict.items():
        if len(members) > 1:
            member_segs = [segments[m] for m in members if m < len(segments)]
            if member_segs:
                avg_gc = sum(s['gc_content'] for s in member_segs) / len(member_segs)
                isochores.append({
                    'n_segments': len(member_segs),
                    'avg_gc': round(avg_gc, 4),
                    'avg_complexity': round(sum(s['complexity'] for s in member_segs) / len(member_segs), 4),
                    'segment_ids': [s['id'] for s in member_segs]
                })

    return {
        'low_complexity_regions': len(low_complexity),
        'high_gc_regions': len(high_gc),
        'low_gc_regions': len(low_gc),
        'avg_complexity': round(avg_complexity, 4),
        'complexity_variance': round(complexity_var, 6),
        'gc_heterogeneity': round(gc_range, 4),
        'isochores': isochores[:20],
        'merge_history': merge_history[:20],
        'n_segment_clusters': len([c for c in clusters_dict.values() if len(c) > 1]),
        'n_segments_analyzed': n
    }


def compute_complexity_map(freq_data):
    """Compute sequence complexity map from nucleotide frequencies.
    Heavy: processes ALL positional frequencies and ALL trinucleotide contexts,
    computes sliding-window Shannon entropy across all positions,
    and builds a trinucleotide transition matrix.
    """
    position_freq = freq_data.get('position_freq', [])
    trinuc_contexts = freq_data.get('trinuc_contexts', {})

    # Per-position Shannon entropy
    position_entropy = []
    for pf in position_freq:
        freqs = [pf.get(b, 0) for b in 'ACGT']
        entropy = 0
        for f in freqs:
            if f > 0:
                entropy -= f * math.log2(f)
        position_entropy.append({
            'position': pf['position'],
            'entropy': round(entropy, 4),
            'dominant': max('ACGT', key=lambda b: pf.get(b, 0))
        })

    # Sliding window entropy (window=20 positions)
    window = 20
    smoothed_entropy = []
    for i in range(len(position_entropy) - window):
        window_vals = [position_entropy[j]['entropy'] for j in range(i, i + window)]
        avg_e = sum(window_vals) / window
        smoothed_entropy.append({
            'position': position_entropy[i + window // 2]['position'],
            'smoothed_entropy': round(avg_e, 4)
        })

    # Overall entropy stats
    all_entropy = [pe['entropy'] for pe in position_entropy]
    if all_entropy:
        avg_entropy = sum(all_entropy) / len(all_entropy)
        max_entropy = max(all_entropy)
        min_entropy = min(all_entropy)
        entropy_range = max_entropy - min_entropy
    else:
        avg_entropy = max_entropy = min_entropy = entropy_range = 0

    # ── Trinucleotide transition matrix ──────────────────────────────
    # Build a 64x64 transition probability matrix from trinucleotide contexts
    all_trinucs = sorted(trinuc_contexts.keys())
    trinuc_total = sum(trinuc_contexts.values())

    # Transition: trinuc[i] → trinuc[j] where trinuc[i][1:] == trinuc[j][:2]
    transitions = {}
    for tri_i in all_trinucs:
        suffix = tri_i[1:]
        for tri_j in all_trinucs:
            prefix = tri_j[:2]
            if suffix == prefix:
                count_i = trinuc_contexts.get(tri_i, 0)
                count_j = trinuc_contexts.get(tri_j, 0)
                # Transition probability approximation
                if count_i > 0:
                    trans_prob = count_j / max(trinuc_total, 1)
                    transitions[(tri_i, tri_j)] = round(trans_prob, 6)

    # Compute entropy of transition matrix per source trinucleotide
    trinuc_trans_entropy = {}
    for tri_i in all_trinucs:
        probs = []
        for tri_j in all_trinucs:
            p = transitions.get((tri_i, tri_j), 0)
            if p > 0:
                probs.append(p)
        if probs:
            total_p = sum(probs)
            entropy = -sum((p / total_p) * math.log2(p / total_p) for p in probs)
            trinuc_trans_entropy[tri_i] = round(entropy, 4)

    # Overall trinucleotide entropy
    trinuc_entropy_vals = list(trinuc_trans_entropy.values())
    if trinuc_entropy_vals:
        avg_trinuc_entropy = sum(trinuc_entropy_vals) / len(trinuc_entropy_vals)
    else:
        avg_trinuc_entropy = 0

    # Context bias
    top_trinucs = sorted(trinuc_contexts.items(), key=lambda x: x[1], reverse=True)
    top_count = top_trinucs[0][1] if top_trinucs else 0
    bottom_count = top_trinucs[-1][1] if top_trinucs else 0
    context_ratio = top_count / max(bottom_count, 1)

    return {
        'position_entropy': position_entropy[:500],
        'smoothed_entropy': smoothed_entropy[:500],
        'avg_entropy': round(avg_entropy, 4),
        'max_entropy': round(max_entropy, 4),
        'min_entropy': round(min_entropy, 4),
        'entropy_range': round(entropy_range, 4),
        'trinuc_entropy': round(avg_trinuc_entropy, 4),
        'trinuc_trans_entropy': trinuc_trans_entropy,
        'context_bias_ratio': round(context_ratio, 4),
        'max_context': dict(top_trinucs[:10]) if top_trinucs else {},
        'n_transitions': len(transitions),
        'n_positions_analyzed': len(position_entropy)
    }


def lambda_handler(event, context):
    """Stage 2: Analyze DNA sequence data from Reader output."""


    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = event.get('Session', '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="DnaAnalyzerFunction",
        field_names=["gc_profile", "codon_usage", "cpg_islands", "repeats", "complexity_map"]
    )
    start_time = time.time()
    seq_id = event.get('kmers', {}).get('seq_id', 'unknown')
    print(f'[Analyzer] Starting on {seq_id}')

    # ── Field 1: gc_profile ← kmers
    t0 = time.time()
    gc_profile = compute_gc_profile(event.get('kmers', {}))
    field_1_time = int((time.time() - t0) * 1000)
    print(f'[Analyzer] gc_profile: bias={gc_profile["gc_bias"]}, chi2_4={gc_profile["chi2_4mer"]} in {field_1_time}ms')
    gc_profile_out = {**gc_profile, 'compute_ms': field_1_time, 'seq_id': seq_id}
    _streaming_publisher.publish('gc_profile', gc_profile_out)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()
    # ── Field 2: codon_usage ← base_composition
    t0 = time.time()
    codon_usage = analyze_codon_usage(event.get('base_composition', {}))
    field_2_time = int((time.time() - t0) * 1000)
    print(f'[Analyzer] codon_usage: CpG ratio={codon_usage["cpg_ratio"]}, model_mse={codon_usage["gc_model_mse"]} in {field_2_time}ms')
    codon_usage_out = {**codon_usage, 'compute_ms': field_2_time, 'seq_id': seq_id}
    _streaming_publisher.publish('codon_usage', codon_usage_out)

    # ── Field 3: cpg_islands ← open_reading_frames
    t0 = time.time()
    cpg_islands = detect_cpg_islands(event.get('open_reading_frames', {}))
    field_3_time = int((time.time() - t0) * 1000)
    print(f'[Analyzer] cpg_islands: {cpg_islands["n_clusters"]} clusters in {field_3_time}ms')
    cpg_islands_out = {**cpg_islands, 'compute_ms': field_3_time, 'seq_id': seq_id}
    _streaming_publisher.publish('cpg_islands', cpg_islands_out)

    # ── Field 4: repeats ← segments
    t0 = time.time()
    repeats = find_repeats(event.get('segments', {}))
    field_4_time = int((time.time() - t0) * 1000)
    print(f'[Analyzer] repeats: {repeats["n_segment_clusters"]} seg-clusters in {field_4_time}ms')
    repeats_out = {**repeats, 'compute_ms': field_4_time, 'seq_id': seq_id}
    _streaming_publisher.publish('repeats', repeats_out)

    # ── Field 5: complexity_map ← nucleotide_freq
    t0 = time.time()
    complexity_map = compute_complexity_map(event.get('nucleotide_freq', {}))
    field_5_time = int((time.time() - t0) * 1000)
    print(f'[Analyzer] complexity_map: avg_entropy={complexity_map["avg_entropy"]}, {complexity_map["n_transitions"]} transitions in {field_5_time}ms')
    complexity_map_out = {**complexity_map, 'compute_ms': field_5_time, 'seq_id': seq_id}
    _streaming_publisher.publish('complexity_map', complexity_map_out)

    total_time = int((time.time() - start_time) * 1000)
    print(f'[Analyzer] COMPLETE in {total_time}ms')

    return {
        'gc_profile': gc_profile_out,
        'codon_usage': codon_usage_out,
        'cpg_islands': cpg_islands_out,
        'repeats': repeats_out,
        'complexity_map': complexity_map_out
    }
