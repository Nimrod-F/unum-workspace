"""
Reporter - Stage 4 (Final) of DNA Sequence Analysis Pipeline

Produces final genome report: statistical summary, squiggle-style coordinates,
quality assessment, functional annotation, and comprehensive report.

Inspired by the Squiggle DNA visualization method (Benjamin Lee, 2019).

All computation is genuine — no artificial delays.
This is the final stage, so no streaming output is needed.

Each output field depends on exactly ONE input field from the Comparator:
  genome_summary    ← alignment_scores
  squiggle_coords   ← motifs
  quality_scores    ← palindromes
  annotation        ← repeat_analysis
  final_report      ← conservation
"""
import json
import time
import math
from collections import Counter


def build_genome_summary(alignment_data):
    """Build genome summary from alignment scores.
    Heavy: processes ALL 3-mer and 4-mer enrichment data, computes
    k-mer enrichment correlation matrix, and runs permutation test
    for overall compositional bias significance.
    """
    scores = alignment_data.get('scores', [])
    enrichment_3mer = alignment_data.get('enrichment_3mer', [])
    similar_pairs = alignment_data.get('similar_4mer_pairs', [])
    dissimilar_pairs = alignment_data.get('dissimilar_4mer_pairs', [])
    n_enriched = alignment_data.get('n_enriched', 0)
    n_depleted = alignment_data.get('n_depleted', 0)
    gc_bias = alignment_data.get('gc_bias', 0)
    complexity_index = alignment_data.get('complexity_index', 0)
    chi2_3 = alignment_data.get('chi2_3mer', 0)
    chi2_4 = alignment_data.get('chi2_4mer', 0)

    # K-mer landscape analysis
    log_odds_values = [s['log_odds'] for s in scores if s['log_odds'] != 0]
    if log_odds_values:
        avg_lo = sum(log_odds_values) / len(log_odds_values)
        max_lo = max(log_odds_values)
        min_lo = min(log_odds_values)
        lo_std = math.sqrt(sum((v - avg_lo) ** 2 for v in log_odds_values) / max(len(log_odds_values) - 1, 1))
    else:
        avg_lo = max_lo = min_lo = lo_std = 0

    total_kmers = len(scores)
    enrichment_profile = {
        'enriched_fraction': round(n_enriched / max(total_kmers, 1), 4),
        'depleted_fraction': round(n_depleted / max(total_kmers, 1), 4),
        'neutral_fraction': round((total_kmers - n_enriched - n_depleted) / max(total_kmers, 1), 4)
    }

    gc_enriched = sum(1 for s in scores if s['enriched'] and s['gc'] > 0.5)
    at_enriched = sum(1 for s in scores if s['enriched'] and s['gc'] <= 0.5)

    # ── 3-mer/4-mer enrichment correlation ──────────────────────────
    # Compare 3-mer and 4-mer enrichment patterns (are sub-k-mers consistent?)
    kmer3_lo = {e['kmer']: e['log_odds'] for e in enrichment_3mer}
    correlations = []
    for s4 in scores:
        kmer4 = s4['kmer']
        # Get constituent 3-mers
        sub3 = [kmer4[i:i + 3] for i in range(2)]
        sub_los = [kmer3_lo.get(s, 0) for s in sub3]
        avg_sub = sum(sub_los) / len(sub_los) if sub_los else 0
        correlations.append({
            'kmer4': kmer4, 'lo_4': s4['log_odds'],
            'avg_lo_3': round(avg_sub, 4),
            'consistent': (s4['log_odds'] > 0) == (avg_sub > 0) if avg_sub != 0 else True
        })

    consistency = sum(1 for c in correlations if c['consistent']) / max(len(correlations), 1)

    # ── Permutation test for compositional bias ────────────────────
    import random
    rng = random.Random(42)

    observed_chi2 = chi2_4
    n_perms = 300
    perm_chi2s = []
    for _ in range(n_perms):
        perm_counts = [max(1, rng.gauss(total_kmers / 256, total_kmers / 512)) for _ in range(256)]
        perm_total = sum(perm_counts)
        expected = perm_total / 256
        perm_chi2 = sum((c - expected) ** 2 / expected for c in perm_counts)
        perm_chi2s.append(perm_chi2)

    p_value = sum(1 for pc in perm_chi2s if pc >= observed_chi2) / n_perms

    return {
        'kmer_landscape': {
            'avg_log_odds': round(avg_lo, 4), 'max_log_odds': round(max_lo, 4),
            'min_log_odds': round(min_lo, 4), 'std_log_odds': round(lo_std, 4)
        },
        'enrichment_profile': enrichment_profile,
        'gc_enrichment_bias': round((gc_enriched - at_enriched) / max(gc_enriched + at_enriched, 1), 4),
        'compositional_bias': round(gc_bias, 4),
        'complexity_index': round(complexity_index, 4),
        'kmer_consistency': round(consistency, 4),
        'n_similar_4mer_pairs': len(similar_pairs),
        'n_dissimilar_4mer_pairs': len(dissimilar_pairs),
        'chi2_3mer': chi2_3,
        'chi2_4mer': chi2_4,
        'chi2_pvalue': round(p_value, 4),
        'bias_significant': p_value < 0.05,
        'total_kmers_analyzed': total_kmers
    }


def generate_squiggle_coords(motif_data):
    """Generate Squiggle-style visualization coordinates.
    Heavy: generates extended Squiggle walk from dinucleotide profiles,
    computes walk statistics including fractal dimension estimate,
    and performs spectral analysis of the walk trajectory.
    """
    import random
    rng = random.Random(42)

    motifs = motif_data.get('motif_predictions', [])
    step_params = motif_data.get('dinuc_step_params', {})
    gc_mean = motif_data.get('gc_mean', 0.5)
    gc_variation = motif_data.get('gc_variation', 0)
    gc_model = motif_data.get('gc_model', {})
    cum_skew = motif_data.get('cumulative_skew', [])

    base_vectors = {'A': (1, 1), 'T': (1, -1), 'G': (1, 0.5), 'C': (1, -0.5)}

    # Generate extended Squiggle walk
    x, y = 0.0, 0.0
    coords = [{'x': 0, 'y': 0}]
    dinucs = sorted(step_params.keys())

    # More repetitions for longer walk
    for rep in range(30):
        for dinuc in dinucs:
            if len(dinuc) == 2:
                dx1, dy1 = base_vectors.get(dinuc[0], (1, 0))
                dx2, dy2 = base_vectors.get(dinuc[1], (1, 0))
                ratio = step_params.get(dinuc, {}).get('ratio', 1.0)
                mod = 1.0 + 0.1 * math.sin(rep * 0.5)
                x += (dx1 + dx2) * ratio * mod * 0.5
                y += (dy1 + dy2) * ratio * mod * 0.5
                coords.append({'x': round(x, 2), 'y': round(y, 2)})

    # Motif markers
    motif_markers = [
        {'motif': m['motif'], 'name': m['name'], 'gc_content': m['gc_content'],
         'score': m['dinuc_score'], 'significant': m.get('significant', False)}
        for m in motifs if m['likely_present']
    ]

    # Walk statistics
    if len(coords) > 1:
        final_x = coords[-1]['x']
        final_y = coords[-1]['y']
        displacement = math.sqrt(final_x ** 2 + final_y ** 2)
        path_length = sum(
            math.sqrt((coords[i]['x'] - coords[i - 1]['x']) ** 2 +
                       (coords[i]['y'] - coords[i - 1]['y']) ** 2)
            for i in range(1, len(coords))
        )
        tortuosity = path_length / max(displacement, 0.01)

        # ── Box-counting fractal dimension estimate ─────────────────
        x_vals = [c['x'] for c in coords]
        y_vals = [c['y'] for c in coords]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        span = max(x_max - x_min, y_max - y_min, 1)

        box_counts = []
        for size_k in range(2, 8):
            box_size = span / (2 ** size_k)
            if box_size <= 0:
                continue
            boxes = set()
            for c in coords:
                bx = int((c['x'] - x_min) / box_size)
                by = int((c['y'] - y_min) / box_size)
                boxes.add((bx, by))
            box_counts.append({'scale': 2 ** size_k, 'n_boxes': len(boxes), 'box_size': round(box_size, 2)})

        # Estimate fractal dimension from log-log slope
        if len(box_counts) >= 2:
            log_sizes = [math.log(bc['scale']) for bc in box_counts]
            log_counts = [math.log(bc['n_boxes']) for bc in box_counts if bc['n_boxes'] > 0]
            if len(log_sizes) == len(log_counts) and len(log_sizes) >= 2:
                n = len(log_sizes)
                sum_x = sum(log_sizes)
                sum_y = sum(log_counts)
                sum_xy = sum(a * b for a, b in zip(log_sizes, log_counts))
                sum_x2 = sum(a * a for a in log_sizes)
                denom = n * sum_x2 - sum_x ** 2
                fractal_dim = (n * sum_xy - sum_x * sum_y) / max(denom, 0.001)
            else:
                fractal_dim = 1.0
        else:
            fractal_dim = 1.0

        # ── Spectral analysis of walk trajectory ────────────────────
        # Compute autocorrelation of y-coordinates (displacement signal)
        y_series = [c['y'] for c in coords]
        n_pts = len(y_series)
        y_mean = sum(y_series) / n_pts
        y_centered = [v - y_mean for v in y_series]
        var_y = sum(v * v for v in y_centered) / n_pts or 1

        max_lag = min(100, n_pts // 2)
        autocorr = []
        for lag in range(1, max_lag + 1):
            corr = sum(y_centered[i] * y_centered[i + lag]
                       for i in range(n_pts - lag)) / ((n_pts - lag) * var_y)
            autocorr.append({'lag': lag, 'correlation': round(corr, 4)})

        # Find dominant period from autocorrelation peaks
        peaks = []
        for i in range(1, len(autocorr) - 1):
            if (autocorr[i]['correlation'] > autocorr[i - 1]['correlation'] and
                    autocorr[i]['correlation'] > autocorr[i + 1]['correlation'] and
                    autocorr[i]['correlation'] > 0.1):
                peaks.append(autocorr[i])

        # ── Bootstrap confidence interval for fractal dimension ──────
        n_boot = 200
        boot_dims = []
        for _ in range(n_boot):
            # Resample coordinates with replacement
            sample_idx = [rng.randint(0, len(coords) - 1) for _ in range(len(coords))]
            sample_idx.sort()
            sample = [coords[i] for i in sample_idx]
            sx_vals = [c['x'] for c in sample]
            sy_vals = [c['y'] for c in sample]
            sx_min, sx_max = min(sx_vals), max(sx_vals)
            sy_min, sy_max = min(sy_vals), max(sy_vals)
            s_span = max(sx_max - sx_min, sy_max - sy_min, 1)

            b_counts = []
            for size_k in [3, 5, 7]:
                bsz = s_span / (2 ** size_k)
                if bsz <= 0:
                    continue
                bxs = set()
                for c in sample:
                    bxs.add((int((c['x'] - sx_min) / bsz), int((c['y'] - sy_min) / bsz)))
                b_counts.append((math.log(2 ** size_k), math.log(len(bxs)) if bxs else 0))

            if len(b_counts) >= 2:
                bn = len(b_counts)
                bsx = sum(a for a, _ in b_counts)
                bsy = sum(b for _, b in b_counts)
                bsxy = sum(a * b for a, b in b_counts)
                bsx2 = sum(a * a for a, _ in b_counts)
                bd = bn * bsx2 - bsx ** 2
                if bd != 0:
                    boot_dims.append((bn * bsxy - bsx * bsy) / bd)

        boot_dims.sort()
        if boot_dims:
            fd_ci = {
                'lower': round(boot_dims[int(len(boot_dims) * 0.025)], 4),
                'upper': round(boot_dims[int(len(boot_dims) * 0.975)], 4),
                'mean': round(sum(boot_dims) / len(boot_dims), 4)
            }
        else:
            fd_ci = {'lower': fractal_dim, 'upper': fractal_dim, 'mean': fractal_dim}

    else:
        displacement = path_length = tortuosity = 0
        fractal_dim = 1.0
        box_counts = []
        autocorr = []
        peaks = []
        fd_ci = {}

    return {
        'coordinates': coords[:500],
        'n_points': len(coords),
        'motif_markers': motif_markers,
        'walk_stats': {
            'displacement': round(displacement, 2),
            'path_length': round(path_length, 2),
            'tortuosity': round(tortuosity, 4),
            'fractal_dimension': round(fractal_dim, 4),
            'fractal_dim_ci': fd_ci,
            'gc_mean': round(gc_mean, 4),
            'gc_variation': round(gc_variation, 4)
        },
        'spectral': {
            'n_autocorr_lags': len(autocorr),
            'dominant_periods': peaks[:5],
        },
        'gc_model': gc_model,
        'box_counts': box_counts
    }


def assess_quality(palindrome_data):
    """Assess sequence quality from structural features.
    Heavy: computes comprehensive ORF quality metrics, runs bootstrap
    on codon frequencies, and assesses reading frame balance.
    """
    symmetry_scores = palindrome_data.get('symmetry_scores', [])
    frame_balance = palindrome_data.get('frame_balance', 0)
    coding_density = palindrome_data.get('coding_density', 0)
    total_orfs = palindrome_data.get('total_orfs', 0)
    frame_entropy = palindrome_data.get('frame_entropy', 0)
    frame_dist = palindrome_data.get('frame_distribution', {})
    codon_freq = palindrome_data.get('codon_freq', {})
    codon_chi2 = palindrome_data.get('codon_chi2', 0)
    overlaps = palindrome_data.get('cluster_overlaps', [])
    prot_hist = palindrome_data.get('protein_length_hist', {})

    quality_metrics = {
        'coding_potential': round(min(coding_density * 5, 1.0), 4),
        'frame_balance': round(frame_balance, 4),
        'orf_density': round(total_orfs / 100, 4),
        'structural_symmetry': round(
            sum(s['symmetry_score'] for s in symmetry_scores) / max(len(symmetry_scores), 1), 4
        )
    }
    quality_score = sum(quality_metrics.values()) / len(quality_metrics)

    grade = 'A' if quality_score > 0.7 else 'B' if quality_score > 0.5 else 'C' if quality_score > 0.3 else 'D'

    # ── Bootstrap codon usage test ────────────────────────────────────
    import random
    rng = random.Random(42)

    codon_list = []
    for aa, count in codon_freq.items():
        codon_list.extend([aa] * count)

    n_bootstrap = 100
    chi2_samples = []
    n_types = len(codon_freq)
    sample_size = min(len(codon_list), 500)  # Cap sample size for performance
    if codon_list and n_types > 0:
        for _ in range(n_bootstrap):
            sample = [codon_list[rng.randint(0, len(codon_list) - 1)] for _ in range(sample_size)]
            sample_counts = Counter(sample)
            sample_total = sum(sample_counts.values())
            expected = sample_total / max(n_types, 1)
            chi2_s = sum((c - expected) ** 2 / max(expected, 1) for c in sample_counts.values())
            chi2_samples.append(chi2_s)

    if chi2_samples:
        chi2_samples.sort()
        p_value = sum(1 for c in chi2_samples if c >= codon_chi2) / len(chi2_samples)
        chi2_ci = {
            'mean': round(sum(chi2_samples) / len(chi2_samples), 2),
            'p5': round(chi2_samples[int(len(chi2_samples) * 0.05)], 2),
            'p95': round(chi2_samples[int(len(chi2_samples) * 0.95)], 2),
        }
    else:
        p_value = 1.0
        chi2_ci = {}

    return {
        'quality_metrics': quality_metrics,
        'overall_quality': round(quality_score, 4),
        'grade': grade,
        'frame_entropy': round(frame_entropy, 4),
        'codon_chi2': round(codon_chi2, 2),
        'codon_chi2_pvalue': round(p_value, 4),
        'codon_chi2_ci': chi2_ci,
        'n_orf_clusters': len(symmetry_scores),
        'n_cluster_overlaps': len(overlaps),
        'total_orfs': total_orfs,
        'frame_distribution': frame_dist,
        'protein_length_hist': prot_hist
    }


def annotate_genome(repeat_data):
    """Functional annotation from repeat analysis.
    Heavy: performs domain boundary detection and characterization,
    runs permutation test for domain transition significance,
    and computes domain homogeneity scores.
    """
    import random
    rng = random.Random(42)

    organization = repeat_data.get('organization', 'unknown')
    gc_heterogeneity = repeat_data.get('gc_heterogeneity', 0)
    n_isochores = repeat_data.get('n_isochores', 0)
    repeat_score = repeat_data.get('repeat_score', 0)
    domain_types = repeat_data.get('domain_types', {})
    avg_complexity = repeat_data.get('avg_complexity', 0)
    optimal_clusters = repeat_data.get('optimal_clusters', 0)
    iso_gc_mean = repeat_data.get('isochore_gc_mean', 0)
    iso_gc_range = repeat_data.get('isochore_gc_range', 0)
    gaps = repeat_data.get('dendrogram_gaps', [])
    domain_transitions = repeat_data.get('domain_transitions', {})

    # Genome classification
    if organization == 'heterogeneous':
        classification = 'complex_eukaryotic_like'
    elif organization == 'moderately_heterogeneous':
        classification = 'moderate_complexity'
    else:
        classification = 'prokaryotic_like'

    repeat_annotation = {
        'repeat_density': round(repeat_score, 2),
        'category': 'high' if repeat_score > 20 else 'moderate' if repeat_score > 10 else 'low',
        'low_complexity_warning': repeat_score > 30
    }

    total_domains = sum(domain_types.values())
    domain_profile = {k: round(v / max(total_domains, 1), 4) for k, v in domain_types.items()}

    # ── Domain boundary characterization ─────────────────────────────
    boundary_analysis = {
        'n_boundaries': max(0, optimal_clusters - 1),
        'sharpest_boundary': gaps[0] if gaps else {},
        'boundary_hierarchy': gaps[:5],
        'avg_gap_size': round(sum(g['gap'] for g in gaps) / max(len(gaps), 1), 4) if gaps else 0
    }

    # Isochore classification
    if iso_gc_mean > 0.55:
        isochore_class = 'H3'
    elif iso_gc_mean > 0.47:
        isochore_class = 'H2'
    elif iso_gc_mean > 0.41:
        isochore_class = 'H1'
    elif iso_gc_mean > 0.37:
        isochore_class = 'L2'
    else:
        isochore_class = 'L1'

    # ── Permutation test for domain transitions ──────────────────────
    # Are the observed transitions non-random?
    trans_values = list(domain_transitions.values()) if domain_transitions else []
    n_perms = 500
    perm_entropy_values = []
    if trans_values:
        obs_entropy = -sum(v * math.log2(v) for v in trans_values if v > 0)
        n_trans = len(trans_values)
        for _ in range(n_perms):
            # Generate random transition probabilities via Dirichlet-like
            raw = [rng.expovariate(1.0) for _ in range(n_trans)]
            total_raw = sum(raw)
            perm_probs = [r / total_raw for r in raw]
            perm_ent = -sum(p * math.log2(p) for p in perm_probs if p > 0)
            perm_entropy_values.append(perm_ent)

        perm_entropy_values.sort()
        trans_pvalue = sum(1 for pe in perm_entropy_values if pe <= obs_entropy) / n_perms
    else:
        obs_entropy = 0
        trans_pvalue = 1.0

    # ── Domain homogeneity scoring ───────────────────────────────────
    homogeneity_scores = {}
    for dtype, count in domain_types.items():
        frac = count / max(total_domains, 1)
        expected = 1.0 / max(len(domain_types), 1)
        deviation = abs(frac - expected)
        homogeneity_scores[dtype] = round(1.0 - deviation, 4)

    overall_homogeneity = sum(homogeneity_scores.values()) / max(len(homogeneity_scores), 1)

    # ── Bootstrap confidence interval for GC heterogeneity ──────────
    n_boot = 500
    het_samples = []
    for _ in range(n_boot):
        sim_gc = [rng.gauss(iso_gc_mean if iso_gc_mean > 0 else 0.5, max(iso_gc_range, 0.01))
                  for _ in range(max(n_isochores, 5))]
        sim_mean = sum(sim_gc) / len(sim_gc)
        sim_range = max(sim_gc) - min(sim_gc) if len(sim_gc) > 1 else 0
        het_samples.append(sim_range)

    het_samples.sort()
    het_ci = {
        'lower': round(het_samples[int(len(het_samples) * 0.025)], 4) if het_samples else 0,
        'upper': round(het_samples[int(len(het_samples) * 0.975)], 4) if het_samples else 0,
        'mean': round(sum(het_samples) / len(het_samples), 4) if het_samples else 0
    }

    return {
        'classification': classification,
        'organization': organization,
        'repeat_annotation': repeat_annotation,
        'domain_profile': domain_profile,
        'domain_transitions': domain_transitions,
        'transition_entropy': round(obs_entropy, 4),
        'transition_pvalue': round(trans_pvalue, 4),
        'boundary_analysis': boundary_analysis,
        'n_isochores': n_isochores,
        'isochore_class': isochore_class,
        'isochore_gc_mean': round(iso_gc_mean, 4),
        'isochore_gc_range': round(iso_gc_range, 4),
        'gc_heterogeneity': round(gc_heterogeneity, 4),
        'gc_het_ci': het_ci,
        'avg_complexity': round(avg_complexity, 4),
        'homogeneity_scores': homogeneity_scores,
        'overall_homogeneity': round(overall_homogeneity, 4)
    }


def compile_final_report(conservation_data):
    """Compile final comprehensive genome report.
    Heavy: performs Monte Carlo simulation to estimate conservation
    score confidence interval and evolutionary rate inference.
    """
    conservation_score = conservation_data.get('conservation_score', 0)
    n_conserved = conservation_data.get('n_conserved_positions', 0)
    n_variable = conservation_data.get('n_variable_positions', 0)
    context_dep = conservation_data.get('context_dependency', 0)
    genome_type = conservation_data.get('genome_type', 'unknown')
    dominant = conservation_data.get('dominant_context', {})
    most_dep = conservation_data.get('most_context_dependent', [])
    least_dep = conservation_data.get('least_context_dependent', [])
    regions = conservation_data.get('conservation_regions', [])
    trinuc_entropy = conservation_data.get('trinuc_transition_entropy', 0)
    n_transitions = conservation_data.get('n_transitions', 0)

    assessment = {
        'genome_type': genome_type,
        'conservation_score': round(conservation_score, 4),
        'variability_score': round(1 - conservation_score, 4),
        'context_sensitivity': round(context_dep, 4),
    }

    if conservation_score > 0.3:
        evo_status = 'under_selection'
        evo_explanation = 'High conservation suggests selective pressure'
    elif conservation_score > 0.1:
        evo_status = 'neutral_drift'
        evo_explanation = 'Moderate conservation consistent with neutral evolution'
    else:
        evo_status = 'rapid_evolution'
        evo_explanation = 'Low conservation suggests rapid evolutionary change'

    # ── Monte Carlo conservation confidence interval ────────────────
    import random
    rng = random.Random(42)

    n_total = n_conserved + n_variable
    if n_total == 0:
        n_total = max(1, int(conservation_score * 100))
        n_conserved = int(conservation_score * n_total)

    n_sim = 500
    cons_samples = []
    for _ in range(n_sim):
        # Binomial resample
        hits = sum(1 for _ in range(n_total) if rng.random() < conservation_score)
        cons_samples.append(hits / max(n_total, 1))

    cons_samples.sort()
    n = len(cons_samples)
    cons_ci = {
        'mean': round(sum(cons_samples) / n, 4),
        'lower': round(cons_samples[int(n * 0.025)], 4),
        'upper': round(cons_samples[int(n * 0.975)], 4),
        'std': round(math.sqrt(sum((c - sum(cons_samples) / n) ** 2 for c in cons_samples) / n), 4)
    }

    # ── Evolutionary rate estimation ─────────────────────────────────
    # Use Jukes-Cantor model: d = -3/4 * ln(1 - 4/3 * p)
    # where p = proportion of variable sites
    p_variable = 1 - conservation_score
    if p_variable < 0.75:
        jc_distance = -0.75 * math.log(1 - (4 / 3) * p_variable)
    else:
        jc_distance = float('inf')

    # Substitution rate per site (assuming 1 million years divergence)
    sub_rate = jc_distance / 1e6 if jc_distance != float('inf') else 0

    # Conservation region summary
    if regions:
        avg_region_len = sum(r['length'] for r in regions) / len(regions)
        max_region_len = max(r['length'] for r in regions)
        total_conserved_span = sum(r['length'] for r in regions)
    else:
        avg_region_len = max_region_len = total_conserved_span = 0

    return {
        'assessment': assessment,
        'evolutionary_status': evo_status,
        'evolutionary_explanation': evo_explanation,
        'jukes_cantor_distance': round(jc_distance, 6) if jc_distance != float('inf') else 'saturated',
        'estimated_sub_rate': round(sub_rate, 10),
        'conservation_ci': cons_ci,
        'n_simulations': n_sim,
        'conservation_regions': {
            'count': len(regions),
            'avg_length': round(avg_region_len, 1),
            'max_length': max_region_len,
            'total_span': total_conserved_span
        },
        'context_dependency_summary': {
            'score': round(context_dep, 4),
            'most_dependent': most_dep[:5],
            'least_dependent': least_dep[:5],
            'trinuc_transition_entropy': round(trinuc_entropy, 4),
            'n_transitions': n_transitions
        },
        'dominant_contexts': dominant
    }


def lambda_handler(event, context):
    """Stage 4 (final): Generate comprehensive genome report."""
    start_time = time.time()
    seq_id = event.get('alignment_scores', {}).get('seq_id', 'unknown')
    print(f'[Reporter] Starting on {seq_id}')

    # ── Field 1: genome_summary ← alignment_scores
    t0 = time.time()
    summary = build_genome_summary(event.get('alignment_scores', {}))
    field_1_time = int((time.time() - t0) * 1000)
    print(f'[Reporter] genome_summary: consistency={summary["kmer_consistency"]}, p={summary["chi2_pvalue"]} in {field_1_time}ms')
    genome_summary = {**summary, 'compute_ms': field_1_time, 'seq_id': seq_id}

    # ── Field 2: squiggle_coords ← motifs
    t0 = time.time()
    squiggle = generate_squiggle_coords(event.get('motifs', {}))
    field_2_time = int((time.time() - t0) * 1000)
    print(f'[Reporter] squiggle: {squiggle["n_points"]} pts, fractal_dim={squiggle["walk_stats"]["fractal_dimension"]} in {field_2_time}ms')
    squiggle_coords = {**squiggle, 'compute_ms': field_2_time, 'seq_id': seq_id}

    # ── Field 3: quality_scores ← palindromes
    t0 = time.time()
    quality = assess_quality(event.get('palindromes', {}))
    field_3_time = int((time.time() - t0) * 1000)
    print(f'[Reporter] quality: grade={quality["grade"]}, codon_p={quality["codon_chi2_pvalue"]} in {field_3_time}ms')
    quality_scores = {**quality, 'compute_ms': field_3_time, 'seq_id': seq_id}

    # ── Field 4: annotation ← repeat_analysis
    t0 = time.time()
    annotation = annotate_genome(event.get('repeat_analysis', {}))
    field_4_time = int((time.time() - t0) * 1000)
    print(f'[Reporter] annotation: class={annotation["classification"]}, iso={annotation["isochore_class"]} in {field_4_time}ms')
    annotation_out = {**annotation, 'compute_ms': field_4_time, 'seq_id': seq_id}

    # ── Field 5: final_report ← conservation
    t0 = time.time()
    report = compile_final_report(event.get('conservation', {}))
    field_5_time = int((time.time() - t0) * 1000)
    print(f'[Reporter] final_report: status={report["evolutionary_status"]}, JC={report["jukes_cantor_distance"]} in {field_5_time}ms')
    final_report = {**report, 'compute_ms': field_5_time, 'seq_id': seq_id}

    total_time = int((time.time() - start_time) * 1000)
    print(f'[Reporter] COMPLETE in {total_time}ms')

    return {
        'genome_summary': genome_summary,
        'squiggle_coords': squiggle_coords,
        'quality_scores': quality_scores,
        'annotation': annotation_out,
        'final_report': final_report
    }
