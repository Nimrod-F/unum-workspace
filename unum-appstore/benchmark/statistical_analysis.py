#!/usr/bin/env python3
"""
Statistical Analysis Module for Unum Evaluation Benchmarks
===========================================================
Computes academic-quality statistics from benchmark results:

  - Descriptive: mean, median, IQR, P95, P99, CV
  - Inferential: Mann-Whitney U test, Cohen's d effect size
  - Confidence: Bootstrap 95% CI (BCa)
  - Improvement: Percentage improvement over baselines
  - Stability: CV-based warm stability assessment

Usage:
    python statistical_analysis.py results/run_20250610_143000
    python statistical_analysis.py results/run_20250610_143000 --format latex
"""

import json
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("statistical_analysis")


# ─── Statistical Functions ──────────────────────────────────────────────────

def iqr(data: np.ndarray) -> float:
    """Interquartile range."""
    return float(np.percentile(data, 75) - np.percentile(data, 25))


def cv(data: np.ndarray) -> float:
    """Coefficient of variation (%)."""
    m = np.mean(data)
    if m == 0:
        return 0.0
    return float(np.std(data, ddof=1) / m * 100)


def percentile(data: np.ndarray, p: float) -> float:
    """P-th percentile."""
    return float(np.percentile(data, p))


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.median,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval using BCa method.
    Returns (lower, upper) bounds.
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    
    # Generate bootstrap samples
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic(sample)
    
    # BCa correction
    z0 = _norm_ppf(np.mean(boot_stats < statistic(data)))
    
    # Jackknife for acceleration
    jack_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(data, i)
        jack_stats[i] = statistic(jack_sample)
    
    jack_mean = np.mean(jack_stats)
    diffs = jack_mean - jack_stats
    a = np.sum(diffs ** 3) / (6 * (np.sum(diffs ** 2)) ** 1.5) if np.sum(diffs ** 2) > 0 else 0
    
    alpha = 1 - confidence
    z_alpha = _norm_ppf(alpha / 2)
    z_1_alpha = _norm_ppf(1 - alpha / 2)
    
    # Adjusted percentiles
    a1 = _norm_cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    a2 = _norm_cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))
    
    lower = np.percentile(boot_stats, 100 * a1)
    upper = np.percentile(boot_stats, 100 * a2)
    
    return float(lower), float(upper)


def _norm_ppf(p: float) -> float:
    """Normal distribution percent point function (inverse CDF)."""
    # Using a rational approximation (Abramowitz and Stegun)
    p = max(1e-10, min(1 - 1e-10, p))
    if p < 0.5:
        return -_rational_approx(math.sqrt(-2.0 * math.log(p)))
    else:
        return _rational_approx(math.sqrt(-2.0 * math.log(1 - p)))


def _rational_approx(t: float) -> float:
    """Helper for _norm_ppf."""
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0] + c[1] * t + c[2] * t * t) / (1 + d[0] * t + d[1] * t * t + d[2] * t * t * t)


def _norm_cdf(x: float) -> float:
    """Normal distribution CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Mann-Whitney U test (non-parametric).
    Returns (U statistic, p-value).
    Uses normal approximation for large samples.
    """
    nx, ny = len(x), len(y)
    
    # Rank all values
    combined = np.concatenate([x, y])
    ranks = _rank_data(combined)
    
    # Sum ranks for first group
    R1 = np.sum(ranks[:nx])
    
    # U statistic
    U1 = R1 - nx * (nx + 1) / 2
    U2 = nx * ny - U1
    U = min(U1, U2)
    
    # Normal approximation for p-value
    mu = nx * ny / 2
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12)
    
    if sigma == 0:
        return float(U), 1.0
    
    z = (U - mu) / sigma
    p_value = 2 * _norm_cdf(-abs(z))
    
    return float(U), float(p_value)


def _rank_data(data: np.ndarray) -> np.ndarray:
    """Assign ranks to data (handles ties with average rank)."""
    n = len(data)
    sorter = np.argsort(data)
    ranks = np.empty_like(sorter, dtype=float)
    ranks[sorter] = np.arange(1, n + 1, dtype=float)
    
    # Handle ties
    sorted_data = data[sorter]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_data[j] == sorted_data[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[sorter[k]] = avg_rank
        i = j
    
    return ranks


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d effect size (pooled standard deviation).
    Interpretation: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large.
    """
    nx, ny = len(x), len(y)
    pooled_std = math.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def effect_size_label(d: float) -> str:
    """Human-readable effect size label."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# ─── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class DescriptiveStats:
    """Descriptive statistics for a single metric."""
    metric: str
    n: int = 0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    iqr: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    p5: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    cv_pct: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "metric": self.metric, "n": self.n,
            "mean": round(self.mean, 4), "median": round(self.median, 4),
            "std": round(self.std, 4), "iqr": round(self.iqr, 4),
            "min": round(self.min_val, 4), "max": round(self.max_val, 4),
            "p5": round(self.p5, 4), "p25": round(self.p25, 4),
            "p75": round(self.p75, 4), "p95": round(self.p95, 4),
            "p99": round(self.p99, 4),
            "cv_pct": round(self.cv_pct, 2),
            "ci_95": [round(self.ci_lower, 4), round(self.ci_upper, 4)],
        }


@dataclass
class ComparisonResult:
    """Statistical comparison between two configurations."""
    metric: str
    config_a: str
    config_b: str
    improvement_pct: float = 0.0
    mann_whitney_u: float = 0.0
    p_value: float = 0.0
    significant: bool = False  # p < 0.05
    cohens_d: float = 0.0
    effect_size: str = ""
    
    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "config_a": self.config_a,
            "config_b": self.config_b,
            "improvement_pct": round(self.improvement_pct, 2),
            "mann_whitney_u": round(self.mann_whitney_u, 2),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
            "cohens_d": round(self.cohens_d, 3),
            "effect_size": self.effect_size,
        }


# ─── Main Analysis ─────────────────────────────────────────────────────────

METRICS = [
    "e2e_latency_ms",
    "aggregator_delay_ms",
    "total_billed_duration_ms",
    "peak_memory_mb",
    "cold_start_count",
    "total_dynamo_reads",
    "total_dynamo_writes",
    "estimated_cost_usd",
]


class StatisticalAnalyzer:
    """Computes comprehensive statistics from benchmark results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.raw_results = self._load_results()
        self.descriptive: Dict[str, Dict[str, Dict[str, DescriptiveStats]]] = {}
        self.comparisons: List[ComparisonResult] = []
    
    def _load_results(self) -> Dict[str, Dict[str, List[dict]]]:
        """Load results from JSON files."""
        all_results_path = self.results_dir / "all_results.json"
        if all_results_path.exists():
            with open(all_results_path) as f:
                return json.load(f)
        
        # Try checkpoint files
        results = {}
        for f in self.results_dir.glob("*__*.json"):
            parts = f.stem.split("__")
            if len(parts) == 2:
                wf_name, cfg_label = parts
                if wf_name not in results:
                    results[wf_name] = {}
                with open(f) as fh:
                    results[wf_name][cfg_label] = json.load(fh)
        
        return results
    
    def analyze_all(self) -> dict:
        """Run full statistical analysis."""
        logger.info("Running statistical analysis...")
        
        # Phase 1: Descriptive statistics
        for wf_name, configs in self.raw_results.items():
            self.descriptive[wf_name] = {}
            for cfg_label, runs in configs.items():
                self.descriptive[wf_name][cfg_label] = {}
                
                # Filter successful runs
                successful = [r for r in runs if r.get("success", False)]
                if not successful:
                    continue
                
                for metric in METRICS:
                    values = np.array([r.get(metric, 0) for r in successful], dtype=float)
                    if len(values) == 0:
                        continue
                    
                    stats = self._compute_descriptive(metric, values)
                    self.descriptive[wf_name][cfg_label][metric] = stats
        
        # Phase 2: Pairwise comparisons (each config vs Unum-Base)
        self.comparisons = []
        for wf_name, configs in self.raw_results.items():
            baseline_runs = configs.get("Unum-Base", [])
            baseline_successful = [r for r in baseline_runs if r.get("success", False)]
            
            if not baseline_successful:
                continue
            
            for cfg_label, runs in configs.items():
                if cfg_label == "Unum-Base":
                    continue
                
                successful = [r for r in runs if r.get("success", False)]
                if not successful:
                    continue
                
                for metric in METRICS:
                    baseline_vals = np.array([r.get(metric, 0) for r in baseline_successful], dtype=float)
                    config_vals = np.array([r.get(metric, 0) for r in successful], dtype=float)
                    
                    comparison = self._compute_comparison(
                        metric, "Unum-Base", cfg_label, baseline_vals, config_vals
                    )
                    comparison_dict = comparison.to_dict()
                    comparison_dict["workflow"] = wf_name
                    self.comparisons.append(comparison)
        
        # Build output report
        report = self._build_report()
        
        # Save
        output_path = self.results_dir / "statistical_analysis.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis saved to {output_path}")
        return report
    
    def _compute_descriptive(self, metric: str, values: np.ndarray) -> DescriptiveStats:
        """Compute descriptive statistics for a metric."""
        stats = DescriptiveStats(metric=metric, n=len(values))
        
        stats.mean = float(np.mean(values))
        stats.median = float(np.median(values))
        stats.std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        stats.iqr = iqr(values)
        stats.min_val = float(np.min(values))
        stats.max_val = float(np.max(values))
        stats.p5 = percentile(values, 5)
        stats.p25 = percentile(values, 25)
        stats.p75 = percentile(values, 75)
        stats.p95 = percentile(values, 95)
        stats.p99 = percentile(values, 99)
        stats.cv_pct = cv(values)
        
        if len(values) >= 5:
            stats.ci_lower, stats.ci_upper = bootstrap_ci(values)
        else:
            stats.ci_lower = stats.ci_upper = stats.median
        
        return stats
    
    def _compute_comparison(
        self, metric: str, label_a: str, label_b: str,
        vals_a: np.ndarray, vals_b: np.ndarray
    ) -> ComparisonResult:
        """Compute pairwise comparison."""
        comp = ComparisonResult(metric=metric, config_a=label_a, config_b=label_b)
        
        # Improvement percentage (positive = config_b is better/lower)
        median_a = float(np.median(vals_a))
        median_b = float(np.median(vals_b))
        if median_a > 0:
            comp.improvement_pct = (median_a - median_b) / median_a * 100
        
        # Mann-Whitney U test
        if len(vals_a) >= 3 and len(vals_b) >= 3:
            comp.mann_whitney_u, comp.p_value = mann_whitney_u(vals_a, vals_b)
            comp.significant = comp.p_value < 0.05
        
        # Cohen's d
        if len(vals_a) >= 2 and len(vals_b) >= 2:
            comp.cohens_d = cohens_d(vals_a, vals_b)
            comp.effect_size = effect_size_label(comp.cohens_d)
        
        return comp
    
    def _build_report(self) -> dict:
        """Build the full analysis report."""
        report = {
            "metadata": {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat() if hasattr(self, '_') else None,
                "results_dir": str(self.results_dir),
                "metrics_analyzed": METRICS,
            },
            "descriptive_statistics": {},
            "pairwise_comparisons": [],
            "summary_tables": {},
        }
        
        # Descriptive stats
        for wf_name, configs in self.descriptive.items():
            report["descriptive_statistics"][wf_name] = {}
            for cfg_label, metrics in configs.items():
                report["descriptive_statistics"][wf_name][cfg_label] = {
                    m: s.to_dict() for m, s in metrics.items()
                }
        
        # Comparisons
        report["pairwise_comparisons"] = [c.to_dict() for c in self.comparisons]
        
        # Summary tables (for easy consumption by chart/table generators)
        report["summary_tables"] = self._build_summary_tables()
        
        return report
    
    def _build_summary_tables(self) -> dict:
        """Build summary tables for the paper."""
        tables = {}
        
        # Table 1: Main Results (median ± IQR for E2E latency, cost, billed duration)
        main_results = {}
        for wf_name, configs in self.descriptive.items():
            main_results[wf_name] = {}
            for cfg_label, metrics in configs.items():
                e2e = metrics.get("e2e_latency_ms")
                cost = metrics.get("estimated_cost_usd")
                billed = metrics.get("total_billed_duration_ms")
                
                main_results[wf_name][cfg_label] = {
                    "e2e_median": e2e.median if e2e else None,
                    "e2e_iqr": e2e.iqr if e2e else None,
                    "e2e_p95": e2e.p95 if e2e else None,
                    "e2e_ci_95": [e2e.ci_lower, e2e.ci_upper] if e2e else None,
                    "cost_mean": cost.mean if cost else None,
                    "billed_median": billed.median if billed else None,
                }
        tables["main_results"] = main_results
        
        # Table 2: Improvement percentages vs baseline
        improvements = {}
        for comp in self.comparisons:
            key = f"{comp.metric}"
            if key not in improvements:
                improvements[key] = {}
            improvements[key][f"{comp.config_b}"] = {
                "improvement_pct": comp.improvement_pct,
                "significant": comp.significant,
                "effect_size": comp.effect_size,
                "p_value": comp.p_value,
            }
        tables["improvements_vs_baseline"] = improvements
        
        # Table 3: DynamoDB I/O comparison
        dynamo_table = {}
        for wf_name, configs in self.descriptive.items():
            dynamo_table[wf_name] = {}
            for cfg_label, metrics in configs.items():
                reads = metrics.get("total_dynamo_reads")
                writes = metrics.get("total_dynamo_writes")
                dynamo_table[wf_name][cfg_label] = {
                    "reads_median": reads.median if reads else None,
                    "writes_median": writes.median if writes else None,
                    "total_io_median": (reads.median + writes.median) if reads and writes else None,
                }
        tables["dynamodb_io"] = dynamo_table
        
        return tables
    
    def print_summary(self):
        """Print a human-readable summary to console."""
        print("\n" + "=" * 80)
        print("  STATISTICAL ANALYSIS SUMMARY")
        print("=" * 80)
        
        for wf_name, configs in self.descriptive.items():
            print(f"\n  Workflow: {wf_name}")
            print(f"  {'Config':<18} {'E2E Med(ms)':<14} {'E2E P95':<12} {'Billed(ms)':<12} {'Cost($)':<14} {'DDB R/W'}")
            print(f"  {'─'*18} {'─'*14} {'─'*12} {'─'*12} {'─'*14} {'─'*12}")
            
            for cfg_label, metrics in configs.items():
                e2e = metrics.get("e2e_latency_ms")
                billed = metrics.get("total_billed_duration_ms")
                cost = metrics.get("estimated_cost_usd")
                reads = metrics.get("total_dynamo_reads")
                writes = metrics.get("total_dynamo_writes")
                
                e2e_str = f"{e2e.median:.0f} ±{e2e.iqr:.0f}" if e2e else "N/A"
                p95_str = f"{e2e.p95:.0f}" if e2e else "N/A"
                billed_str = f"{billed.median:.0f}" if billed else "N/A"
                cost_str = f"{cost.mean:.8f}" if cost else "N/A"
                ddb_str = f"{reads.median:.0f}/{writes.median:.0f}" if reads and writes else "N/A"
                
                print(f"  {cfg_label:<18} {e2e_str:<14} {p95_str:<12} {billed_str:<12} {cost_str:<14} {ddb_str}")
        
        # Print significant improvements
        sig_comps = [c for c in self.comparisons if c.significant and c.metric == "e2e_latency_ms"]
        if sig_comps:
            print(f"\n  Significant E2E Latency Improvements (p < 0.05):")
            for c in sig_comps:
                direction = "faster" if c.improvement_pct > 0 else "slower"
                print(f"    {c.config_b} vs {c.config_a}: {abs(c.improvement_pct):.1f}% {direction} "
                      f"(p={c.p_value:.4f}, d={c.cohens_d:.2f} [{c.effect_size}])")


# ─── CLI ────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone

def main():
    parser = argparse.ArgumentParser(description="Statistical Analysis of Benchmark Results")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--format", choices=["json", "text"], default="text",
                        help="Output format")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    
    analyzer = StatisticalAnalyzer(args.results_dir)
    report = analyzer.analyze_all()
    
    if args.format == "text":
        analyzer.print_summary()
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
