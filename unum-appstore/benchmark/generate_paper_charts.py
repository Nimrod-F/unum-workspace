#!/usr/bin/env python3
"""
Academic Paper Chart Generator for Unum Evaluation
====================================================
Generates publication-quality charts from benchmark results for the
Evaluation section of the research paper.

Charts generated:
  1. Grouped Bar Chart    — E2E Latency comparison (main result)
  2. Stacked Cost Chart   — Cost breakdown (Lambda + DynamoDB + S3)
  3. Heatmap              — DynamoDB I/O operations per config
  4. Box Plots            — Distribution comparison across configs
  5. CDF                  — Cumulative distribution of latencies
  6. Spider/Radar         — Multi-metric improvement radar
  7. Improvement Bars     — % improvement over baseline
  8. Cold vs Warm         — Cold start impact analysis

Usage:
    python generate_paper_charts.py results/run_20250610_143000
    python generate_paper_charts.py results/run_20250610_143000 --dpi 600
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("chart_generator")

# ─── Color Palette (colorblind-friendly, ACM/IEEE compatible) ───────────────

COLORS = {
    "SF":           "#e41a1c",   # Red
    "Unum-Base":    "#377eb8",   # Blue
    "Unum-Fus":     "#4daf4a",   # Green
    "Unum-Str":     "#984ea3",   # Purple
    "Unum-Fut":     "#ff7f00",   # Orange
    "Unum-Fut+Fus": "#a65628",   # Brown
    "Unum-Fut+Str": "#f781bf",   # Pink
    "Unum-All":     "#999999",   # Grey
}

HATCHES = {
    "SF":           "//",
    "Unum-Base":    "",
    "Unum-Fus":     "\\\\",
    "Unum-Str":     "..",
    "Unum-Fut":     "xx",
    "Unum-Fut+Fus": "++",
    "Unum-Fut+Str": "OO",
    "Unum-All":     "**",
}

CONFIG_ORDER = ["SF", "Unum-Base", "Unum-Fus", "Unum-Str", "Unum-Fut", 
                "Unum-Fut+Fus", "Unum-Fut+Str", "Unum-All"]

WORKFLOW_LABELS = {
    "nlp-pipeline": "NLP\nPipeline",
    "text-processing": "Text\nProcessing",
    "graph-analysis": "Graph\nAnalysis",
    "montecarlo-pipeline": "Monte Carlo\nPipeline",
    "wordcount": "Word\nCount",
}


class ChartGenerator:
    """Generates publication-quality charts from benchmark results."""
    
    def __init__(self, results_dir: str, dpi: int = 300, font_size: int = 10):
        self.results_dir = Path(results_dir)
        self.dpi = dpi
        self.font_size = font_size
        self.output_dir = self.results_dir / "charts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.raw_results = self._load_results()
        
        # Load statistical analysis if available
        stats_path = self.results_dir / "statistical_analysis.json"
        self.stats = json.load(open(stats_path)) if stats_path.exists() else None
        
        # Setup matplotlib
        self._setup_matplotlib()
    
    def _load_results(self) -> dict:
        """Load benchmark results."""
        all_path = self.results_dir / "all_results.json"
        if all_path.exists():
            with open(all_path) as f:
                return json.load(f)
        
        results = {}
        for f in self.results_dir.glob("*__*.json"):
            parts = f.stem.split("__")
            if len(parts) == 2:
                wf, cfg = parts
                if wf not in results:
                    results[wf] = {}
                with open(f) as fh:
                    results[wf][cfg] = json.load(fh)
        return results
    
    def _setup_matplotlib(self):
        """Configure matplotlib for publication-quality output."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": self.font_size,
            "axes.labelsize": self.font_size + 1,
            "axes.titlesize": self.font_size + 2,
            "xtick.labelsize": self.font_size - 1,
            "ytick.labelsize": self.font_size - 1,
            "legend.fontsize": self.font_size - 1,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        })
    
    def _extract_metric(self, wf_name: str, cfg_label: str, metric: str) -> np.ndarray:
        """Extract metric values from successful runs."""
        runs = self.raw_results.get(wf_name, {}).get(cfg_label, [])
        successful = [r for r in runs if r.get("success", False)]
        return np.array([r.get(metric, 0) for r in successful], dtype=float)
    
    def _get_configs_for_workflow(self, wf_name: str) -> List[str]:
        """Get available configs for a workflow in display order."""
        available = set(self.raw_results.get(wf_name, {}).keys())
        return [c for c in CONFIG_ORDER if c in available]
    
    def generate_all(self):
        """Generate all charts."""
        logger.info("Generating publication charts...")
        
        charts = [
            ("grouped_bar_e2e", self.chart_grouped_bar_e2e),
            ("stacked_cost", self.chart_stacked_cost),
            ("heatmap_dynamo", self.chart_heatmap_dynamo),
            ("boxplots_e2e", self.chart_boxplots_e2e),
            ("cdf_latency", self.chart_cdf_latency),
            ("spider_improvement", self.chart_spider_improvement),
            ("improvement_bars", self.chart_improvement_bars),
            ("cold_vs_warm", self.chart_cold_warm_comparison),
        ]
        
        for name, func in charts:
            try:
                func()
                logger.info(f"  Generated: {name}")
            except Exception as e:
                logger.warning(f"  Failed: {name} — {e}")
        
        logger.info(f"Charts saved to {self.output_dir}")
    
    # ─── Chart 1: Grouped Bar — E2E Latency ────────────────────────────
    
    def chart_grouped_bar_e2e(self):
        """Grouped bar chart of median E2E latency with error bars (IQR)."""
        import matplotlib.pyplot as plt
        
        workflows = list(self.raw_results.keys())
        n_wf = len(workflows)
        
        fig, ax = plt.subplots(figsize=(3.5 * n_wf / 3 + 2, 3.5))
        
        all_configs = set()
        for wf in workflows:
            all_configs.update(self.raw_results[wf].keys())
        configs = [c for c in CONFIG_ORDER if c in all_configs]
        n_cfg = len(configs)
        
        bar_width = 0.8 / n_cfg
        x = np.arange(n_wf)
        
        for i, cfg in enumerate(configs):
            medians = []
            iqrs_lo = []
            iqrs_hi = []
            
            for wf in workflows:
                vals = self._extract_metric(wf, cfg, "e2e_latency_ms")
                if len(vals) > 0:
                    med = np.median(vals)
                    q25 = np.percentile(vals, 25)
                    q75 = np.percentile(vals, 75)
                    medians.append(med)
                    iqrs_lo.append(med - q25)
                    iqrs_hi.append(q75 - med)
                else:
                    medians.append(0)
                    iqrs_lo.append(0)
                    iqrs_hi.append(0)
            
            offset = (i - n_cfg / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, medians, bar_width,
                yerr=[iqrs_lo, iqrs_hi],
                label=cfg,
                color=COLORS.get(cfg, "#888888"),
                hatch=HATCHES.get(cfg, ""),
                edgecolor="black",
                linewidth=0.5,
                capsize=2,
                error_kw={"linewidth": 0.7},
            )
        
        ax.set_xlabel("Workflow")
        ax.set_ylabel("E2E Latency (ms)")
        ax.set_title("End-to-End Latency Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([WORKFLOW_LABELS.get(w, w) for w in workflows])
        ax.legend(
            ncol=min(4, n_cfg), loc="upper right", framealpha=0.9,
            fontsize=self.font_size - 2
        )
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "01_grouped_bar_e2e.pdf")
        fig.savefig(self.output_dir / "01_grouped_bar_e2e.png")
        plt.close(fig)
    
    # ─── Chart 2: Stacked Cost Breakdown ───────────────────────────────
    
    def chart_stacked_cost(self):
        """Stacked bar chart showing cost breakdown: Lambda compute + DynamoDB + S3."""
        import matplotlib.pyplot as plt
        from metrics_collector import PRICING
        
        workflows = list(self.raw_results.keys())
        n_wf = len(workflows)
        
        all_configs = set()
        for wf in workflows:
            all_configs.update(self.raw_results[wf].keys())
        configs = [c for c in CONFIG_ORDER if c in all_configs]
        
        fig, axes = plt.subplots(1, n_wf, figsize=(3 * n_wf, 3.5), sharey=True)
        if n_wf == 1:
            axes = [axes]
        
        for ax, wf in zip(axes, workflows):
            cfg_labels = []
            lambda_costs = []
            dynamo_costs = []
            s3_costs = []
            
            for cfg in configs:
                runs = self.raw_results.get(wf, {}).get(cfg, [])
                successful = [r for r in runs if r.get("success", False)]
                if not successful:
                    continue
                
                cfg_labels.append(cfg)
                
                # Average costs across runs
                avg_billed = np.mean([r.get("total_billed_duration_ms", 0) for r in successful])
                avg_reads = np.mean([r.get("total_dynamo_reads", 0) for r in successful])
                avg_writes = np.mean([r.get("total_dynamo_writes", 0) for r in successful])
                avg_s3_puts = np.mean([r.get("total_s3_puts", 0) for r in successful])
                avg_s3_gets = np.mean([r.get("total_s3_gets", 0) for r in successful])
                
                # Approximate: assume 512MB average memory
                lam = (avg_billed / 1000) * (0.5) * PRICING["lambda_gb_second"]
                lam += len(successful[0].get("function_metrics", [])) * PRICING["lambda_request"]
                dyn = avg_reads * PRICING["dynamodb_rcu"] + avg_writes * PRICING["dynamodb_wcu"]
                s3 = avg_s3_puts * PRICING["s3_put"] + avg_s3_gets * PRICING["s3_get"]
                
                lambda_costs.append(lam * 1e6)   # Convert to micro-dollars for readability
                dynamo_costs.append(dyn * 1e6)
                s3_costs.append(s3 * 1e6)
            
            x = np.arange(len(cfg_labels))
            ax.bar(x, lambda_costs, label="Lambda", color="#377eb8", edgecolor="black", linewidth=0.5)
            ax.bar(x, dynamo_costs, bottom=lambda_costs, label="DynamoDB", color="#4daf4a", edgecolor="black", linewidth=0.5)
            bottom2 = np.array(lambda_costs) + np.array(dynamo_costs)
            ax.bar(x, s3_costs, bottom=bottom2, label="S3", color="#ff7f00", edgecolor="black", linewidth=0.5)
            
            ax.set_title(WORKFLOW_LABELS.get(wf, wf).replace("\n", " "), fontsize=self.font_size)
            ax.set_xticks(x)
            ax.set_xticklabels(cfg_labels, rotation=45, ha="right", fontsize=self.font_size - 2)
        
        axes[0].set_ylabel("Cost (µ$)")
        axes[-1].legend(loc="upper right", fontsize=self.font_size - 2)
        
        fig.suptitle("Cost Breakdown by Component", fontsize=self.font_size + 2, y=1.02)
        plt.tight_layout()
        fig.savefig(self.output_dir / "02_stacked_cost.pdf")
        fig.savefig(self.output_dir / "02_stacked_cost.png")
        plt.close(fig)
    
    # ─── Chart 3: Heatmap — DynamoDB I/O ──────────────────────────────
    
    def chart_heatmap_dynamo(self):
        """Heatmap of DynamoDB read+write operations."""
        import matplotlib.pyplot as plt
        
        workflows = list(self.raw_results.keys())
        all_configs = set()
        for wf in workflows:
            all_configs.update(self.raw_results[wf].keys())
        configs = [c for c in CONFIG_ORDER if c in all_configs]
        
        matrix = np.zeros((len(workflows), len(configs)))
        
        for i, wf in enumerate(workflows):
            for j, cfg in enumerate(configs):
                vals_r = self._extract_metric(wf, cfg, "total_dynamo_reads")
                vals_w = self._extract_metric(wf, cfg, "total_dynamo_writes")
                if len(vals_r) > 0:
                    matrix[i, j] = np.median(vals_r) + np.median(vals_w)
        
        fig, ax = plt.subplots(figsize=(max(4, len(configs) * 0.8 + 1), max(3, len(workflows) * 0.6 + 1)))
        
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        
        ax.set_xticks(np.arange(len(configs)))
        ax.set_yticks(np.arange(len(workflows)))
        ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=self.font_size - 2)
        ax.set_yticklabels([WORKFLOW_LABELS.get(w, w).replace("\n", " ") for w in workflows],
                           fontsize=self.font_size - 1)
        
        # Annotate cells
        for i in range(len(workflows)):
            for j in range(len(configs)):
                val = matrix[i, j]
                if val > 0:
                    color = "white" if val > matrix.max() * 0.6 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            color=color, fontsize=self.font_size - 2)
        
        ax.set_title("DynamoDB I/O Operations (Reads + Writes)")
        plt.colorbar(im, ax=ax, label="Total Operations", shrink=0.8)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "03_heatmap_dynamo.pdf")
        fig.savefig(self.output_dir / "03_heatmap_dynamo.png")
        plt.close(fig)
    
    # ─── Chart 4: Box Plots ────────────────────────────────────────────
    
    def chart_boxplots_e2e(self):
        """Box plots comparing E2E latency distributions."""
        import matplotlib.pyplot as plt
        
        workflows = list(self.raw_results.keys())
        n_wf = len(workflows)
        
        fig, axes = plt.subplots(1, n_wf, figsize=(3 * n_wf, 3.5), sharey=False)
        if n_wf == 1:
            axes = [axes]
        
        for ax, wf in zip(axes, workflows):
            configs = self._get_configs_for_workflow(wf)
            data = []
            labels = []
            colors_list = []
            
            for cfg in configs:
                vals = self._extract_metric(wf, cfg, "e2e_latency_ms")
                if len(vals) > 0:
                    data.append(vals)
                    labels.append(cfg)
                    colors_list.append(COLORS.get(cfg, "#888888"))
            
            if not data:
                continue
            
            bp = ax.boxplot(
                data, labels=labels, patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"linewidth": 0.8},
                flierprops={"markersize": 3, "alpha": 0.5},
            )
            
            for patch, color in zip(bp["boxes"], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(WORKFLOW_LABELS.get(wf, wf).replace("\n", " "), fontsize=self.font_size)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=self.font_size - 2)
            ax.set_ylabel("E2E Latency (ms)" if ax == axes[0] else "")
        
        fig.suptitle("E2E Latency Distribution", fontsize=self.font_size + 2, y=1.02)
        plt.tight_layout()
        fig.savefig(self.output_dir / "04_boxplots_e2e.pdf")
        fig.savefig(self.output_dir / "04_boxplots_e2e.png")
        plt.close(fig)
    
    # ─── Chart 5: CDF ──────────────────────────────────────────────────
    
    def chart_cdf_latency(self):
        """CDF of E2E latency for each workflow."""
        import matplotlib.pyplot as plt
        
        workflows = list(self.raw_results.keys())
        n_wf = len(workflows)
        
        fig, axes = plt.subplots(1, n_wf, figsize=(3 * n_wf, 3), sharey=True)
        if n_wf == 1:
            axes = [axes]
        
        for ax, wf in zip(axes, workflows):
            configs = self._get_configs_for_workflow(wf)
            
            for cfg in configs:
                vals = self._extract_metric(wf, cfg, "e2e_latency_ms")
                if len(vals) == 0:
                    continue
                sorted_vals = np.sort(vals)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                ax.step(sorted_vals, cdf, label=cfg, color=COLORS.get(cfg, "#888"),
                        linewidth=1.2)
            
            ax.set_title(WORKFLOW_LABELS.get(wf, wf).replace("\n", " "), fontsize=self.font_size)
            ax.set_xlabel("E2E Latency (ms)")
            ax.set_ylabel("CDF" if ax == axes[0] else "")
            ax.legend(fontsize=self.font_size - 3, loc="lower right")
        
        fig.suptitle("CDF of End-to-End Latency", fontsize=self.font_size + 2, y=1.02)
        plt.tight_layout()
        fig.savefig(self.output_dir / "05_cdf_latency.pdf")
        fig.savefig(self.output_dir / "05_cdf_latency.png")
        plt.close(fig)
    
    # ─── Chart 6: Spider/Radar — Multi-Metric ─────────────────────────
    
    def chart_spider_improvement(self):
        """Spider/radar chart showing normalized improvement across metrics."""
        import matplotlib.pyplot as plt
        
        if not self.stats:
            logger.warning("No statistical analysis available for spider chart")
            return
        
        metrics_to_show = [
            ("e2e_latency_ms", "E2E Latency"),
            ("total_billed_duration_ms", "Billed Duration"),
            ("estimated_cost_usd", "Cost"),
            ("total_dynamo_reads", "DDB Reads"),
            ("total_dynamo_writes", "DDB Writes"),
            ("peak_memory_mb", "Peak Memory"),
        ]
        
        # Average improvement across all workflows for each config
        configs_to_plot = ["Unum-Fus", "Unum-Str", "Unum-Fut", "Unum-All"]
        
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection="polar"))
        
        n_metrics = len(metrics_to_show)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        for cfg in configs_to_plot:
            improvements = []
            for metric_key, _ in metrics_to_show:
                # Get average improvement across workflows from comparisons
                comps = [
                    c for c in self.stats.get("pairwise_comparisons", [])
                    if c.get("config_b") == cfg and c.get("metric") == metric_key
                ]
                if comps:
                    avg_imp = np.mean([c.get("improvement_pct", 0) for c in comps])
                    improvements.append(max(0, avg_imp))  # Clip negative
                else:
                    improvements.append(0)
            
            improvements += improvements[:1]  # Close
            ax.plot(angles, improvements, linewidth=1.5, label=cfg,
                    color=COLORS.get(cfg, "#888"))
            ax.fill(angles, improvements, alpha=0.1, color=COLORS.get(cfg, "#888"))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label for _, label in metrics_to_show], fontsize=self.font_size - 2)
        ax.set_title("Improvement Over Baseline (%)", fontsize=self.font_size + 1, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=self.font_size - 2)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "06_spider_improvement.pdf")
        fig.savefig(self.output_dir / "06_spider_improvement.png")
        plt.close(fig)
    
    # ─── Chart 7: Improvement Bars ─────────────────────────────────────
    
    def chart_improvement_bars(self):
        """Horizontal bar chart of % improvement over Unum-Base for E2E latency."""
        import matplotlib.pyplot as plt
        
        workflows = list(self.raw_results.keys())
        configs_to_compare = [c for c in CONFIG_ORDER if c not in ("SF", "Unum-Base")]
        
        fig, axes = plt.subplots(len(workflows), 1, figsize=(6, 2 * len(workflows)), sharex=True)
        if len(workflows) == 1:
            axes = [axes]
        
        for ax, wf in zip(axes, workflows):
            improvements = []
            labels = []
            colors_list = []
            
            base_vals = self._extract_metric(wf, "Unum-Base", "e2e_latency_ms")
            if len(base_vals) == 0:
                continue
            base_median = np.median(base_vals)
            
            for cfg in configs_to_compare:
                vals = self._extract_metric(wf, cfg, "e2e_latency_ms")
                if len(vals) == 0:
                    continue
                
                imp = (base_median - np.median(vals)) / base_median * 100
                improvements.append(imp)
                labels.append(cfg)
                colors_list.append(COLORS.get(cfg, "#888"))
            
            if not improvements:
                continue
            
            y = np.arange(len(labels))
            bars = ax.barh(y, improvements, color=colors_list, edgecolor="black", linewidth=0.5, height=0.6)
            
            # Add value labels
            for bar, imp in zip(bars, improvements):
                x_pos = bar.get_width()
                ha = "left" if x_pos >= 0 else "right"
                offset = 1 if x_pos >= 0 else -1
                ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                        f"{imp:.1f}%", va="center", ha=ha, fontsize=self.font_size - 2)
            
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=self.font_size - 1)
            ax.set_title(WORKFLOW_LABELS.get(wf, wf).replace("\n", " "), fontsize=self.font_size, loc="left")
            ax.axvline(x=0, color="black", linewidth=0.8)
        
        axes[-1].set_xlabel("Improvement over Unum-Base (%)")
        fig.suptitle("E2E Latency Improvement", fontsize=self.font_size + 2, y=1.01)
        plt.tight_layout()
        fig.savefig(self.output_dir / "07_improvement_bars.pdf")
        fig.savefig(self.output_dir / "07_improvement_bars.png")
        plt.close(fig)
    
    # ─── Chart 8: Cold vs Warm ─────────────────────────────────────────
    
    def chart_cold_warm_comparison(self):
        """Grouped comparison of cold start vs warm invocation latencies."""
        import matplotlib.pyplot as plt
        
        workflows = list(self.raw_results.keys())
        
        fig, axes = plt.subplots(1, len(workflows), figsize=(3 * len(workflows), 3.5), sharey=False)
        if len(workflows) == 1:
            axes = [axes]
        
        for ax, wf in zip(axes, workflows):
            configs = self._get_configs_for_workflow(wf)
            cold_medians = []
            warm_medians = []
            cfg_labels = []
            
            for cfg in configs:
                runs = self.raw_results.get(wf, {}).get(cfg, [])
                successful = [r for r in runs if r.get("success", False)]
                
                cold = [r["e2e_latency_ms"] for r in successful if r.get("cold_start_count", 0) > 0]
                warm = [r["e2e_latency_ms"] for r in successful if r.get("cold_start_count", 0) == 0]
                
                if cold or warm:
                    cold_medians.append(np.median(cold) if cold else 0)
                    warm_medians.append(np.median(warm) if warm else 0)
                    cfg_labels.append(cfg)
            
            if not cfg_labels:
                continue
            
            x = np.arange(len(cfg_labels))
            width = 0.35
            
            ax.bar(x - width/2, cold_medians, width, label="Cold", color="#e41a1c", 
                   edgecolor="black", linewidth=0.5, alpha=0.8)
            ax.bar(x + width/2, warm_medians, width, label="Warm", color="#377eb8",
                   edgecolor="black", linewidth=0.5, alpha=0.8)
            
            ax.set_title(WORKFLOW_LABELS.get(wf, wf).replace("\n", " "), fontsize=self.font_size)
            ax.set_xticks(x)
            ax.set_xticklabels(cfg_labels, rotation=45, ha="right", fontsize=self.font_size - 2)
            ax.set_ylabel("E2E Latency (ms)" if ax == axes[0] else "")
            ax.legend(fontsize=self.font_size - 2)
        
        fig.suptitle("Cold Start vs Warm Invocation", fontsize=self.font_size + 2, y=1.02)
        plt.tight_layout()
        fig.savefig(self.output_dir / "08_cold_vs_warm.pdf")
        fig.savefig(self.output_dir / "08_cold_vs_warm.png")
        plt.close(fig)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Paper Charts")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")
    parser.add_argument("--font-size", type=int, default=10, help="Base font size")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    
    gen = ChartGenerator(args.results_dir, dpi=args.dpi, font_size=args.font_size)
    gen.generate_all()


if __name__ == "__main__":
    main()
