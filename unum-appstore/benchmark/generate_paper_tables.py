#!/usr/bin/env python3
"""
LaTeX Table Generator for Unum Evaluation Paper
=================================================
Generates publication-ready LaTeX tables from benchmark results.

Tables generated:
  1. Main Results          — E2E latency, cost, billed duration per config/workflow
  2. Cost Breakdown        — Lambda, DynamoDB, S3 per component
  3. DynamoDB I/O          — Read/write operations comparison
  4. Statistical Tests     — Mann-Whitney U, Cohen's d, significance
  5. Enhancement Impact    — % improvement per technique, per topology

Usage:
    python generate_paper_tables.py results/run_20250610_143000
    python generate_paper_tables.py results/run_20250610_143000 --output tables/
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("table_generator")

METRICS_DISPLAY = {
    "e2e_latency_ms": "E2E Latency (ms)",
    "aggregator_delay_ms": "Agg. Delay (ms)",
    "total_billed_duration_ms": "Billed Duration (ms)",
    "peak_memory_mb": "Peak Memory (MB)",
    "cold_start_count": "Cold Starts",
    "total_dynamo_reads": "DDB Reads",
    "total_dynamo_writes": "DDB Writes",
    "estimated_cost_usd": "Cost (\\$)",
}

CONFIG_ORDER = ["SF", "Unum-Base", "Unum-Fus", "Unum-Str", "Unum-Fut",
                "Unum-Fut+Fus", "Unum-Fut+Str", "Unum-All"]

WORKFLOW_SHORT = {
    "nlp-pipeline": "NLP",
    "text-processing": "TextProc",
    "graph-analysis": "Graph",
    "montecarlo-pipeline": "MonteCarlo",
    "wordcount": "WordCount",
}


class TableGenerator:
    """Generates LaTeX tables for the paper's Evaluation section."""
    
    def __init__(self, results_dir: str, output_dir: Optional[str] = None):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "tables"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.raw_results = self._load_results()
        
        stats_path = self.results_dir / "statistical_analysis.json"
        self.stats = json.load(open(stats_path)) if stats_path.exists() else None
    
    def _load_results(self) -> dict:
        all_path = self.results_dir / "all_results.json"
        if all_path.exists():
            with open(all_path) as f:
                return json.load(f)
        results = {}
        for f in self.results_dir.glob("*__*.json"):
            parts = f.stem.split("__")
            if len(parts) == 2:
                wf, cfg = parts
                results.setdefault(wf, {})[cfg] = json.load(open(f))
        return results
    
    def _extract_metric(self, wf: str, cfg: str, metric: str) -> np.ndarray:
        runs = self.raw_results.get(wf, {}).get(cfg, [])
        successful = [r for r in runs if r.get("success", False)]
        return np.array([r.get(metric, 0) for r in successful], dtype=float)
    
    def generate_all(self):
        """Generate all LaTeX tables."""
        tables = [
            ("table1_main_results", self.table_main_results),
            ("table2_cost_breakdown", self.table_cost_breakdown),
            ("table3_dynamo_io", self.table_dynamo_io),
            ("table4_statistical_tests", self.table_statistical_tests),
            ("table5_enhancement_impact", self.table_enhancement_impact),
        ]
        
        all_latex = []
        for name, func in tables:
            try:
                latex = func()
                path = self.output_dir / f"{name}.tex"
                with open(path, "w") as f:
                    f.write(latex)
                all_latex.append(f"% ─── {name} ───\n{latex}\n")
                logger.info(f"  Generated: {name}")
            except Exception as e:
                logger.warning(f"  Failed: {name} — {e}")
        
        # Combined file
        combined_path = self.output_dir / "all_tables.tex"
        with open(combined_path, "w") as f:
            f.write("% Auto-generated LaTeX tables for Unum Evaluation\n")
            f.write("% Include with \\input{tables/all_tables}\n\n")
            f.write("\n\n".join(all_latex))
        
        logger.info(f"Tables saved to {self.output_dir}")
    
    # ─── Table 1: Main Results ─────────────────────────────────────────
    
    def table_main_results(self) -> str:
        """
        Main results table: median E2E latency (±IQR) for each workflow × config.
        """
        workflows = list(self.raw_results.keys())
        all_configs = set()
        for wf in workflows:
            all_configs.update(self.raw_results[wf].keys())
        configs = [c for c in CONFIG_ORDER if c in all_configs]
        
        n_cols = len(configs) + 1
        col_spec = "l" + "r" * len(configs)
        
        lines = []
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{End-to-End Latency (ms): Median $\pm$ IQR across configurations. " 
                      r"Bold indicates best per workflow. $\dagger$ indicates $p < 0.05$ vs.\ Unum-Base.}")
        lines.append(r"\label{tab:main-results}")
        lines.append(r"\small")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
        
        # Header
        header = "\\textbf{Workflow}"
        for cfg in configs:
            header += f" & \\textbf{{{self._escape(cfg)}}}"
        header += r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        
        # Data rows
        for wf in workflows:
            row_vals = []
            for cfg in configs:
                vals = self._extract_metric(wf, cfg, "e2e_latency_ms")
                if len(vals) > 0:
                    med = np.median(vals)
                    iqr_val = np.percentile(vals, 75) - np.percentile(vals, 25)
                    row_vals.append((med, iqr_val, cfg))
                else:
                    row_vals.append(None)
            
            # Find best (lowest median)
            valid_vals = [(v[0], i) for i, v in enumerate(row_vals) if v is not None]
            best_idx = min(valid_vals, key=lambda x: x[0])[1] if valid_vals else -1
            
            row = WORKFLOW_SHORT.get(wf, wf)
            for i, v in enumerate(row_vals):
                if v is None:
                    row += " & ---"
                else:
                    med, iqr_val, cfg = v
                    cell = f"{med:.0f} $\\pm$ {iqr_val:.0f}"
                    
                    # Check significance
                    sig = self._is_significant(wf, cfg, "e2e_latency_ms")
                    if sig:
                        cell += "$^\\dagger$"
                    
                    if i == best_idx:
                        cell = f"\\textbf{{{cell}}}"
                    
                    row += f" & {cell}"
            
            row += r" \\"
            lines.append(row)
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")
        
        return "\n".join(lines)
    
    # ─── Table 2: Cost Breakdown ───────────────────────────────────────
    
    def table_cost_breakdown(self) -> str:
        """Cost breakdown: Lambda compute + DynamoDB + Total per config."""
        from metrics_collector import PRICING
        
        workflows = list(self.raw_results.keys())
        all_configs = set()
        for wf in workflows:
            all_configs.update(self.raw_results[wf].keys())
        configs = [c for c in CONFIG_ORDER if c in all_configs]
        
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Average Cost per Invocation (µ\$). Lower is better.}")
        lines.append(r"\label{tab:cost-breakdown}")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{ll" + "r" * len(configs) + "}")
        lines.append(r"\toprule")
        
        header = r"\textbf{Workflow} & \textbf{Component}"
        for cfg in configs:
            header += f" & \\textbf{{{self._escape(cfg)}}}"
        header += r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        
        for wf in workflows:
            for component in ["Lambda", "DynamoDB", "Total"]:
                row = ""
                if component == "Lambda":
                    row = f"\\multirow{{3}}{{*}}{{{WORKFLOW_SHORT.get(wf, wf)}}}"
                row += f" & {component}"
                
                for cfg in configs:
                    runs = self.raw_results.get(wf, {}).get(cfg, [])
                    successful = [r for r in runs if r.get("success", False)]
                    
                    if not successful:
                        row += " & ---"
                        continue
                    
                    if component == "Lambda":
                        # Approximate Lambda cost
                        avg_billed = np.mean([r.get("total_billed_duration_ms", 0) for r in successful])
                        cost = (avg_billed / 1000) * 0.5 * PRICING["lambda_gb_second"]
                        n_funcs = np.mean([len(r.get("function_metrics", [])) for r in successful])
                        cost += n_funcs * PRICING["lambda_request"]
                        row += f" & {cost * 1e6:.2f}"
                    
                    elif component == "DynamoDB":
                        avg_reads = np.mean([r.get("total_dynamo_reads", 0) for r in successful])
                        avg_writes = np.mean([r.get("total_dynamo_writes", 0) for r in successful])
                        cost = avg_reads * PRICING["dynamodb_rcu"] + avg_writes * PRICING["dynamodb_wcu"]
                        row += f" & {cost * 1e6:.2f}"
                    
                    elif component == "Total":
                        avg_cost = np.mean([r.get("estimated_cost_usd", 0) for r in successful])
                        row += f" & {avg_cost * 1e6:.2f}"
                
                row += r" \\"
                lines.append(row)
            
            if wf != workflows[-1]:
                lines.append(r"\midrule")
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        
        return "\n".join(lines)
    
    # ─── Table 3: DynamoDB I/O ─────────────────────────────────────────
    
    def table_dynamo_io(self) -> str:
        """DynamoDB read/write operations comparison."""
        workflows = list(self.raw_results.keys())
        all_configs = set()
        for wf in workflows:
            all_configs.update(self.raw_results[wf].keys())
        configs = [c for c in CONFIG_ORDER if c in all_configs]
        
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{DynamoDB I/O Operations per Invocation (Median). " 
                      r"R = Reads, W = Writes.}")
        lines.append(r"\label{tab:dynamo-io}")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{l" + "c" * len(configs) + "}")
        lines.append(r"\toprule")
        
        header = r"\textbf{Workflow}"
        for cfg in configs:
            header += f" & \\textbf{{{self._escape(cfg)}}}"
        header += r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        
        for wf in workflows:
            row = WORKFLOW_SHORT.get(wf, wf)
            for cfg in configs:
                reads = self._extract_metric(wf, cfg, "total_dynamo_reads")
                writes = self._extract_metric(wf, cfg, "total_dynamo_writes")
                
                if len(reads) > 0:
                    r_med = int(np.median(reads))
                    w_med = int(np.median(writes))
                    row += f" & {r_med}R/{w_med}W"
                else:
                    row += " & ---"
            
            row += r" \\"
            lines.append(row)
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        
        return "\n".join(lines)
    
    # ─── Table 4: Statistical Tests ────────────────────────────────────
    
    def table_statistical_tests(self) -> str:
        """Mann-Whitney U test results and Cohen's d for E2E latency."""
        if not self.stats:
            return "% No statistical analysis available"
        
        comparisons = self.stats.get("pairwise_comparisons", [])
        e2e_comps = [c for c in comparisons if c.get("metric") == "e2e_latency_ms"]
        
        if not e2e_comps:
            return "% No E2E comparisons available"
        
        # Group by workflow
        by_workflow = {}
        for c in e2e_comps:
            wf = c.get("workflow", "unknown")
            by_workflow.setdefault(wf, []).append(c)
        
        lines = []
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Statistical Significance of E2E Latency Improvements vs.\ Unum-Base. "
                      r"$^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$.}")
        lines.append(r"\label{tab:stat-tests}")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{llrrrl}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Workflow} & \textbf{Config} & \textbf{Impr. (\%)} & "
                      r"\textbf{$U$} & \textbf{$p$-value} & \textbf{Cohen's $d$} \\")
        lines.append(r"\midrule")
        
        for wf, comps in by_workflow.items():
            for i, c in enumerate(comps):
                wf_label = WORKFLOW_SHORT.get(wf, wf) if i == 0 else ""
                cfg = c.get("config_b", "")
                imp = c.get("improvement_pct", 0)
                u_val = c.get("mann_whitney_u", 0)
                p_val = c.get("p_value", 1)
                d_val = c.get("cohens_d", 0)
                effect = c.get("effect_size", "")
                
                # Significance stars
                if p_val < 0.001:
                    sig = "$^{***}$"
                elif p_val < 0.01:
                    sig = "$^{**}$"
                elif p_val < 0.05:
                    sig = "$^{*}$"
                else:
                    sig = ""
                
                sign = "+" if imp > 0 else ""
                row = (f"{wf_label} & {self._escape(cfg)}{sig} & {sign}{imp:.1f} & "
                       f"{u_val:.0f} & {p_val:.4f} & {d_val:.2f} ({effect})")
                row += r" \\"
                lines.append(row)
            
            lines.append(r"\midrule")
        
        # Remove last midrule
        if lines[-1] == r"\midrule":
            lines[-1] = r"\bottomrule"
        
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")
        
        return "\n".join(lines)
    
    # ─── Table 5: Enhancement Impact ───────────────────────────────────
    
    def table_enhancement_impact(self) -> str:
        """
        Summary table: which enhancement helps which topology type most.
        Rows = Techniques (Fusion, Streaming, Future-Based, Combined)
        Columns = Topology types
        """
        technique_configs = {
            "Fusion": "Unum-Fus",
            "Streaming": "Unum-Str",
            "Future-Based": "Unum-Fut",
            "All Combined": "Unum-All",
        }
        
        topology_workflows = {
            "Chain": ["nlp-pipeline"],
            "Parallel/Fan-in": ["text-processing", "graph-analysis"],
            "Diamond": ["montecarlo-pipeline"],
            "MapReduce": ["wordcount"],
        }
        
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{E2E Latency Improvement (\%) by Enhancement Technique and Topology. " 
                      r"``---'' indicates technique not applicable.}")
        lines.append(r"\label{tab:enhancement-impact}")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{l" + "r" * len(topology_workflows) + "}")
        lines.append(r"\toprule")
        
        header = r"\textbf{Technique}"
        for topo in topology_workflows:
            header += f" & \\textbf{{{topo}}}"
        header += r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        
        for tech_name, cfg_label in technique_configs.items():
            row = tech_name
            
            for topo, wf_list in topology_workflows.items():
                improvements = []
                for wf in wf_list:
                    base = self._extract_metric(wf, "Unum-Base", "e2e_latency_ms")
                    enhanced = self._extract_metric(wf, cfg_label, "e2e_latency_ms")
                    
                    if len(base) > 0 and len(enhanced) > 0:
                        imp = (np.median(base) - np.median(enhanced)) / np.median(base) * 100
                        improvements.append(imp)
                
                if improvements:
                    avg_imp = np.mean(improvements)
                    sign = "+" if avg_imp > 0 else ""
                    row += f" & {sign}{avg_imp:.1f}"
                else:
                    row += " & ---"
            
            row += r" \\"
            lines.append(row)
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        
        return "\n".join(lines)
    
    # ─── Utilities ─────────────────────────────────────────────────────
    
    def _escape(self, text: str) -> str:
        """Escape LaTeX special characters."""
        return text.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
    
    def _is_significant(self, wf: str, cfg: str, metric: str) -> bool:
        """Check if a config is significantly different from baseline."""
        if not self.stats:
            return False
        
        for comp in self.stats.get("pairwise_comparisons", []):
            if (comp.get("workflow") == wf and 
                comp.get("config_b") == cfg and 
                comp.get("metric") == metric):
                return comp.get("significant", False)
        
        return False


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX Tables")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for .tex files")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    
    gen = TableGenerator(args.results_dir, output_dir=args.output)
    gen.generate_all()


if __name__ == "__main__":
    main()
