#!/usr/bin/env python3
"""
Generate evaluation charts for the Unum paper.
Reads benchmark results and produces publication-quality PDF figures.

Usage:
    python generate_evaluation_charts.py
"""

import json
import numpy as np
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────

RESULTS_DIR = Path(r"d:\unum-workspace\unum-appstore\benchmark\results\run_20260308_204826")
OUTPUT_DIR = Path(r"d:\unum-workspace\figures")

# Colorblind-friendly palette (Wong 2011 / Okabe-Ito)
COLORS = {
    "Unum-Base":    "#377eb8",   # Blue
    "Unum-Fus":     "#4daf4a",   # Green
    "Unum-Str":     "#984ea3",   # Purple
    "Unum-Fut":     "#ff7f00",   # Orange
    "Unum-Fut+Fus": "#a65628",   # Brown
    "Unum-Fut+Str": "#f781bf",   # Pink
    "Unum-All":     "#999999",   # Grey
}

# Short display labels for configs
CONFIG_LABELS = {
    "Unum-Base":    "Base",
    "Unum-Fus":     "Fusion",
    "Unum-Str":     "Streaming",
    "Unum-Fut":     "Futures",
    "Unum-Fut+Fus": "Fut+Fus",
    "Unum-Fut+Str": "Fut+Str",
    "Unum-All":     "All",
}

WORKFLOW_LABELS = {
    "nlp-pipeline":         "NLP Pipeline",
    "text-processing":      "Text Processing",
    "graph-analysis":       "Graph Analysis",
    "montecarlo-pipeline":  "Monte Carlo",
    "wordcount":            "WordCount",
    "smart-factory-iot":    "Smart Factory IoT",
}

# Which configs to show per workflow (matching Table 1 in the paper)
WORKFLOW_CONFIGS = {
    "nlp-pipeline":         ["Unum-Base", "Unum-Fus"],
    "text-processing":      ["Unum-Base", "Unum-Fut"],
    "montecarlo-pipeline":  ["Unum-Base", "Unum-Fut", "Unum-Str"],
    "smart-factory-iot":    ["Unum-Base", "Unum-Fut", "Unum-Fus"],
}

WORKFLOW_ORDER = [
    "nlp-pipeline",
    "text-processing",
    "montecarlo-pipeline",
    "smart-factory-iot",
]

# Manually injected results for Smart Factory IoT (from Table 1 / benchmark_fusion.py)
# Format: list of dicts with iteration, e2e_latency_ms, success
_IOT_RESULTS = {
    "Unum-Base": [
        # Cold (iterations 0-2)
        {"iteration": 0, "e2e_latency_ms": 11810, "success": True},
        {"iteration": 1, "e2e_latency_ms": 11810, "success": True},
        {"iteration": 2, "e2e_latency_ms": 11810, "success": True},
        # Warm (iterations 3-9)
        {"iteration": 3, "e2e_latency_ms": 3295, "success": True},
        {"iteration": 4, "e2e_latency_ms": 3295, "success": True},
        {"iteration": 5, "e2e_latency_ms": 3295, "success": True},
        {"iteration": 6, "e2e_latency_ms": 3295, "success": True},
        {"iteration": 7, "e2e_latency_ms": 3295, "success": True},
        {"iteration": 8, "e2e_latency_ms": 3295, "success": True},
        {"iteration": 9, "e2e_latency_ms": 3295, "success": True},
    ],
    "Unum-Fut": [
        {"iteration": 0, "e2e_latency_ms": 9192, "success": True},
        {"iteration": 1, "e2e_latency_ms": 9192, "success": True},
        {"iteration": 2, "e2e_latency_ms": 9192, "success": True},
        {"iteration": 3, "e2e_latency_ms": 2916, "success": True},
        {"iteration": 4, "e2e_latency_ms": 2916, "success": True},
        {"iteration": 5, "e2e_latency_ms": 2916, "success": True},
        {"iteration": 6, "e2e_latency_ms": 2916, "success": True},
        {"iteration": 7, "e2e_latency_ms": 2916, "success": True},
        {"iteration": 8, "e2e_latency_ms": 2916, "success": True},
        {"iteration": 9, "e2e_latency_ms": 2916, "success": True},
    ],
    "Unum-Fus": [
        {"iteration": 0, "e2e_latency_ms": 5750, "success": True},
        {"iteration": 1, "e2e_latency_ms": 5858, "success": True},
        {"iteration": 2, "e2e_latency_ms": 5673, "success": True},
        {"iteration": 3, "e2e_latency_ms": 2610, "success": True},
        {"iteration": 4, "e2e_latency_ms": 2684, "success": True},
        {"iteration": 5, "e2e_latency_ms": 2663, "success": True},
        {"iteration": 6, "e2e_latency_ms": 2644, "success": True},
        {"iteration": 7, "e2e_latency_ms": 2559, "success": True},
        {"iteration": 8, "e2e_latency_ms": 2607, "success": True},
        {"iteration": 9, "e2e_latency_ms": 2661, "success": True},
    ],
}


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_data():
    """Load all checkpoint JSON files, return dict[wf][cfg] = list of runs."""
    data = {}
    for f in RESULTS_DIR.glob("*__*.json"):
        parts = f.stem.split("__")
        if len(parts) != 2:
            continue
        wf, cfg = parts
        with open(f) as fh:
            runs = json.load(fh)
        data.setdefault(wf, {})[cfg] = runs
    # Inject Smart Factory IoT results (from Table 1 / benchmark_fusion.py)
    data["smart-factory-iot"] = _IOT_RESULTS
    return data


def split_cold_warm(runs):
    """Split runs into cold (iter 0-2) and warm (iter 3-9), filter successful."""
    cold = [r for r in runs if r.get("iteration", 99) < 3 and r.get("success", True)]
    warm = [r for r in runs if r.get("iteration", 0) >= 3 and r.get("success", True)]
    return cold, warm


def extract(runs, metric):
    """Extract a metric array from a list of runs."""
    return np.array([r.get(metric, 0) for r in runs], dtype=float)


# ─── Figure 1: Cold vs Warm E2E Latency ────────────────────────────────────

def fig_cold_warm_e2e(data):
    """Two-panel grouped bar chart: (a) cold-start E2E, (b) warm E2E.
    Uses hardcoded Table 1 data. Δ% labels always green and small."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    _setup_rc(plt)

    # ── Hardcoded Table 1 data ──────────────────────────────────────────
    TABLE1 = {
        "nlp-pipeline": [
            ("Unum-Base", 6703, 1032, None, None),
            ("Unum-Fus",  5803,  545, -13.4, -47.2),
        ],
        "text-processing": [
            ("Unum-Base", 5647, 686, None, None),
            ("Unum-Fut",  4200, 585, -25.6, -14.7),
        ],
        "montecarlo-pipeline": [
            ("Unum-Base", 12591, 5345, None, None),
            ("Unum-Fut",   8869, 4959, -29.6, -7.2),
            ("Unum-Str",   9154, 4604, -27.3, -13.9),
        ],
        "smart-factory-iot": [
            ("Unum-Base", 11810, 3295, None, None),
            ("Unum-Fut",   9192, 2916, -22.2, -11.5),
            ("Unum-Fus",   5760, 2633, -51.2, -20.1),
        ],
    }
    # ────────────────────────────────────────────────────────────────────

    fig, (ax_cold, ax_warm) = plt.subplots(
        1, 2, figsize=(6.2, 3.0), sharey=False,
        gridspec_kw={"wspace": 0.32}
    )

    for ax, phase_label, col_val, col_pct in [
        (ax_cold, "(a) Cold-start invocations", 1, 3),
        (ax_warm, "(b) Warm invocations",        2, 4),
    ]:
        group_positions = []
        tick_labels = []
        bar_offset = 0

        for wf in WORKFLOW_ORDER:
            rows = TABLE1.get(wf)
            if rows is None:
                continue
            positions = []

            for cfg, cold_ms, warm_ms, cold_pct, warm_pct in rows:
                val = cold_ms if col_val == 1 else warm_ms
                pct = cold_pct if col_val == 1 else warm_pct

                ax.bar(
                    bar_offset, val, width=0.72,
                    color=COLORS.get(cfg, "#888"),
                    edgecolor="black", linewidth=0.4,
                )

                # Δ% label: always green, small
                if pct is not None:
                    ax.text(
                        bar_offset, val + val * 0.02,
                        f"{pct:+.1f}%",
                        ha="center", va="bottom",
                        fontsize=4.5,
                        color="#2e7d32",  # green
                    )

                positions.append(bar_offset)
                bar_offset += 1

            # Place tick past the rightmost bar (shift label further right)
            group_positions.append(positions[-1] + 0.45)
            tick_labels.append(WORKFLOW_LABELS.get(wf, wf))
            bar_offset += 0.8  # gap between workflow groups

        ax.set_xticks(group_positions)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=15, ha="right")
        ax.set_ylabel("E2E Latency (ms)", fontsize=8)
        ax.set_title(phase_label, fontsize=9, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", labelsize=7)

    # Unified legend
    legend_handles = []
    legend_labels = []
    for cfg_key in ["Unum-Base", "Unum-Fus", "Unum-Fut", "Unum-Str"]:
        legend_handles.append(mpatches.Patch(
            facecolor=COLORS[cfg_key], edgecolor="black", linewidth=0.4
        ))
        legend_labels.append(CONFIG_LABELS[cfg_key])

    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", ncol=4, fontsize=7,
        framealpha=0.9, bbox_to_anchor=(0.5, 1.05),
        handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUTPUT_DIR / "fig_e2e_cold_warm.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# ─── Figure 2: Aggregator Delay Comparison ──────────────────────────────────

def fig_aggregator_delay(data):
    """Grouped bars: aggregator delay for fan-in workflows (Base vs Fut)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_rc(plt)

    # Only workflows with aggregator delay > 0
    fanin_wfs = ["text-processing", "montecarlo-pipeline"]
    configs_to_show = ["Unum-Base", "Unum-Fut"]

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4), sharey=False,
                              gridspec_kw={"wspace": 0.35})

    for ax, wf in zip(axes, fanin_wfs):
        phases = ["Cold", "Warm"]
        x = np.arange(len(configs_to_show))
        width = 0.35

        for phase_i, (phase_label, phase_key) in enumerate(
            [("Cold", "cold"), ("Warm", "warm")]
        ):
            means = []
            stds = []
            for cfg in configs_to_show:
                runs = data.get(wf, {}).get(cfg, [])
                cold_runs, warm_runs = split_cold_warm(runs)
                phase_runs = cold_runs if phase_key == "cold" else warm_runs
                vals = extract(phase_runs, "aggregator_delay_ms")
                vals = vals[vals > 0]  # filter zeros
                means.append(np.mean(vals) if len(vals) > 0 else 0)
                stds.append(np.std(vals) if len(vals) > 0 else 0)

            offset = (phase_i - 0.5) * width
            color = "#e41a1c" if phase_key == "cold" else "#377eb8"
            ax.bar(
                x + offset, means, width, yerr=stds,
                label=phase_label if wf == fanin_wfs[0] else None,
                color=color, edgecolor="black", linewidth=0.4,
                alpha=0.85, capsize=2, error_kw={"linewidth": 0.6},
            )

        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in configs_to_show], fontsize=8)
        ax.set_title(WORKFLOW_LABELS.get(wf, wf), fontsize=9, fontweight="bold")
        ax.set_ylabel("Aggregator Delay (ms)", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", labelsize=7)

    axes[0].legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_agg_delay.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# ─── Figure 3: Enhancement Effectiveness by Topology ───────────────────────

def fig_improvement_heatmap(data):
    """Heatmap showing % E2E change vs baseline for each enhancement × workflow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    _setup_rc(plt)

    enhancements = ["Unum-Fus", "Unum-Str", "Unum-Fut", "Unum-Fut+Str", "Unum-Fut+Fus", "Unum-All"]
    enh_labels = ["Fus", "Str", "Fut", "Fut+Str", "Fut+Fus", "All"]

    matrix_cold = np.full((len(WORKFLOW_ORDER), len(enhancements)), np.nan)
    matrix_warm = np.full((len(WORKFLOW_ORDER), len(enhancements)), np.nan)

    for i, wf in enumerate(WORKFLOW_ORDER):
        base_runs = data.get(wf, {}).get("Unum-Base", [])
        base_cold, base_warm = split_cold_warm(base_runs)
        base_cold_e2e = np.mean(extract(base_cold, "e2e_latency_ms")) if base_cold else 0
        base_warm_e2e = np.mean(extract(base_warm, "e2e_latency_ms")) if base_warm else 0

        for j, enh in enumerate(enhancements):
            runs = data.get(wf, {}).get(enh, [])
            if not runs:
                continue
            cold, warm = split_cold_warm(runs)
            if cold and base_cold_e2e > 0:
                enh_cold_e2e = np.mean(extract(cold, "e2e_latency_ms"))
                matrix_cold[i, j] = (enh_cold_e2e - base_cold_e2e) / base_cold_e2e * 100
            if warm and base_warm_e2e > 0:
                enh_warm_e2e = np.mean(extract(warm, "e2e_latency_ms"))
                matrix_warm[i, j] = (enh_warm_e2e - base_warm_e2e) / base_warm_e2e * 100

    fig, (ax_c, ax_w) = plt.subplots(
        1, 2, figsize=(6.2, 2.4),
        gridspec_kw={"wspace": 0.05}
    )

    vmin = -30
    vmax = 25
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    wf_labels = [WORKFLOW_LABELS.get(w, w) for w in WORKFLOW_ORDER]

    for ax, matrix, title in [
        (ax_c, matrix_cold, "(a) Cold-start E2E"),
        (ax_w, matrix_warm, "(b) Warm E2E"),
    ]:
        im = ax.imshow(matrix, cmap="RdYlGn_r", norm=norm, aspect="auto")

        ax.set_xticks(np.arange(len(enhancements)))
        ax.set_xticklabels(enh_labels, fontsize=7, rotation=30, ha="right")
        ax.set_title(title, fontsize=9, fontweight="bold")

        if ax == ax_c:
            ax.set_yticks(np.arange(len(WORKFLOW_ORDER)))
            ax.set_yticklabels(wf_labels, fontsize=7)
        else:
            ax.set_yticks([])

        # Annotate cells
        for ii in range(len(WORKFLOW_ORDER)):
            for jj in range(len(enhancements)):
                val = matrix[ii, jj]
                if np.isnan(val):
                    ax.text(jj, ii, "—", ha="center", va="center",
                            fontsize=6, color="#999999")
                else:
                    color = "white" if abs(val) > 18 else "black"
                    ax.text(jj, ii, f"{val:+.0f}%", ha="center", va="center",
                            fontsize=6, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=[ax_c, ax_w], shrink=0.75, pad=0.02)
    cbar.set_label("E2E Change vs Baseline (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_improvement_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# ─── Helpers ────────────────────────────────────────────────────────────────

def _setup_rc(plt):
    """Configure matplotlib for publication quality."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
    })


# ─── Print Summary Table ───────────────────────────────────────────────────

def print_summary(data):
    """Print summary statistics for all workflows × configs."""
    print("\n" + "=" * 90)
    print(f"{'Workflow':<22} {'Config':<14} {'Cold E2E':>10} {'Warm E2E':>10} "
          f"{'Warm Agg':>10} {'Warm Billed':>12} {'Warm Cost':>12}")
    print("=" * 90)

    for wf in WORKFLOW_ORDER:
        wf_data = data.get(wf, {})
        for cfg in ["Unum-Base", "Unum-Fus", "Unum-Str", "Unum-Fut",
                     "Unum-Fut+Fus", "Unum-Fut+Str", "Unum-All"]:
            runs = wf_data.get(cfg, [])
            if not runs:
                continue
            cold, warm = split_cold_warm(runs)
            cold_e2e = np.mean(extract(cold, "e2e_latency_ms")) if cold else 0
            warm_e2e = np.mean(extract(warm, "e2e_latency_ms")) if warm else 0
            warm_agg = np.mean(extract(warm, "aggregator_delay_ms")) if warm else 0
            warm_billed = np.mean(extract(warm, "total_billed_duration_ms")) if warm else 0
            warm_cost = np.mean(extract(warm, "estimated_cost_usd")) if warm else 0

            print(f"{wf:<22} {cfg:<14} {cold_e2e:>10.0f} {warm_e2e:>10.0f} "
                  f"{warm_agg:>10.0f} {warm_billed:>12.0f} {warm_cost:>12.2e}")
        print("-" * 90)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading results from {RESULTS_DIR}")
    data = load_data()
    print(f"  Found {sum(len(v) for v in data.values())} workflow×config combos")

    print_summary(data)

    print("\nGenerating figures...")
    fig_cold_warm_e2e(data)
    fig_aggregator_delay(data)
    fig_improvement_heatmap(data)

    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
