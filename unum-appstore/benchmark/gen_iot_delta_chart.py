#!/usr/bin/env python3
"""Bar chart: Smart Factory IoT — Δ% vs Base for Futures and Fusion."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

metrics = ["Cold E2E", "Warm E2E", "Cost"]
futures_pct = [-22.2, -11.5, +43.8]
fusion_pct  = [-51.2, -20.1, -42.4]

x = np.arange(len(metrics))
width = 0.32

fig, ax = plt.subplots(figsize=(5.5, 3.2))

bars_fut = ax.bar(x - width/2, futures_pct, width, label="Futures",
                  color="#377eb8", edgecolor="black", linewidth=0.5, alpha=0.88)
bars_fus = ax.bar(x + width/2, fusion_pct, width, label="Fusion",
                  color="#e41a1c", edgecolor="black", linewidth=0.5, alpha=0.88)

# Labels on bars
for bar, val in zip(bars_fut, futures_pct):
    y = bar.get_height()
    offset = -1.8 if y < 0 else 1.2
    ax.text(bar.get_x() + bar.get_width()/2, y + offset,
            f"{val:+.1f}%", ha="center", va="bottom" if y >= 0 else "top",
            fontsize=8, fontweight="bold", color="#377eb8")

for bar, val in zip(bars_fus, fusion_pct):
    y = bar.get_height()
    offset = -1.8 if y < 0 else 1.2
    ax.text(bar.get_x() + bar.get_width()/2, y + offset,
            f"{val:+.1f}%", ha="center", va="bottom" if y >= 0 else "top",
            fontsize=8, fontweight="bold", color="#e41a1c")

ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Change vs Baseline (%)")
ax.set_title("Smart Factory IoT — Enhancement vs Baseline")
ax.set_ylim(-62, 55)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.4)

plt.tight_layout()
out = r"d:\unum-workspace\figures\fig_iot_delta_pct.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
