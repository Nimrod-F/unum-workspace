#!/usr/bin/env python3
"""Generate chart for extended warm-state evaluation across additional workflows."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- data ----------
# Extended warm-state evaluations from earlier per-workflow benchmarks
workflows = [
    'Text\nProcessing',
    'Order\nProcessing',
    'NLP Pipeline\n(Streaming)',
    'Multi-Source\nDashboard',
    'Smart Factory\nIoT',
    'Progressive\nAggregator',
]

classic_warm = [2602, 15900, 7036, 14286, 3295, 4844]
enhanced_warm = [1599, 11390, 5730, 11756, 2916, 4559]
improvement_pct = [
    (c - e) / c * 100 for c, e in zip(classic_warm, enhanced_warm)
]
# [38.5, 28.4, 18.6, 17.7, 11.5, 5.9]

modes = ['Futures', 'Futures', 'Streaming', 'Futures', 'Futures', 'Futures']

# ---------- style ----------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

# LLNCS column width ≈ 12.2 cm
fig_w = 12.2 / 2.54  # inches
fig_h = 3.2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                 gridspec_kw={'width_ratios': [3, 2]})

# --- Left panel: grouped bars (Classic vs Enhanced warm E2E) ---
x = np.arange(len(workflows))
bar_w = 0.35
c_base = '#4575b4'
c_enh = '#d73027'

bars1 = ax1.bar(x - bar_w/2, classic_warm, bar_w, label='Baseline',
                color=c_base, edgecolor='white', linewidth=0.5)
bars2 = ax1.bar(x + bar_w/2, enhanced_warm, bar_w, label='Enhanced',
                color=c_enh, edgecolor='white', linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(workflows, fontsize=7)
ax1.set_ylabel('Warm E2E Latency (ms)')
ax1.set_title('(a) Warm-state latency comparison')
ax1.legend(loc='upper left', framealpha=0.9)

ax1.set_ylim(0, max(classic_warm) * 1.15)

# --- Right panel: horizontal bar chart of improvement % ---
colors = ['#d73027' if m == 'Futures' else '#fc8d59' for m in modes]

y = np.arange(len(workflows))
bars_h = ax2.barh(y, improvement_pct, color=colors, edgecolor='white', linewidth=0.5)

ax2.set_yticks(y)
ax2.set_yticklabels(workflows, fontsize=7)
ax2.set_xlabel('Warm E2E Improvement (%)')
ax2.set_title('(b) Improvement by workflow')
ax2.invert_yaxis()

ax2.set_xlim(0, max(improvement_pct) * 1.15)

# Legend for enhancement type
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#d73027', label='Futures'),
                   Patch(facecolor='#fc8d59', label='Streaming')]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=7)

plt.tight_layout(pad=0.5)

out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
os.makedirs(out_dir, exist_ok=True)
for ext in ('pdf', 'png'):
    path = os.path.join(out_dir, f'fig_extended_warm.{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved {path}')

plt.close()
print('Done.')
