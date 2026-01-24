#!/usr/bin/env python3
"""Generate benchmark comparison charts for CLASSIC vs FUTURE_BASED modes."""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create results directory if not exists
os.makedirs('results', exist_ok=True)

# =============================================================================
# UPDATED BENCHMARK DATA (January 16, 2026)
# From actual CloudWatch-validated benchmark runs
# =============================================================================

# Cold Start Data
COLD_DATA = {
    "CLASSIC": {
        "e2e_mean": 7913, "e2e_std": 58.9, "e2e_median": 7931,
        "init_duration": 6373, "cold_rate": 100, "pre_resolved": 0,
        "cost": 0.0000438
    },
    "FUTURE_BASED": {
        "e2e_mean": 6160, "e2e_std": 43.0, "e2e_median": 6153,
        "init_duration": 6371, "cold_rate": 100, "pre_resolved": 3,
        "cost": 0.0000483
    }
}

# Warm Start Data  
WARM_DATA = {
    "CLASSIC": {
        "e2e_mean": 4844, "e2e_std": 65.7, "e2e_median": 4837,
        "cold_rate": 0, "pre_resolved": 0, "cost": 0.0000255
    },
    "FUTURE_BASED": {
        "e2e_mean": 4559, "e2e_std": 25.4, "e2e_median": 4553,
        "cold_rate": 0, "pre_resolved": 2, "cost": 0.0000332
    }
}

modes = ['CLASSIC', 'FUTURE_BASED']

# Legacy variables for backward compatibility
e2e_mean = [COLD_DATA["CLASSIC"]["e2e_mean"], COLD_DATA["FUTURE_BASED"]["e2e_mean"]]
e2e_median = [COLD_DATA["CLASSIC"]["e2e_median"], COLD_DATA["FUTURE_BASED"]["e2e_median"]]
e2e_std = [COLD_DATA["CLASSIC"]["e2e_std"], COLD_DATA["FUTURE_BASED"]["e2e_std"]]
e2e_min = [e2e_mean[0] - 100, e2e_mean[1] - 100]
e2e_max = [e2e_mean[0] + 100, e2e_mean[1] + 100]

# Other metrics
total_duration = [11239.2, 11282.6]
wait_time = [0.0, 17.0]
pre_resolved = [0, 3]
cost_per_run = [COLD_DATA["CLASSIC"]["cost"], COLD_DATA["FUTURE_BASED"]["cost"]]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Progressive-Aggregator Benchmark: CLASSIC vs FUTURE_BASED\n(Cold Start Comparison)', fontsize=14, fontweight='bold')

colors = ['#e74c3c', '#27ae60']  # Red for CLASSIC, Green for FUTURE_BASED

# 1. E2E Latency Bar Chart with Error Bars
ax1 = axes[0, 0]
x = np.arange(len(modes))
bars = ax1.bar(x, e2e_mean, yerr=e2e_std, capsize=5, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Latency (ms)', fontsize=11)
ax1.set_title('End-to-End Latency - Cold Start (Mean ± Std Dev)', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(modes, fontsize=11)
ax1.set_ylim(0, 9000)
for i, (bar, val) in enumerate(zip(bars, e2e_mean)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + e2e_std[i] + 100, f'{val:.0f}ms', ha='center', fontsize=10, fontweight='bold')
# Add improvement annotation
improvement = (e2e_mean[0] - e2e_mean[1]) / e2e_mean[0] * 100
ax1.annotate(f'↓{improvement:.1f}%', xy=(0.5, 7500), fontsize=12, color='#27ae60', fontweight='bold', ha='center')
ax1.grid(axis='y', alpha=0.3)

# 2. Cold vs Warm Comparison
ax2 = axes[0, 1]
x = np.arange(2)
width = 0.35
cold_latencies = [COLD_DATA["CLASSIC"]["e2e_mean"], COLD_DATA["FUTURE_BASED"]["e2e_mean"]]
warm_latencies = [WARM_DATA["CLASSIC"]["e2e_mean"], WARM_DATA["FUTURE_BASED"]["e2e_mean"]]
bars1 = ax2.bar(x - width/2, cold_latencies, width, label='Cold Start', color='#3498db', edgecolor='black')
bars2 = ax2.bar(x + width/2, warm_latencies, width, label='Warm Start', color='#f39c12', edgecolor='black')
ax2.set_ylabel('Latency (ms)', fontsize=11)
ax2.set_title('Cold vs Warm Start Impact', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(modes, fontsize=11)
ax2.legend(loc='upper right')
ax2.set_ylim(0, 9000)
for bar in bars1 + bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{bar.get_height():.0f}', ha='center', fontsize=9, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Background Polling Efficiency - Cold vs Warm
ax3 = axes[1, 0]
bar_width = 0.25
x = np.arange(2)
# Cold: CLASSIC=0, FUTURE=3 pre-resolved
# Warm: CLASSIC=0, FUTURE=2 pre-resolved
cold_pre = [0, COLD_DATA["FUTURE_BASED"]["pre_resolved"]]
warm_pre = [0, WARM_DATA["FUTURE_BASED"]["pre_resolved"]]
cold_wait = [5, 5 - COLD_DATA["FUTURE_BASED"]["pre_resolved"]]
warm_wait = [5, 5 - WARM_DATA["FUTURE_BASED"]["pre_resolved"]]

# Stacked bars
bars1_cold = ax3.bar(x - bar_width/2, cold_pre, bar_width, label='Pre-Resolved (Cold)', color='#27ae60', edgecolor='black')
ax3.bar(x - bar_width/2, cold_wait, bar_width, bottom=cold_pre, label='Wait Required (Cold)', color='#2c3e50', edgecolor='black', alpha=0.7)
bars1_warm = ax3.bar(x + bar_width/2, warm_pre, bar_width, label='Pre-Resolved (Warm)', color='#2ecc71', edgecolor='black')
ax3.bar(x + bar_width/2, warm_wait, bar_width, bottom=warm_pre, label='Wait Required (Warm)', color='#95a5a6', edgecolor='black', alpha=0.7)

ax3.set_ylabel('Number of Inputs', fontsize=11)
ax3.set_title('Fan-In Input Resolution (out of 5)', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(modes, fontsize=11)
ax3.legend(loc='upper right', fontsize=8)
ax3.set_ylim(0, 6)
ax3.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
ax3.grid(axis='y', alpha=0.3)

# 4. Cost and Consistency Summary
ax4 = axes[1, 1]
metrics = ['E2E Cold\n(ms)', 'E2E Warm\n(ms)', 'Std Dev Cold\n(ms)', 'Std Dev Warm\n(ms)']
classic_values = [
    COLD_DATA["CLASSIC"]["e2e_mean"] / 100,  # Scale for visualization
    WARM_DATA["CLASSIC"]["e2e_mean"] / 100,
    COLD_DATA["CLASSIC"]["e2e_std"],
    WARM_DATA["CLASSIC"]["e2e_std"]
]
future_values = [
    COLD_DATA["FUTURE_BASED"]["e2e_mean"] / 100,
    WARM_DATA["FUTURE_BASED"]["e2e_mean"] / 100,
    COLD_DATA["FUTURE_BASED"]["e2e_std"],
    WARM_DATA["FUTURE_BASED"]["e2e_std"]
]

x = np.arange(len(metrics))
bar_width = 0.35
bars1 = ax4.bar(x - bar_width/2, classic_values, bar_width, label='CLASSIC', color='#e74c3c', edgecolor='black')
bars2 = ax4.bar(x + bar_width/2, future_values, bar_width, label='FUTURE_BASED', color='#27ae60', edgecolor='black')
ax4.set_ylabel('Value (scaled)', fontsize=11)
ax4.set_title('Performance Metrics Comparison', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, fontsize=9)
ax4.legend(loc='upper right')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/benchmark_comparison_chart.png', dpi=150, bbox_inches='tight')
print('Chart saved to results/benchmark_comparison_chart.png')

# Also create individual charts
# Chart 1: E2E Latency Comparison - All 4 scenarios
fig2, ax = plt.subplots(figsize=(10, 6))
scenarios = ['CLASSIC\nCold', 'FUTURE\nCold', 'CLASSIC\nWarm', 'FUTURE\nWarm']
latencies = [
    COLD_DATA["CLASSIC"]["e2e_mean"],
    COLD_DATA["FUTURE_BASED"]["e2e_mean"],
    WARM_DATA["CLASSIC"]["e2e_mean"],
    WARM_DATA["FUTURE_BASED"]["e2e_mean"]
]
std_devs = [
    COLD_DATA["CLASSIC"]["e2e_std"],
    COLD_DATA["FUTURE_BASED"]["e2e_std"],
    WARM_DATA["CLASSIC"]["e2e_std"],
    WARM_DATA["FUTURE_BASED"]["e2e_std"]
]
bar_colors = ['#e74c3c', '#27ae60', '#e74c3c', '#27ae60']

bars = ax.bar(scenarios, latencies, yerr=std_devs, capsize=8, color=bar_colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('End-to-End Latency (ms)', fontsize=12)
ax.set_title('E2E Latency: CLASSIC vs FUTURE_BASED\n(Cold and Warm Start)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 9000)
for i, (bar, val, std) in enumerate(zip(bars, latencies, std_devs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 100, 
            f'{val:.0f}ms', ha='center', fontsize=10, fontweight='bold')
# Add improvement annotations
cold_imp = (COLD_DATA["CLASSIC"]["e2e_mean"] - COLD_DATA["FUTURE_BASED"]["e2e_mean"]) / COLD_DATA["CLASSIC"]["e2e_mean"] * 100
warm_imp = (WARM_DATA["CLASSIC"]["e2e_mean"] - WARM_DATA["FUTURE_BASED"]["e2e_mean"]) / WARM_DATA["CLASSIC"]["e2e_mean"] * 100
ax.annotate(f'↓{cold_imp:.1f}%', xy=(0.5, 7200), fontsize=11, color='#27ae60', fontweight='bold', ha='center')
ax.annotate(f'↓{warm_imp:.1f}%', xy=(2.5, 5200), fontsize=11, color='#27ae60', fontweight='bold', ha='center')
# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', edgecolor='black', label='CLASSIC'),
                   Patch(facecolor='#27ae60', edgecolor='black', label='FUTURE_BASED')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/e2e_latency_comparison.png', dpi=150, bbox_inches='tight')
print('Chart saved to results/e2e_latency_comparison.png')

# Chart 2: Background Polling Efficiency - Cold vs Warm
fig3, ax = plt.subplots(figsize=(10, 6))
scenarios = ['CLASSIC\nCold', 'FUTURE\nCold', 'CLASSIC\nWarm', 'FUTURE\nWarm']
pre_resolved = [0, COLD_DATA["FUTURE_BASED"]["pre_resolved"], 0, WARM_DATA["FUTURE_BASED"]["pre_resolved"]]
wait_required = [5, 5 - COLD_DATA["FUTURE_BASED"]["pre_resolved"], 5, 5 - WARM_DATA["FUTURE_BASED"]["pre_resolved"]]

bars1 = ax.bar(scenarios, pre_resolved, label='Pre-Resolved (Background Polling)', color='#27ae60', edgecolor='black')
bars2 = ax.bar(scenarios, wait_required, bottom=pre_resolved, label='Required Waiting', color='#95a5a6', edgecolor='black')
ax.set_ylabel('Number of Inputs (out of 5)', fontsize=12)
ax.set_title('Fan-In Input Resolution Strategy\nBackground Polling Effectiveness', fontsize=14, fontweight='bold')
ax.set_ylim(0, 6)
ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
ax.text(3.5, 5.1, 'Total Inputs', fontsize=9, color='gray')
for i, (bar, val) in enumerate(zip(bars1, pre_resolved)):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, val/2, f'{val}', ha='center', fontsize=11, fontweight='bold', color='white')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/background_polling_efficiency.png', dpi=150, bbox_inches='tight')
print('Chart saved to results/background_polling_efficiency.png')

# Chart 3: Cold Start Impact (NEW)
fig4, ax = plt.subplots(figsize=(10, 6))
x = np.arange(2)
width = 0.35
cold = [COLD_DATA["CLASSIC"]["e2e_mean"], COLD_DATA["FUTURE_BASED"]["e2e_mean"]]
warm = [WARM_DATA["CLASSIC"]["e2e_mean"], WARM_DATA["FUTURE_BASED"]["e2e_mean"]]

bars1 = ax.bar(x - width/2, cold, width, label='Cold Start', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, warm, width, label='Warm Start', color='#f39c12', edgecolor='black')

ax.set_ylabel('E2E Latency (ms)', fontsize=12)
ax.set_title('Cold Start Impact by Execution Mode', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(modes, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 9000)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars1 + bars2:
    ax.annotate(f'{bar.get_height():.0f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add cold->warm improvement annotations
classic_imp = (COLD_DATA["CLASSIC"]["e2e_mean"] - WARM_DATA["CLASSIC"]["e2e_mean"]) / COLD_DATA["CLASSIC"]["e2e_mean"] * 100
future_imp = (COLD_DATA["FUTURE_BASED"]["e2e_mean"] - WARM_DATA["FUTURE_BASED"]["e2e_mean"]) / COLD_DATA["FUTURE_BASED"]["e2e_mean"] * 100
ax.annotate(f'↓{classic_imp:.1f}%', xy=(0, 6500), fontsize=11, color='#2c3e50', fontweight='bold', ha='center')
ax.annotate(f'↓{future_imp:.1f}%', xy=(1, 5500), fontsize=11, color='#2c3e50', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('results/cold_vs_warm_impact.png', dpi=150, bbox_inches='tight')
print('Chart saved to results/cold_vs_warm_impact.png')

# Chart 4: Consistency Comparison (NEW)
fig5, ax = plt.subplots(figsize=(10, 6))
scenarios = ['CLASSIC\nCold', 'FUTURE\nCold', 'CLASSIC\nWarm', 'FUTURE\nWarm']
std_devs = [
    COLD_DATA["CLASSIC"]["e2e_std"],
    COLD_DATA["FUTURE_BASED"]["e2e_std"],
    WARM_DATA["CLASSIC"]["e2e_std"],
    WARM_DATA["FUTURE_BASED"]["e2e_std"]
]
bar_colors = ['#e74c3c', '#27ae60', '#e74c3c', '#27ae60']

bars = ax.bar(scenarios, std_devs, color=bar_colors, edgecolor='black')
ax.set_ylabel('Standard Deviation (ms)', fontsize=12)
ax.set_title('Consistency Comparison (Lower = More Consistent)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, std_devs):
    ax.annotate(f'{val:.1f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Add legend
legend_elements = [Patch(facecolor='#e74c3c', edgecolor='black', label='CLASSIC'),
                   Patch(facecolor='#27ae60', edgecolor='black', label='FUTURE_BASED')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('results/consistency_comparison.png', dpi=150, bbox_inches='tight')
print('Chart saved to results/consistency_comparison.png')

# Chart 5: Cost Comparison (NEW)
fig6, ax = plt.subplots(figsize=(10, 6))
scenarios = ['CLASSIC\nCold', 'FUTURE\nCold', 'CLASSIC\nWarm', 'FUTURE\nWarm']
costs = [
    COLD_DATA["CLASSIC"]["cost"] * 1e5,
    COLD_DATA["FUTURE_BASED"]["cost"] * 1e5,
    WARM_DATA["CLASSIC"]["cost"] * 1e5,
    WARM_DATA["FUTURE_BASED"]["cost"] * 1e5
]
bar_colors = ['#e74c3c', '#27ae60', '#e74c3c', '#27ae60']

bars = ax.bar(scenarios, costs, color=bar_colors, edgecolor='black')
ax.set_ylabel('Cost ($ × 10⁻⁵)', fontsize=12)
ax.set_title('Cost per Run Comparison', fontsize=14, fontweight='bold')
for bar, val in zip(bars, costs):
    ax.annotate(f'${val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5)

legend_elements = [Patch(facecolor='#e74c3c', edgecolor='black', label='CLASSIC'),
                   Patch(facecolor='#27ae60', edgecolor='black', label='FUTURE_BASED')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('results/cost_comparison.png', dpi=150, bbox_inches='tight')
print('Chart saved to results/cost_comparison.png')

print('\nAll charts generated successfully!')
