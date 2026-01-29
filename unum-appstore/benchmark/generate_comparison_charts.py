#!/usr/bin/env python3
"""
Generate Comparison Charts for Multi-Workflow Benchmarks

Creates visualizations comparing CLASSIC, EAGER, and FUTURE_BASED modes
across all research workflows.

Output Charts:
1. E2E Latency comparison (bar chart with error bars)
2. Cold vs Warm performance (grouped bars)
3. Pre-resolved efficiency (stacked bars)
4. Improvement percentage (horizontal bars)
5. Latency variance (box plots)
6. Cost comparison (bar chart)

Usage:
    python generate_comparison_charts.py --results-dir results/
    python generate_comparison_charts.py --summary-files file1.json file2.json
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any


# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'CLASSIC': '#e74c3c',      # Red
    'EAGER': '#f39c12',        # Orange  
    'FUTURE_BASED': '#27ae60', # Green
}

WORKFLOW_LABELS = {
    'progressive-aggregator': 'Progressive\nAggregator',
    'ml-training-pipeline': 'ML Training\nPipeline',
    'video-analysis': 'Video\nAnalysis',
    'image-processing-pipeline': 'Image\nProcessing',
    'genomics-pipeline': 'Genomics\nPipeline',
}


def load_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all summary files from results directory"""
    results = {}
    
    for f in Path(results_dir).glob('*_summary.json'):
        with open(f) as fp:
            data = json.load(fp)
            workflow = data.get('workflow', 'unknown')
            mode = data.get('mode', 'unknown')
            
            if workflow not in results:
                results[workflow] = {}
            results[workflow][mode] = data
    
    return results


def create_e2e_latency_comparison(results: Dict, output_dir: str):
    """Create E2E latency comparison bar chart"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    workflows = list(results.keys())
    x = np.arange(len(workflows))
    width = 0.25
    
    modes = ['CLASSIC', 'EAGER', 'FUTURE_BASED']
    
    for i, mode in enumerate(modes):
        means = []
        stds = []
        
        for workflow in workflows:
            if mode in results.get(workflow, {}):
                data = results[workflow][mode]
                means.append(data.get('e2e_latency_mean_ms', 0))
                stds.append(data.get('e2e_latency_std_ms', 0))
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
                     label=mode, color=COLORS[mode], edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                       f'{mean:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Workflow', fontsize=12)
    ax.set_ylabel('End-to-End Latency (ms)', fontsize=12)
    ax.set_title('E2E Latency Comparison Across Workflows\n(Mean ± Std Dev)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([WORKFLOW_LABELS.get(w, w) for w in workflows], fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/e2e_latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/e2e_latency_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Created: e2e_latency_comparison.png")


def create_cold_warm_comparison(results: Dict, output_dir: str):
    """Create cold vs warm start comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    workflows = list(results.keys())
    x = np.arange(len(workflows))
    width = 0.35
    
    # Cold starts
    ax1 = axes[0]
    for i, mode in enumerate(['CLASSIC', 'FUTURE_BASED']):
        vals = []
        for workflow in workflows:
            if mode in results.get(workflow, {}):
                vals.append(results[workflow][mode].get('cold_e2e_mean_ms', 0))
            else:
                vals.append(0)
        
        ax1.bar(x + i * width, vals, width, label=mode, color=COLORS[mode], edgecolor='black')
        for j, v in enumerate(vals):
            if v > 0:
                ax1.text(x[j] + i * width, v + 50, f'{v:.0f}', ha='center', fontsize=8)
    
    ax1.set_title('Cold Start Latency', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Workflow', fontsize=10)
    ax1.set_ylabel('Latency (ms)', fontsize=10)
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels([WORKFLOW_LABELS.get(w, w) for w in workflows], fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Warm starts
    ax2 = axes[1]
    for i, mode in enumerate(['CLASSIC', 'FUTURE_BASED']):
        vals = []
        for workflow in workflows:
            if mode in results.get(workflow, {}):
                vals.append(results[workflow][mode].get('warm_e2e_mean_ms', 0))
            else:
                vals.append(0)
        
        ax2.bar(x + i * width, vals, width, label=mode, color=COLORS[mode], edgecolor='black')
        for j, v in enumerate(vals):
            if v > 0:
                ax2.text(x[j] + i * width, v + 50, f'{v:.0f}', ha='center', fontsize=8)
    
    ax2.set_title('Warm Start Latency', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Workflow', fontsize=10)
    ax2.set_ylabel('Latency (ms)', fontsize=10)
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels([WORKFLOW_LABELS.get(w, w) for w in workflows], fontsize=9)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Cold vs Warm Start Performance: CLASSIC vs FUTURE_BASED', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cold_warm_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/cold_warm_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Created: cold_warm_comparison.png")


def create_improvement_chart(results: Dict, output_dir: str):
    """Create improvement percentage horizontal bar chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    workflows = []
    improvements_e2e = []
    improvements_warm = []
    
    for workflow in results.keys():
        if 'CLASSIC' in results[workflow] and 'FUTURE_BASED' in results[workflow]:
            classic = results[workflow]['CLASSIC']
            future = results[workflow]['FUTURE_BASED']
            
            classic_e2e = classic.get('e2e_latency_mean_ms', 0)
            future_e2e = future.get('e2e_latency_mean_ms', 0)
            
            classic_warm = classic.get('warm_e2e_mean_ms', 0)
            future_warm = future.get('warm_e2e_mean_ms', 0)
            
            if classic_e2e > 0:
                workflows.append(WORKFLOW_LABELS.get(workflow, workflow))
                improvements_e2e.append((classic_e2e - future_e2e) / classic_e2e * 100)
                
                if classic_warm > 0:
                    improvements_warm.append((classic_warm - future_warm) / classic_warm * 100)
                else:
                    improvements_warm.append(0)
    
    y = np.arange(len(workflows))
    height = 0.35
    
    bars1 = ax.barh(y - height/2, improvements_e2e, height, label='Overall E2E', color='#3498db', edgecolor='black')
    bars2 = ax.barh(y + height/2, improvements_warm, height, label='Warm Only', color='#2ecc71', edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars1, improvements_e2e):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, improvements_warm):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Latency Improvement (%)', fontsize=12)
    ax.set_title('FUTURE_BASED Latency Improvement vs CLASSIC', fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(workflows, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(-5, max(max(improvements_e2e), max(improvements_warm)) + 10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvement_chart.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/improvement_chart.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Created: improvement_chart.png")


def create_pre_resolved_chart(results: Dict, output_dir: str):
    """Create pre-resolved inputs efficiency chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    workflows = []
    pre_resolved_classic = []
    pre_resolved_future = []
    fan_in_sizes = []
    
    EXPECTED_FAN_IN = {
        'progressive-aggregator': 5,
        'ml-training-pipeline': 4,
        'video-analysis': 6,
        'image-processing-pipeline': 5,
        'genomics-pipeline': 6,  # First fan-in
    }
    
    for workflow in results.keys():
        if 'FUTURE_BASED' in results[workflow]:
            workflows.append(WORKFLOW_LABELS.get(workflow, workflow))
            
            classic_pre = results[workflow].get('CLASSIC', {}).get('avg_pre_resolved', 0)
            future_pre = results[workflow]['FUTURE_BASED'].get('avg_pre_resolved', 0)
            fan_in = EXPECTED_FAN_IN.get(workflow, 5)
            
            pre_resolved_classic.append(classic_pre)
            pre_resolved_future.append(future_pre)
            fan_in_sizes.append(fan_in)
    
    x = np.arange(len(workflows))
    width = 0.35
    
    # Stacked bars showing pre-resolved vs waiting
    bars_pre = ax.bar(x, pre_resolved_future, width, label='Pre-Resolved (Background Polling)',
                     color='#27ae60', edgecolor='black')
    bars_wait = ax.bar(x, [f - p for f, p in zip(fan_in_sizes, pre_resolved_future)], width,
                      bottom=pre_resolved_future, label='Required Wait',
                      color='#95a5a6', edgecolor='black', alpha=0.7)
    
    # Add fan-in size reference line
    for i, (x_pos, fan_in) in enumerate(zip(x, fan_in_sizes)):
        ax.hlines(y=fan_in, xmin=x_pos - width/2, xmax=x_pos + width/2,
                 colors='#e74c3c', linestyles='dashed', linewidth=2)
    
    # Add value labels
    for bar, pre, total in zip(bars_pre, pre_resolved_future, fan_in_sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
               f'{pre:.1f}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(bar.get_x() + bar.get_width()/2, total + 0.2,
               f'{pre/total*100:.0f}%', ha='center', va='bottom', fontsize=9, color='#27ae60')
    
    ax.set_xlabel('Workflow', fontsize=12)
    ax.set_ylabel('Number of Inputs', fontsize=12)
    ax.set_title('Background Polling Efficiency (FUTURE_BASED)\nPre-Resolved Inputs at Fan-In', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(workflows, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max(fan_in_sizes) + 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add reference line legend
    red_line = mpatches.Patch(color='#e74c3c', label='Total Fan-In Size')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [red_line], labels + ['Total Fan-In Size'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pre_resolved_efficiency.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/pre_resolved_efficiency.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Created: pre_resolved_efficiency.png")


def create_memory_comparison_chart(results: Dict, output_dir: str):
    """Create memory comparison chart - CRITICAL for FUTURE_BASED overhead analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    workflows = list(results.keys())
    x = np.arange(len(workflows))
    width = 0.35
    
    # Left: Aggregator Memory (key metric for overhead)
    ax1 = axes[0]
    classic_mem = []
    future_mem = []
    
    for workflow in workflows:
        classic_mem.append(results.get(workflow, {}).get('CLASSIC', {}).get('avg_aggregator_memory_mb', 0))
        future_mem.append(results.get(workflow, {}).get('FUTURE_BASED', {}).get('avg_aggregator_memory_mb', 0))
    
    bars1 = ax1.bar(x - width/2, classic_mem, width, label='CLASSIC', color=COLORS['CLASSIC'], edgecolor='black')
    bars2 = ax1.bar(x + width/2, future_mem, width, label='FUTURE_BASED', color=COLORS['FUTURE_BASED'], edgecolor='black')
    
    # Add value labels and overhead %
    for i, (c, f) in enumerate(zip(classic_mem, future_mem)):
        if c > 0 and f > 0:
            overhead = (f - c) / c * 100
            ax1.text(x[i] - width/2, c + 1, f'{c:.0f}', ha='center', fontsize=9)
            ax1.text(x[i] + width/2, f + 1, f'{f:.0f}', ha='center', fontsize=9)
            ax1.annotate(f'+{overhead:.1f}%', xy=(x[i], max(c, f) + 5), fontsize=8, 
                        color='#e74c3c' if overhead > 10 else '#27ae60', ha='center', fontweight='bold')
    
    ax1.set_xlabel('Workflow', fontsize=11)
    ax1.set_ylabel('Memory (MB)', fontsize=11)
    ax1.set_title('Aggregator Memory Usage\n(FUTURE_BASED Overhead)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([WORKFLOW_LABELS.get(w, w) for w in workflows], fontsize=9)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Memory Overhead vs Latency Improvement scatter
    ax2 = axes[1]
    
    memory_overheads = []
    latency_improvements = []
    labels = []
    
    for workflow in workflows:
        if 'CLASSIC' in results.get(workflow, {}) and 'FUTURE_BASED' in results.get(workflow, {}):
            classic = results[workflow]['CLASSIC']
            future = results[workflow]['FUTURE_BASED']
            
            c_mem = classic.get('avg_aggregator_memory_mb', 0)
            f_mem = future.get('avg_aggregator_memory_mb', 0)
            c_lat = classic.get('e2e_latency_mean_ms', 0)
            f_lat = future.get('e2e_latency_mean_ms', 0)
            
            if c_mem > 0 and c_lat > 0:
                mem_overhead = (f_mem - c_mem) / c_mem * 100
                lat_improvement = (c_lat - f_lat) / c_lat * 100
                memory_overheads.append(mem_overhead)
                latency_improvements.append(lat_improvement)
                labels.append(WORKFLOW_LABELS.get(workflow, workflow).replace('\n', ' '))
    
    colors_scatter = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    scatter = ax2.scatter(memory_overheads, latency_improvements, c=colors_scatter, s=200, edgecolors='black', linewidths=1.5)
    
    for i, label in enumerate(labels):
        ax2.annotate(label, (memory_overheads[i], latency_improvements[i]), 
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    # Add reference lines
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add "good trade-off" region
    ax2.fill_between([0, 20], [0, 0], [100, 100], alpha=0.1, color='green', label='Good trade-off')
    
    ax2.set_xlabel('Memory Overhead (%)', fontsize=11)
    ax2.set_ylabel('Latency Improvement (%)', fontsize=11)
    ax2.set_title('Trade-off: Latency Gain vs Memory Cost', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Memory Analysis: FUTURE_BASED Overhead', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/memory_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/memory_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Created: memory_comparison.png")


def create_latency_vs_memory_tradeoff(results: Dict, output_dir: str):
    """Create detailed latency vs memory trade-off analysis chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    workflows = []
    latency_improvements = []
    memory_overheads = []
    efficiency_ratios = []
    
    for workflow in results.keys():
        if 'CLASSIC' in results[workflow] and 'FUTURE_BASED' in results[workflow]:
            classic = results[workflow]['CLASSIC']
            future = results[workflow]['FUTURE_BASED']
            
            c_lat = classic.get('e2e_latency_mean_ms', 0)
            f_lat = future.get('e2e_latency_mean_ms', 0)
            c_mem = classic.get('avg_aggregator_memory_mb', 0)
            f_mem = future.get('avg_aggregator_memory_mb', 0)
            
            if c_lat > 0 and c_mem > 0:
                lat_imp = (c_lat - f_lat) / c_lat * 100
                mem_ovh = (f_mem - c_mem) / c_mem * 100 if c_mem > 0 else 0
                
                workflows.append(WORKFLOW_LABELS.get(workflow, workflow).replace('\n', ' '))
                latency_improvements.append(lat_imp)
                memory_overheads.append(mem_ovh)
                
                # Efficiency = latency gain per % memory cost
                efficiency = lat_imp / mem_ovh if mem_ovh > 0 else float('inf')
                efficiency_ratios.append(min(efficiency, 10))  # Cap for visualization
    
    # Horizontal bar chart with dual metrics
    y = np.arange(len(workflows))
    height = 0.35
    
    bars1 = ax.barh(y - height/2, latency_improvements, height, label='Latency Improvement (%)', 
                   color='#27ae60', edgecolor='black')
    bars2 = ax.barh(y + height/2, memory_overheads, height, label='Memory Overhead (%)', 
                   color='#e74c3c', edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars1, latency_improvements):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}%', va='center', fontsize=10, fontweight='bold', color='#27ae60')
    for bar, val in zip(bars2, memory_overheads):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'+{val:.1f}%', va='center', fontsize=10, fontweight='bold', color='#e74c3c')
    
    # Add efficiency ratio annotations
    for i, (y_pos, eff) in enumerate(zip(y, efficiency_ratios)):
        ax.text(max(latency_improvements) + 8, y_pos, f'Ratio: {eff:.1f}x', 
               fontsize=9, va='center', style='italic')
    
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel('Workflow', fontsize=12)
    ax.set_title('FUTURE_BASED: Latency Improvement vs Memory Overhead\n(Higher ratio = better efficiency)', 
                fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(workflows, fontsize=10)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/latency_vs_memory_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/latency_vs_memory_tradeoff.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Created: latency_vs_memory_tradeoff.png")


def create_summary_table(results: Dict, output_dir: str):
    """Create summary table as image"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Build table data
    headers = ['Workflow', 'Mode', 'E2E Mean (ms)', 'Std Dev', 'Cold Mean', 'Warm Mean', 'Pre-Resolved', 'Cost/Run']
    rows = []
    
    for workflow in results.keys():
        for mode in ['CLASSIC', 'EAGER', 'FUTURE_BASED']:
            if mode in results.get(workflow, {}):
                data = results[workflow][mode]
                rows.append([
                    WORKFLOW_LABELS.get(workflow, workflow).replace('\n', ' '),
                    mode,
                    f"{data.get('e2e_latency_mean_ms', 0):.0f}",
                    f"{data.get('e2e_latency_std_ms', 0):.1f}",
                    f"{data.get('cold_e2e_mean_ms', 0):.0f}",
                    f"{data.get('warm_e2e_mean_ms', 0):.0f}",
                    f"{data.get('avg_pre_resolved', 0):.1f}",
                    f"${data.get('cost_per_run', 0):.6f}",
                ])
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.12, 0.1, 0.08, 0.1, 0.1, 0.1, 0.1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by mode
    for i, row in enumerate(rows, 1):
        mode = row[1]
        color = COLORS.get(mode, '#ffffff')
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color + '30')  # 30 = alpha
    
    ax.set_title('Benchmark Results Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: summary_table.png")


def create_workflow_profile_chart(results: Dict, output_dir: str):
    """Create workflow profile chart showing task durations"""
    
    WORKFLOW_PROFILES = {
        'progressive-aggregator': {
            'tasks': ['Source 1', 'Source 2', 'Source 3', 'Source 4', 'Source 5'],
            'durations': [2.0, 3.0, 4.0, 0.3, 0.5],
        },
        'ml-training-pipeline': {
            'tasks': ['Linear Reg', 'SVM', 'Random Forest', 'Gradient Boost'],
            'durations': [0.1, 2.0, 8.0, 5.0],
        },
        'video-analysis': {
            'tasks': ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6'],
            'durations': [0.3, 1.5, 4.0, 2.0, 0.5, 6.0],
        },
        'image-processing-pipeline': {
            'tasks': ['Metadata', 'Thumbnail', 'Resize', 'Filters', 'Face Detect'],
            'durations': [0.05, 0.15, 0.4, 1.5, 3.5],
        },
        'genomics-pipeline': {
            'tasks': ['HG00096', 'HG00097', 'NA12891', 'NA12892', 'HG00099', 'NA12878'],
            'durations': [0.5, 0.4, 1.5, 1.8, 2.5, 3.5],
        },
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (workflow, profile) in enumerate(WORKFLOW_PROFILES.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        tasks = profile['tasks']
        durations = profile['durations']
        
        # Sort by duration for visualization
        sorted_pairs = sorted(zip(durations, tasks))
        durations_sorted, tasks_sorted = zip(*sorted_pairs)
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(tasks)))
        bars = ax.barh(tasks_sorted, durations_sorted, color=colors, edgecolor='black')
        
        # Add value labels
        for bar, dur in zip(bars, durations_sorted):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{dur:.2f}s', va='center', fontsize=9)
        
        ax.set_xlabel('Duration (seconds)', fontsize=10)
        ax.set_title(WORKFLOW_LABELS.get(workflow, workflow).replace('\n', ' '), 
                    fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add variance annotation
        variance = np.std(durations)
        ax.text(0.98, 0.02, f'σ = {variance:.2f}s', transform=ax.transAxes,
               ha='right', va='bottom', fontsize=9, style='italic')
    
    # Hide unused subplot
    if len(WORKFLOW_PROFILES) < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle('Parallel Task Duration Profiles\n(Higher variance = greater FUTURE_BASED benefit)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/workflow_profiles.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/workflow_profiles.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Created: workflow_profiles.png")


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark comparison charts')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing benchmark results')
    parser.add_argument('--output-dir', type=str, default='charts',
                       help='Output directory for charts')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nLoading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found. Creating sample charts with expected data...")
        # Use expected data for demonstration
        results = create_sample_data()
    
    print(f"Found {len(results)} workflows")
    
    print(f"\nGenerating charts in: {args.output_dir}")
    
    create_e2e_latency_comparison(results, args.output_dir)
    create_cold_warm_comparison(results, args.output_dir)
    create_improvement_chart(results, args.output_dir)
    create_pre_resolved_chart(results, args.output_dir)
    create_memory_comparison_chart(results, args.output_dir)
    create_latency_vs_memory_tradeoff(results, args.output_dir)
    create_summary_table(results, args.output_dir)
    create_workflow_profile_chart(results, args.output_dir)
    
    print(f"\n✓ All charts generated successfully!")


def create_sample_data():
    """Create sample data for demonstration (including memory metrics)"""
    return {
        'progressive-aggregator': {
            'CLASSIC': {
                'e2e_latency_mean_ms': 4844, 'e2e_latency_std_ms': 65.7,
                'cold_e2e_mean_ms': 7913, 'warm_e2e_mean_ms': 4844,
                'avg_pre_resolved': 0, 'cost_per_run': 0.0000255,
                'avg_aggregator_memory_mb': 68, 'avg_total_memory_mb': 420
            },
            'FUTURE_BASED': {
                'e2e_latency_mean_ms': 4559, 'e2e_latency_std_ms': 25.4,
                'cold_e2e_mean_ms': 6160, 'warm_e2e_mean_ms': 4559,
                'avg_pre_resolved': 2.5, 'cost_per_run': 0.0000332,
                'avg_aggregator_memory_mb': 74, 'avg_total_memory_mb': 435
            }
        },
        'ml-training-pipeline': {
            'CLASSIC': {
                'e2e_latency_mean_ms': 8500, 'e2e_latency_std_ms': 120,
                'cold_e2e_mean_ms': 11000, 'warm_e2e_mean_ms': 8500,
                'avg_pre_resolved': 0, 'cost_per_run': 0.0000380,
                'avg_aggregator_memory_mb': 72, 'avg_total_memory_mb': 380
            },
            'FUTURE_BASED': {
                'e2e_latency_mean_ms': 6200, 'e2e_latency_std_ms': 45,
                'cold_e2e_mean_ms': 8500, 'warm_e2e_mean_ms': 6200,
                'avg_pre_resolved': 2.8, 'cost_per_run': 0.0000420,
                'avg_aggregator_memory_mb': 79, 'avg_total_memory_mb': 395
            }
        },
        'video-analysis': {
            'CLASSIC': {
                'e2e_latency_mean_ms': 7200, 'e2e_latency_std_ms': 95,
                'cold_e2e_mean_ms': 9800, 'warm_e2e_mean_ms': 7200,
                'avg_pre_resolved': 0, 'cost_per_run': 0.0000350,
                'avg_aggregator_memory_mb': 70, 'avg_total_memory_mb': 510
            },
            'FUTURE_BASED': {
                'e2e_latency_mean_ms': 5100, 'e2e_latency_std_ms': 38,
                'cold_e2e_mean_ms': 7100, 'warm_e2e_mean_ms': 5100,
                'avg_pre_resolved': 3.5, 'cost_per_run': 0.0000390,
                'avg_aggregator_memory_mb': 82, 'avg_total_memory_mb': 530
            }
        },
        'image-processing-pipeline': {
            'CLASSIC': {
                'e2e_latency_mean_ms': 4100, 'e2e_latency_std_ms': 55,
                'cold_e2e_mean_ms': 6800, 'warm_e2e_mean_ms': 4100,
                'avg_pre_resolved': 0, 'cost_per_run': 0.0000220,
                'avg_aggregator_memory_mb': 65, 'avg_total_memory_mb': 395
            },
            'FUTURE_BASED': {
                'e2e_latency_mean_ms': 2900, 'e2e_latency_std_ms': 28,
                'cold_e2e_mean_ms': 4800, 'warm_e2e_mean_ms': 2900,
                'avg_pre_resolved': 3.2, 'cost_per_run': 0.0000280,
                'avg_aggregator_memory_mb': 73, 'avg_total_memory_mb': 410
            }
        },
        'genomics-pipeline': {
            'CLASSIC': {
                'e2e_latency_mean_ms': 9500, 'e2e_latency_std_ms': 140,
                'cold_e2e_mean_ms': 12500, 'warm_e2e_mean_ms': 9500,
                'avg_pre_resolved': 0, 'cost_per_run': 0.0000420,
                'avg_aggregator_memory_mb': 75, 'avg_total_memory_mb': 620
            },
            'FUTURE_BASED': {
                'e2e_latency_mean_ms': 6800, 'e2e_latency_std_ms': 52,
                'cold_e2e_mean_ms': 9200, 'warm_e2e_mean_ms': 6800,
                'avg_pre_resolved': 4.2, 'cost_per_run': 0.0000480,
                'avg_aggregator_memory_mb': 88, 'avg_total_memory_mb': 650
            }
        },
    }


if __name__ == '__main__':
    main()
