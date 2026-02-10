"""
Generate benchmark charts for Partial Parameter Streaming demo.

This script creates visualization charts for the benchmark results
comparing Normal vs Streaming execution modes.

Requirements:
    pip install matplotlib seaborn pandas numpy

Usage:
    python generate_charts.py
"""

import os
import json

# Benchmark data from actual runs
BENCHMARK_DATA = {
    "latency": {
        "warm": {"normal": 10.3, "streaming": 4.4},
        "cold": {"normal": 12.1, "streaming": 8.2}
    },
    "improvement": {
        "warm_latency": 57,
        "cold_latency": 32,
        "throughput": 134,
        "parallelism": 300
    },
    "function_duration": {
        "Generator": {"normal": 2.52, "streaming": 2.58},
        "Processor": {"normal": 2.58, "streaming": 2.61},
        "Analyzer": {"normal": 2.55, "streaming": 2.59},
        "Reporter": {"normal": 2.51, "streaming": 2.53}
    },
    "memory": {
        "Generator": {"normal": 68, "streaming": 72},
        "Processor": {"normal": 65, "streaming": 75},
        "Analyzer": {"normal": 66, "streaming": 74},
        "Reporter": {"normal": 64, "streaming": 64}
    },
    "cost": {
        "Generator": {"normal": 4.20e-6, "streaming": 4.31e-6},
        "Processor": {"normal": 4.31e-6, "streaming": 4.35e-6},
        "Analyzer": {"normal": 4.25e-6, "streaming": 4.32e-6},
        "Reporter": {"normal": 4.19e-6, "streaming": 4.22e-6},
        "DynamoDB": {"normal": 0, "streaming": 5e-7}
    },
    "network_overhead": {
        "DynamoDB Write": 15,
        "DynamoDB Read": 12,
        "Future Resolution": 18,
        "Lambda Invoke": 45
    },
    "billed_duration": {
        "normal": 41200,
        "streaming": 42400
    }
}

# Chart URLs generated via MCP chart server
CHART_URLS = {
    "latency_comparison": "https://mdn.alipayobjects.com/one_clip/afts/img/kNk7QrZg4t0AAAAARSAAAAgAoEACAQFr/original",
    "improvements": "https://mdn.alipayobjects.com/one_clip/afts/img/QUc2RbnmeiUAAAAARLAAAAgAoEACAQFr/original",
    "data_flow": "https://mdn.alipayobjects.com/one_clip/afts/img/EczVTbERpmoAAAAAQlAAAAgAoEACAQFr/original",
    "active_functions": "https://mdn.alipayobjects.com/one_clip/afts/img/WTWgQJCojQgAAAAARsAAAAgAoEACAQFr/original",
    "concurrency": "https://mdn.alipayobjects.com/one_clip/afts/img/DCupT6mSGPsAAAAARPAAAAgAoEACAQFr/original",
    "cost_comparison": "https://mdn.alipayobjects.com/one_clip/afts/img/DDFcTrU3ZhUAAAAARVAAAAgAoEACAQFr/original",
    "memory_usage": "https://mdn.alipayobjects.com/one_clip/afts/img/c41tSol1XZwAAAAARRAAAAgAoEACAQFr/original",
    "network_overhead": "https://mdn.alipayobjects.com/one_clip/afts/img/DHDpTY-80lgAAAAARHAAAAgAoEACAQFr/original",
    "function_duration": "https://mdn.alipayobjects.com/one_clip/afts/img/VQZWQaOIp7kAAAAARSAAAAgAoEACAQFr/original",
    "billed_duration": "https://mdn.alipayobjects.com/one_clip/afts/img/J3-OTJRbJL4AAAAARJAAAAgAoEACAQFr/original",
    "latency_waterfall": "https://mdn.alipayobjects.com/one_clip/afts/img/gneMQKAsseAAAAAARfAAAAgAoEACAQFr/original",
    "time_breakdown": "https://mdn.alipayobjects.com/one_clip/afts/img/sh1CQburomIAAAAARWAAAAgAoEACAQFr/original",
    "radar_comparison": "https://mdn.alipayobjects.com/one_clip/afts/img/sKx8TaYQm90AAAAATLAAAAgAoEACAQFr/original"
}


def print_summary():
    """Print benchmark summary to console."""
    print("=" * 60)
    print("PARTIAL PARAMETER STREAMING BENCHMARK RESULTS")
    print("=" * 60)
    
    print("\n📊 LATENCY COMPARISON")
    print("-" * 40)
    print(f"  Warm Start:")
    print(f"    Normal:    {BENCHMARK_DATA['latency']['warm']['normal']:.1f}s")
    print(f"    Streaming: {BENCHMARK_DATA['latency']['warm']['streaming']:.1f}s")
    print(f"    Improvement: {BENCHMARK_DATA['improvement']['warm_latency']}%")
    
    print(f"\n  Cold Start:")
    print(f"    Normal:    {BENCHMARK_DATA['latency']['cold']['normal']:.1f}s")
    print(f"    Streaming: {BENCHMARK_DATA['latency']['cold']['streaming']:.1f}s")
    print(f"    Improvement: {BENCHMARK_DATA['improvement']['cold_latency']}%")
    
    print("\n💰 COST ANALYSIS")
    print("-" * 40)
    normal_total = sum(v["normal"] for v in BENCHMARK_DATA["cost"].values())
    streaming_total = sum(v["streaming"] for v in BENCHMARK_DATA["cost"].values())
    print(f"  Normal Mode:    ${normal_total*1e6:.2f}µ per invocation")
    print(f"  Streaming Mode: ${streaming_total*1e6:.2f}µ per invocation")
    print(f"  Cost Overhead:  {((streaming_total/normal_total)-1)*100:.1f}%")
    
    print("\n🧠 MEMORY USAGE")
    print("-" * 40)
    normal_mem = sum(v["normal"] for v in BENCHMARK_DATA["memory"].values()) / 4
    streaming_mem = sum(v["streaming"] for v in BENCHMARK_DATA["memory"].values()) / 4
    print(f"  Normal Mode:    {normal_mem:.1f} MB average")
    print(f"  Streaming Mode: {streaming_mem:.1f} MB average")
    print(f"  Memory Overhead: {((streaming_mem/normal_mem)-1)*100:.1f}%")
    
    print("\n🌐 NETWORK OVERHEAD")
    print("-" * 40)
    for op, latency in BENCHMARK_DATA["network_overhead"].items():
        print(f"  {op}: {latency} ms")
    
    print("\n📈 CHART URLs")
    print("-" * 40)
    for name, url in CHART_URLS.items():
        print(f"  {name}:")
        print(f"    {url}")
    
    print("\n" + "=" * 60)


def save_data():
    """Save benchmark data to JSON file."""
    output_path = os.path.join(os.path.dirname(__file__), "charts", "benchmark_data.json")
    
    with open(output_path, "w") as f:
        json.dump({
            "benchmark_data": BENCHMARK_DATA,
            "chart_urls": CHART_URLS
        }, f, indent=2)
    
    print(f"Benchmark data saved to: {output_path}")


def try_matplotlib_charts():
    """
    Attempt to generate charts using matplotlib.
    Falls back gracefully if not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        charts_dir = os.path.join(os.path.dirname(__file__), "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # 1. Latency Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(2)
        width = 0.35
        
        normal_vals = [BENCHMARK_DATA["latency"]["warm"]["normal"], 
                       BENCHMARK_DATA["latency"]["cold"]["normal"]]
        streaming_vals = [BENCHMARK_DATA["latency"]["warm"]["streaming"],
                          BENCHMARK_DATA["latency"]["cold"]["streaming"]]
        
        bars1 = ax.bar(x - width/2, normal_vals, width, label='Normal', color='#FF6B6B')
        bars2 = ax.bar(x + width/2, streaming_vals, width, label='Streaming', color='#4ECDC4')
        
        ax.set_ylabel('Latency (seconds)')
        ax.set_title('End-to-End Latency: Normal vs Streaming')
        ax.set_xticks(x)
        ax.set_xticklabels(['Warm Start', 'Cold Start'])
        ax.legend()
        ax.bar_label(bars1, fmt='%.1f')
        ax.bar_label(bars2, fmt='%.1f')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "latency_comparison_local.png"), dpi=150)
        plt.close()
        
        # 2. Improvement Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = list(BENCHMARK_DATA["improvement"].keys())
        values = list(BENCHMARK_DATA["improvement"].values())
        colors = ['#4ECDC4'] * len(metrics)
        
        bars = ax.bar(metrics, values, color=colors)
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Streaming Mode Improvements')
        ax.bar_label(bars, fmt='%d%%')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "improvements_local.png"), dpi=150)
        plt.close()
        
        # 3. Memory Usage Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        functions = list(BENCHMARK_DATA["memory"].keys())
        x = np.arange(len(functions))
        
        normal_mem = [v["normal"] for v in BENCHMARK_DATA["memory"].values()]
        streaming_mem = [v["streaming"] for v in BENCHMARK_DATA["memory"].values()]
        
        bars1 = ax.bar(x - width/2, normal_mem, width, label='Normal', color='#FF6B6B')
        bars2 = ax.bar(x + width/2, streaming_mem, width, label='Streaming', color='#4ECDC4')
        
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage per Function')
        ax.set_xticks(x)
        ax.set_xticklabels(functions)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "memory_local.png"), dpi=150)
        plt.close()
        
        print("✅ Local matplotlib charts generated in charts/ folder")
        return True
        
    except ImportError:
        print("ℹ️  matplotlib not installed. Using hosted chart URLs instead.")
        print("   Install with: pip install matplotlib")
        return False


if __name__ == "__main__":
    print_summary()
    save_data()
    try_matplotlib_charts()
