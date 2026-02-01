# Multi-Source Dashboard Benchmark Quick Reference

## Two Ways to Benchmark Your App

### 🎯 Option 1: App-Specific (Recommended for Development)

**Location**: `multi-source-dashboard/benchmark/`

**Single command**:
```bash
cd multi-source-dashboard/benchmark
python run_benchmark.py --mode all --iterations 10
python generate_charts.py --results-dir results/
```

**Generates**:
- 6 detailed charts focused on your app
- Pre-resolved efficiency (4-5 of 6 inputs)
- CLASSIC vs EAGER vs FUTURE_BASED comparison
- Results in one JSON file

**Best for**:
- ✅ Quick iteration during development
- ✅ Deep analysis of your specific app
- ✅ Understanding your 6-source fan-in pattern

---

### 🌐 Option 2: General Framework (Recommended for Research)

**Location**: `unum-appstore/benchmark/`

**Single command**:
```bash
cd unum-appstore/benchmark
python run_all_benchmarks.py --workflow multi-source-dashboard --mode all --iterations 10
python generate_comparison_charts.py --results-dir results/
```

**Generates**:
- Cross-workflow comparison charts
- Shows your app vs other workflows
- Side-by-side performance analysis
- Results in multiple JSON files (per mode)

**Best for**:
- ✅ Research paper with multiple case studies
- ✅ Comparing different workflow patterns
- ✅ Showing breadth of FUTURE_BASED benefits

---

## Quick Commands Cheat Sheet

### App-Specific Benchmarks

```bash
# Full benchmark (all modes)
cd multi-source-dashboard/benchmark
python run_benchmark.py --mode all --iterations 10

# Single mode
python run_benchmark.py --mode FUTURE_BASED --iterations 20

# Generate charts
python generate_charts.py --results-dir results/

# Analyze results
python collect_metrics.py --results-file results/benchmark_*.json
```

### General Framework Benchmarks

```bash
# Your app only
cd unum-appstore/benchmark
python run_all_benchmarks.py --workflow multi-source-dashboard --mode all --iterations 10

# Compare 2 workflows
python run_all_benchmarks.py --workflow progressive-aggregator,multi-source-dashboard --mode all --iterations 10

# All workflows (takes time!)
python run_all_benchmarks.py --workflow all --mode all --iterations 10

# Generate comparison charts
python generate_comparison_charts.py --results-dir results/
```

---

## Chart Types Comparison

### Your App-Specific Charts (6 charts)

1. **e2e_latency_comparison.png** - Bar chart with error bars
2. **cold_vs_warm_comparison.png** - Cold start impact
3. **improvement_over_classic.png** - Percentage improvement
4. **pre_resolved_efficiency.png** - Background polling (6 inputs)
5. **latency_distribution.png** - Box plot showing variance
6. **summary_table.png** - Complete metrics table

### General Framework Charts (5-6 charts)

1. **e2e_latency_comparison.png** - Multi-workflow bar chart
2. **cold_warm_comparison.png** - Cross-workflow cold/warm
3. **improvement_chart.png** - Improvement % for all workflows
4. **pre_resolved_efficiency.png** - Cross-workflow efficiency
5. **workflow_profiles.png** - Task duration distributions
6. **summary_table.png** - All workflows metrics

---

## Complete Workflow for Research Paper

```bash
# 1. Fix and deploy (one time)
cd multi-source-dashboard
unum-cli build -g -p aws
unum-cli deploy -b

# 2. Run general benchmark (for comparison)
cd ../benchmark
python run_all_benchmarks.py \
    --workflow progressive-aggregator,multi-source-dashboard,video-analysis \
    --mode all --iterations 15

# 3. Generate cross-workflow charts
python generate_comparison_charts.py --results-dir results/

# 4. Run detailed benchmark for your app
cd ../multi-source-dashboard/benchmark
python run_benchmark.py --mode all --iterations 15

# 5. Generate detailed charts
python generate_charts.py --results-dir results/

# Now you have:
# - General comparison (breadth): unum-appstore/benchmark/charts/
# - Detailed analysis (depth): multi-source-dashboard/benchmark/charts/
```

---

## File Locations

```
unum-appstore/
├── benchmark/                           # General framework
│   ├── run_all_benchmarks.py           # Multi-workflow runner
│   ├── generate_comparison_charts.py   # Cross-workflow charts
│   ├── results/                        # Per-mode JSON files
│   └── charts/                         # Comparison charts
│
└── multi-source-dashboard/
    └── benchmark/                       # App-specific
        ├── run_benchmark.py             # Your app runner
        ├── generate_charts.py           # Your app charts
        ├── results/                     # All-modes-in-one JSON
        └── charts/                      # Detailed charts
```

---

## When to Use Which?

| Scenario | Use This |
|----------|----------|
| Testing during development | App-specific |
| Quick iteration | App-specific |
| Detailed analysis of 6-source fan-in | App-specific |
| Compare multiple workflows | General framework |
| Research paper with case studies | General framework |
| Show breadth AND depth | **Both!** |

---

## Expected Results After Fix

Once you rebuild with correct dependencies:

### CLASSIC Mode
- E2E: 3,800-4,200 ms
- Pre-resolved: 0

### FUTURE_BASED Mode
- E2E: 3,000-3,400 ms
- Pre-resolved: 4-5 (of 6)
- Improvement: **15-25%**

---

## Troubleshooting

### "Workflow not found"
Make sure you've deployed the stack:
```bash
aws cloudformation describe-stacks --stack-name multi-source-dashboard
```

### "No results to plot"
Check that results files exist:
```bash
ls -la results/benchmark_*.json
```

### Charts show wrong data
Verify which generator you're using:
- App-specific: expects one JSON with all modes
- General: expects multiple JSONs (one per mode)

---

## Quick Start (Right Now)

```bash
# 1. Go to benchmark directory
cd unum-appstore/benchmark

# 2. Run a quick test
python run_all_benchmarks.py \
    --workflow multi-source-dashboard \
    --mode CLASSIC \
    --iterations 2

# 3. If it works, run full benchmark after fixing Lambda
```

**Everything is ready - just rebuild your Lambda functions with the fixed requirements.txt!**
