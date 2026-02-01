# Using General Benchmark System with Multi-Source Dashboard

## What Was Added

Your `multi-source-dashboard` application has been integrated into the general benchmark framework at `unum-appstore/benchmark/`. This allows you to:

1. **Use the unified benchmark runner** to run your app alongside other workflows
2. **Generate comparison charts** showing your app vs other workflows
3. **Benefit from advanced analysis** tools in the central benchmark system

## Integration Details

Added to `unum-appstore/benchmark/run_all_benchmarks.py`:

```python
'multi-source-dashboard': {
    'stack_name': 'multi-source-dashboard',
    'entry_function': 'TriggerDashboard',
    'terminal_function': 'MergeDashboardData',
    'dynamodb_table': 'unum-intermediary-multi-dashboard',
    'fan_in_sizes': [6],  # 6 parallel data sources
    'expected_delays': [0.25, 0.325, 0.5, 1.75, 0.9, 1.45],
    'description': 'Dashboard: Sales(250ms), Inventory(325ms), Marketing(500ms), Market(1750ms), Weather(900ms), Competitor(1450ms)',
}
```

## Usage

### Option 1: Run Only Multi-Source Dashboard

```bash
cd unum-appstore/benchmark

# Benchmark all modes for your app
python run_all_benchmarks.py \
    --workflow multi-source-dashboard \
    --mode all \
    --iterations 10

# Benchmark specific mode
python run_all_benchmarks.py \
    --workflow multi-source-dashboard \
    --mode FUTURE_BASED \
    --iterations 20
```

### Option 2: Compare with Other Workflows

```bash
# Run benchmarks for multiple workflows
python run_all_benchmarks.py \
    --workflow progressive-aggregator,multi-source-dashboard \
    --mode all \
    --iterations 10

# Run all workflows (takes a while!)
python run_all_benchmarks.py \
    --workflow all \
    --mode all \
    --iterations 10
```

### Option 3: Generate Comparison Charts

After running benchmarks for multiple workflows:

```bash
# Generate charts comparing all workflows
python generate_comparison_charts.py --results-dir results/

# This creates:
# - e2e_latency_comparison.png (all workflows side-by-side)
# - cold_warm_comparison.png
# - improvement_chart.png
# - pre_resolved_efficiency.png
# - workflow_profiles.png (task duration distributions)
# - summary_table.png
```

## Two Chart Generation Systems

You now have **two ways** to generate charts:

### 1. App-Specific Charts (Your Generator)

**Location**: `multi-source-dashboard/benchmark/generate_charts.py`

**Use when**: You want detailed analysis of just your app

**Generates**:
- Focused on CLASSIC vs EAGER vs FUTURE_BASED for your app only
- Shows your specific 6-input fan-in metrics
- Pre-resolved efficiency chart tailored to 6 sources

**Usage**:
```bash
cd multi-source-dashboard/benchmark
python generate_charts.py --results-file results/benchmark_*.json
```

### 2. General Comparison Charts (Framework Generator)

**Location**: `unum-appstore/benchmark/generate_comparison_charts.py`

**Use when**: You want to compare multiple workflows

**Generates**:
- Side-by-side comparison of different workflows
- Shows which workflows benefit most from FUTURE_BASED
- Task duration profile comparison
- Cross-workflow analysis

**Usage**:
```bash
cd unum-appstore/benchmark
python generate_comparison_charts.py --results-dir results/
```

## Results File Naming

The general benchmark system uses a specific naming convention:

```
results/
├── benchmark_multi-source-dashboard_CLASSIC_20260201_210000_runs.json
├── benchmark_multi-source-dashboard_CLASSIC_20260201_210000_summary.json
├── benchmark_multi-source-dashboard_FUTURE_BASED_20260201_210500_runs.json
├── benchmark_multi-source-dashboard_FUTURE_BASED_20260201_210500_summary.json
└── COMPARISON_REPORT_20260201_211000.md
```

vs your app-specific format:

```
multi-source-dashboard/benchmark/results/
└── benchmark_20260201_205450.json  # All modes in one file
```

## Complete Workflow Example

### 1. Deploy Your App

```bash
cd unum-appstore/multi-source-dashboard
unum-cli build -g -p aws
unum-cli deploy -b
```

### 2. Run Benchmark (Choose One)

**Option A: Use your app-specific runner**
```bash
cd benchmark
python run_benchmark.py --mode all --iterations 10
```

**Option B: Use general framework**
```bash
cd ../../benchmark  # Go to unum-appstore/benchmark
python run_all_benchmarks.py \
    --workflow multi-source-dashboard \
    --mode all \
    --iterations 10
```

### 3. Generate Charts (Choose One or Both!)

**Your app-specific charts**:
```bash
cd ../multi-source-dashboard/benchmark
python generate_charts.py --results-dir results/
# Output: charts/ directory with 6 focused charts
```

**General comparison charts**:
```bash
cd ../../benchmark
python generate_comparison_charts.py --results-dir results/
# Output: charts/ directory with cross-workflow comparison
```

## When to Use Each System

### Use App-Specific (`multi-source-dashboard/benchmark/`)

✅ Deep-dive analysis of your single app
✅ Detailed pre-resolved efficiency metrics
✅ Quick iteration during development
✅ Focus on your 6-source fan-in pattern

### Use General Framework (`unum-appstore/benchmark/`)

✅ Compare your app to other workflows
✅ Show that FUTURE_BASED benefits vary by workload
✅ Research paper showing multiple case studies
✅ Demonstrate breadth of applicability

## Example Research Use Case

For a comprehensive research paper:

```bash
# 1. Run benchmarks for multiple workflows
cd unum-appstore/benchmark
python run_all_benchmarks.py \
    --workflow progressive-aggregator,multi-source-dashboard,video-analysis \
    --mode all \
    --iterations 15

# 2. Generate cross-workflow comparison
python generate_comparison_charts.py --results-dir results/

# 3. Generate detailed charts for your app
cd ../multi-source-dashboard/benchmark
python generate_charts.py --results-dir results/

# Now you have:
# - General charts showing multi-source-dashboard vs other workflows
# - Detailed charts diving deep into multi-source-dashboard
```

## Configuration in General Framework

The configuration you added specifies:

- **stack_name**: CloudFormation stack name
- **entry_function**: First function to invoke (TriggerDashboard)
- **terminal_function**: Final aggregator (MergeDashboardData)
- **dynamodb_table**: Where checkpoints are stored
- **fan_in_sizes**: [6] - one fan-in of 6 branches
- **expected_delays**: Average delay for each of 6 sources (in seconds)

These match your application's actual structure.

## Verifying Integration

Test that the integration works:

```bash
cd unum-appstore/benchmark

# List available workflows
python run_all_benchmarks.py --list-workflows

# Should show:
# - progressive-aggregator
# - ml-training-pipeline
# - video-analysis
# - image-processing-pipeline
# - genomics-pipeline
# - multi-source-dashboard  ← Your app!

# Quick test run
python run_all_benchmarks.py \
    --workflow multi-source-dashboard \
    --mode CLASSIC \
    --iterations 2
```

## Benefits of Integration

1. **Consistency**: Same benchmark methodology across all apps
2. **Comparison**: Show your app in context of other workflows
3. **Automation**: Use existing CI/CD infrastructure
4. **Analysis**: Leverage advanced metrics collection
5. **Visualization**: Professional cross-workflow charts

## Summary

You now have **flexibility**:

- **Quick development**: Use `multi-source-dashboard/benchmark/`
- **Comprehensive research**: Use `unum-appstore/benchmark/`
- **Both!**: Run both and include different chart types in your paper

The integration is complete and ready to use!
