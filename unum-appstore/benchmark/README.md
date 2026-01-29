# Research Workflow Benchmarks

This directory contains unified benchmark tools for evaluating serverless workflow execution modes across multiple research-inspired workflows.

## Workflows

| Workflow | Pattern | Fan-In Size | Task Durations | Research Origin |
|----------|---------|-------------|----------------|-----------------|
| progressive-aggregator | Fan-Out→Fan-In | 5 | 0.3s - 4.0s | Baseline |
| ml-training-pipeline | Ensemble Training | 4 | 0.1s - 8.0s | ML Training (SeBS-Flow) |
| video-analysis | Parallel Decode | 6 | 0.3s - 6.0s | Video Analytics |
| image-processing-pipeline | Multi-Operation | 5 | 50ms - 3.5s | Image Processing |
| genomics-pipeline | Scientific Pipeline | 6+2 | 0.4s - 3.5s | 1000Genomes |

## Execution Modes

1. **CLASSIC**: Synchronous fan-in where last completing branch executes aggregator
2. **EAGER**: Polling-based blocking fan-in with LazyInput proxy
3. **FUTURE_BASED**: Async fan-in with parallel background polling

## Usage

### Run Complete Benchmark Suite

```bash
# All workflows, all modes, 10 iterations each
python run_all_benchmarks.py --workflow all --mode all --iterations 10

# Single workflow, single mode
python run_all_benchmarks.py --workflow ml-training-pipeline --mode FUTURE_BASED --iterations 5

# Custom cold/warm split
python run_all_benchmarks.py --workflow all --mode all --iterations 10 --cold-iterations 3
```

### Generate Charts

```bash
# From benchmark results
python generate_comparison_charts.py --results-dir results/

# Output to custom directory
python generate_comparison_charts.py --results-dir results/ --output-dir charts/
```

## Output Files

### Results Directory
```
results/
├── benchmark_<workflow>_<mode>_<timestamp>_runs.json     # Individual run data
├── benchmark_<workflow>_<mode>_<timestamp>_summary.json  # Statistical summary
└── COMPARISON_REPORT_<timestamp>.md                      # Markdown report
```

### Charts Directory
```
charts/
├── e2e_latency_comparison.png     # Bar chart with error bars
├── cold_warm_comparison.png       # Cold vs warm performance
├── improvement_chart.png          # FUTURE_BASED improvement %
├── pre_resolved_efficiency.png    # Background polling efficiency
├── workflow_profiles.png          # Task duration profiles
└── summary_table.png              # Results table
```

## Expected Results

Based on task duration variance, expected FUTURE_BASED improvements:

| Workflow | Duration Variance | Expected Improvement |
|----------|-------------------|---------------------|
| ml-training-pipeline | σ=3.2s | 25-30% |
| video-analysis | σ=2.1s | 20-25% |
| genomics-pipeline | σ=1.2s | 15-20% |
| image-processing-pipeline | σ=1.4s | 18-22% |
| progressive-aggregator | σ=1.5s | 5-10% |

Higher task duration variance leads to greater benefit from FUTURE_BASED execution,
as faster tasks complete and results are pre-resolved before aggregator access.

## Prerequisites

1. AWS credentials configured
2. Workflows deployed to AWS Lambda
3. Python dependencies:
   ```bash
   pip install boto3 matplotlib numpy
   ```

## Architecture

```
                    ┌─────────────┐
                    │   Entry     │
                    │  Function   │
                    └──────┬──────┘
                           │ Fan-Out
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │ Task 1  │    │ Task 2  │    │ Task N  │
      │ (fast)  │    │(medium) │    │ (slow)  │
      └────┬────┘    └────┬────┘    └────┬────┘
           │               │               │
           └───────────────┼───────────────┘
                           │ Fan-In
                    ┌──────▼──────┐
                    │  Aggregator │
                    │  (Terminal) │
                    └─────────────┘
```

FUTURE_BASED mode enables background polling during aggregator execution,
so fast tasks are already resolved when accessed.
