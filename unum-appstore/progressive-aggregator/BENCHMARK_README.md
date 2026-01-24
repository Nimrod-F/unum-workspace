# Progressive-Aggregator Benchmark Suite

This directory contains comprehensive benchmarking tools for comparing the three execution modes in Unum:

1. **CLASSIC** - Traditional last-invoker fan-in
2. **EAGER** - Polling-based fan-in with LazyInput proxy
3. **FUTURE_BASED** - Async fan-in with parallel background polling

## Quick Start

```powershell
# Run quick test (single iteration, no mode switching)
python benchmark\quick_test.py --iterations 3

# Run full benchmark for current mode
python benchmark\run_benchmark.py --mode FUTURE_BASED --iterations 10 --no-deploy

# Run full benchmark with mode switching and deployment
python benchmark\run_benchmark.py --all --iterations 10

# Or use PowerShell script
.\benchmark\run_full_benchmark.ps1 -Iterations 10
```

## Scripts

### `run_benchmark.py`

Main benchmark runner. Invokes workflows, collects CloudWatch metrics, and saves results.

```bash
# Single mode
python benchmark\run_benchmark.py --mode CLASSIC --iterations 10

# All modes (switches config and redeploys)
python benchmark\run_benchmark.py --all --iterations 10

# With cold start tests
python benchmark\run_benchmark.py --mode FUTURE_BASED --iterations 20 --cold 5 --warmup 3
```

Options:

- `--mode`: CLASSIC, EAGER, or FUTURE_BASED
- `--all`: Run all three modes
- `--iterations`: Number of warm iterations
- `--warmup`: Warmup runs (not counted)
- `--cold`: Cold start iterations
- `--no-deploy`: Skip mode switching/deployment
- `--output`: Results directory

### `quick_test.py`

Quick verification script for current deployment.

```bash
python benchmark\quick_test.py --iterations 3
```

### `analyze_results.py`

Statistical analysis of benchmark results.

```bash
python benchmark\analyze_results.py benchmark\results\
python benchmark\analyze_results.py benchmark\results\ --latex
python benchmark\analyze_results.py benchmark\results\ --json
```

### `generate_charts.py`

Generate publication-quality visualizations.

```bash
python benchmark\generate_charts.py benchmark\results\ --output benchmark\figures\
python benchmark\generate_charts.py benchmark\results\ --format svg
```

## Metrics Collected

### Lambda Metrics (from CloudWatch REPORT logs)

- Duration (ms)
- Billed Duration (ms)
- Memory Size (MB)
- Max Memory Used (MB)
- Init Duration (cold starts only)

### Fan-In Metrics (from Aggregator logs)

- Initially Ready count
- Pre-Resolved count (background polling effectiveness)
- Wait Duration (ms)
- Poll Count

### Derived Metrics

- E2E Latency (invoke → completion)
- Cold Start Rate
- Estimated Cost (Lambda compute + requests + DynamoDB)

## Output Files

Results are saved in `benchmark/results/`:

- `benchmark_{MODE}_{TIMESTAMP}_runs.json` - Raw per-run data
- `benchmark_{MODE}_{TIMESTAMP}_summary.json` - Statistical summary

Charts are saved in `benchmark/figures/`:

- `e2e_latency_comparison.png` - E2E latency bar chart
- `fanin_wait_comparison.png` - Fan-in wait times
- `pre_resolved_comparison.png` - Background polling effectiveness
- `cost_comparison.png` - Cost per run
- `combined_metrics.png` - Dashboard with all metrics
- `chart_data.json` - Data for external visualization tools

## Expected Results

Based on the Future-Based optimization (parallel background polling):

| Metric       | CLASSIC  | EAGER       | FUTURE_BASED   |
| ------------ | -------- | ----------- | -------------- |
| Pre-Resolved | 0/5      | ~1/5        | ~3-4/5         |
| Fan-In Wait  | Highest  | Medium      | Lowest         |
| E2E Latency  | Baseline | ~10% faster | ~20-40% faster |
| Cost         | Baseline | +~5%        | +~10%          |

The FUTURE_BASED mode trades slightly higher cost (more DynamoDB reads for parallel polling) for significantly lower latency by resolving multiple inputs in parallel.

## Workflow Structure

```
FanOut (1 invocation)
   ↓ fans out to 5 parallel Source functions
Source[0..4] (5 invocations, with varying delays: 2s, 3s, 4s, 0.3s, 0.5s)
   ↓ fans in to single Aggregator
Aggregator (1 invocation)
   → Receives AsyncFutureInputList with 5 futures
   → Background polling resolves fast sources (3,4) while waiting for slow ones
```

Total: 7 Lambda invocations per workflow run.
