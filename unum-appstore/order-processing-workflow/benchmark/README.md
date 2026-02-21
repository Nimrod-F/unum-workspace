# Order Processing Workflow - Benchmark Scripts

This directory contains comprehensive benchmarking tools to measure and visualize the performance improvements of Unum's Future-Based execution over CLASSIC mode.

## Scripts Overview

### 1. `benchmark_complete.py` (Recommended)

**Complete benchmark suite with automated metrics collection and visualization.**

**Features:**
- ✅ Loads function ARNs from `function-arn.yaml`
- ✅ Automatically configures both CLASSIC and FUTURE_BASED modes
- ✅ Forces cold starts for accurate measurements
- ✅ Collects detailed CloudWatch metrics with retry logic
- ✅ Calculates comprehensive performance statistics
- ✅ Generates professional comparison charts
- ✅ Saves results to JSON for further analysis

**Usage:**
```bash
# Run with default settings (5 iterations per mode)
python benchmark_complete.py

# Run with more iterations for statistical significance
python benchmark_complete.py --iterations 10

# Force cold starts for all iterations (not just first)
python benchmark_complete.py --iterations 5 --cold-all

# Skip chart generation (if matplotlib not available)
python benchmark_complete.py --skip-charts

# Run only one mode
python benchmark_complete.py --skip-classic  # Only FUTURE_BASED
python benchmark_complete.py --skip-future   # Only CLASSIC
```

**Output:**
- Console output with detailed statistics
- `results/benchmark_summary_<timestamp>.json` - Statistical summary
- `results/benchmark_runs_<timestamp>.json` - All raw data
- `results/e2e_latency_comparison.png` - Bar chart comparing E2E latency
- `results/aggregator_invocation_timing.png` - Key metric: when Aggregator starts
- `results/per_function_duration.png` - Individual function performance
- `results/execution_timeline.png` - Visual timeline showing execution flow

### 2. `run_benchmark.py`

**Basic benchmark script for quick testing.**

**Usage:**
```bash
# Run single mode
python run_benchmark.py --mode CLASSIC --iterations 5
python run_benchmark.py --mode FUTURE_BASED --iterations 5

# Run both modes
python run_benchmark.py --all --iterations 5
```

## Metrics Collected

### Primary Metrics

1. **E2E Latency** - Total time from workflow start to completion
2. **Aggregator Invocation Delay** - Time from workflow start to when Aggregator is invoked
   - CLASSIC: ~3100ms (waits for SlowChainEnd)
   - FUTURE_BASED: ~100ms (triggered by FastProcessor)
3. **Per-Function Duration** - Execution time of each Lambda function
4. **Cold Start Count** - Number of functions with cold starts
5. **Total Billed Duration** - For cost calculation

### Secondary Metrics

- Memory usage per function
- Init duration (cold start overhead)
- Invoker distribution (which function triggered Aggregator)
- Slow chain total time (sum of sequential chain functions)
- Cost estimation

## Expected Results

### Performance Improvement

Based on the workflow design:

| Metric | CLASSIC | FUTURE_BASED | Improvement |
|--------|---------|--------------|-------------|
| Aggregator Invocation | ~3100ms | ~100ms | **~3000ms (96%)** |
| E2E Latency (cold) | ~3300ms | ~3100ms | **~200ms (6%)** |
| Invoker Branch | SlowChainEnd | FastProcessor | ✓ |

### Why Future-Based is Faster

1. **Early Invocation**: Aggregator starts as soon as FastProcessor completes (~100ms)
2. **Cold Start Hiding**: Aggregator's cold start (~200ms) overlaps with slow chain execution
3. **Parallel Waiting**: Aggregator awaits slow chain results asynchronously while already running

In CLASSIC mode, the workflow must wait ~3100ms for SlowChainEnd to complete, THEN invoke Aggregator, THEN pay cold start penalty. In FUTURE_BASED mode, the cold start happens early and is hidden behind the slow chain execution.

## Understanding the Charts

### 1. E2E Latency Comparison
- Bar chart showing average end-to-end latency
- Error bars show min/max range
- Annotation shows absolute and percentage improvement

### 2. Aggregator Invocation Timing
- **Most important chart** - shows the key benefit of Future-Based execution
- Shows when Aggregator is invoked relative to workflow start
- CLASSIC: ~3100ms (after slow chain)
- FUTURE_BASED: ~100ms (after fast branch)
- Labels indicate which function triggered it

### 3. Per-Function Duration
- Side-by-side comparison of each function's execution time
- Helps identify if modes affect individual function performance
- Should be similar between modes (functions do the same work)

### 4. Execution Timeline
- Visual representation of when each function executes
- Top: CLASSIC mode timeline
- Bottom: FUTURE_BASED mode timeline
- Vertical dashed line: When Aggregator is invoked
- Color coding: Normal execution vs cold starts

## Troubleshooting

### Issue: "function-arn.yaml not found"

**Solution:** Deploy the workflow first:
```bash
cd ..
python ..\..\unum\unum-cli\unum-cli.py deploy -t unum-template.yaml
```

### Issue: "Could not fetch logs for function"

**Causes:**
- CloudWatch logs not yet propagated
- Function didn't execute
- Incorrect permissions

**Solutions:**
- Wait longer (script retries 3 times with 5s delay)
- Check Lambda execution logs manually
- Verify IAM role has CloudWatch Logs permissions

### Issue: Empty or zero metrics

**Causes:**
- Workflow failed to execute
- CloudWatch log delay
- Function ARN mismatch

**Solutions:**
- Check AWS Console for Lambda errors
- Increase retry count in script
- Verify function names match in `function-arn.yaml`

### Issue: Charts not generated

**Cause:** matplotlib not installed

**Solution:**
```bash
pip install matplotlib
```

Or run with `--skip-charts` to skip chart generation.

### Issue: Both modes show same performance

**Causes:**
- Environment variable not updated properly
- Using wrong function ARNs
- Lambda containers not recycled

**Solutions:**
- Check Aggregator environment in AWS Console
- Wait 30s between mode runs for Lambda updates
- Force cold starts with `--cold-all`

## Advanced Usage

### Analyzing Results

Results are saved as JSON for further analysis:

```python
import json

# Load summary
with open('results/benchmark_summary_<timestamp>.json') as f:
    data = json.load(f)

classic = data['CLASSIC']
future = data['FUTURE_BASED']

print(f"E2E Improvement: {data['improvement']['e2e_latency_ms']:.2f}ms")
print(f"Aggregator Improvement: {data['improvement']['aggregator_invocation_ms']:.2f}ms")
```

### Custom Chart Generation

The `generate_charts()` function can be modified to create custom visualizations:

```python
# Add to benchmark_complete.py
def generate_custom_chart(classic_runs, future_runs):
    # Custom analysis
    pass
```

### Statistical Significance

For statistically significant results, run with higher iterations:

```bash
# 20 iterations provides good statistical power
python benchmark_complete.py --iterations 20
```

Calculate p-value:
```python
from scipy import stats

classic_latencies = [r.e2e_latency_ms for r in classic_runs]
future_latencies = [r.e2e_latency_ms for r in future_runs]

t_stat, p_value = stats.ttest_ind(classic_latencies, future_latencies)
print(f"P-value: {p_value}")  # < 0.05 indicates significant difference
```

## Dependencies

Required:
- `boto3` - AWS SDK
- `pyyaml` - YAML parsing

Optional:
- `matplotlib` - Chart generation
- `scipy` - Statistical analysis

Install all:
```bash
pip install boto3 pyyaml matplotlib scipy
```

## References

- [Unum Future-Based Implementation](../../FUTURE_BASED_IMPLEMENTATION.md)
- [Unum Architecture](../../UNUM_ARCHITECTURE_AND_FUTURES.md)
- [Image Pipeline Benchmark](../../image-pipeline/benchmark/)
