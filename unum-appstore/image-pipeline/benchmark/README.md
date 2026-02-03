# Image Pipeline Benchmark

Comprehensive benchmark comparing **CLASSIC** vs **FUTURE_BASED** execution modes for the image processing workflow.

## Workflow Structure

```
ImageLoader → [Thumbnail, Transform, Filters, Contour] → Publisher
              ^----------- fan-out -----------^        ^fan-in^
```

### Expected Task Durations (Real PIL Computation)
| Function | Expected Duration | Notes |
|----------|------------------|-------|
| ImageLoader | ~50ms | Load image from S3 |
| Thumbnail | ~80ms | Resize to 128x128 (**FASTEST**) |
| Transform | ~120ms | Rotate + flip |
| Filters | ~180ms | Blur + sharpen |
| Contour | ~300ms | Edge detection (**SLOWEST**) |
| Publisher | ~50ms | Aggregate results |

## Benchmarks

### 1. Quick Benchmark (`quick_benchmark.py`)
Standard comparison without artificial delays - tests natural execution.

### 2. Artificial Delay Benchmark (`delay_benchmark.py`) ⭐ NEW
Introduces configurable artificial delays to each branch to clearly demonstrate
Future-Based execution benefits under different variance scenarios.

## Artificial Delay Benchmark

This benchmark is specifically designed to highlight the advantages of Future-Based execution
by creating controlled scenarios with different branch execution time variances.

### Key Insight

| Mode | Behavior | E2E Latency |
|------|----------|-------------|
| **CLASSIC** | Fan-in waits for ALL branches | `max(branch_times)` |
| **FUTURE_BASED** | Fan-in starts with FIRST branch | `min(branch_time) + processing` |

The larger the variance between branch times, the greater the benefit of Future-Based execution.

### Delay Scenarios

| Scenario | Thumbnail | Transform | Filters | Contour | Expected Savings |
|----------|-----------|-----------|---------|---------|------------------|
| **Uniform** | 0ms | 0ms | 0ms | 0ms | Baseline (natural variance) |
| **Staggered** | 0ms | 1000ms | 2000ms | 3000ms | ~3000ms |
| **Extreme** | 0ms | 0ms | 0ms | 5000ms | ~5000ms |
| **Reversed** | 3000ms | 2000ms | 1000ms | 0ms | ~3000ms |
| **Moderate** | 0ms | 500ms | 1000ms | 1500ms | ~1500ms |
| **Bimodal** | 0ms | 0ms | 2000ms | 2000ms | ~2000ms |

### Usage

```bash
# Run staggered scenario (default)
python delay_benchmark.py

# Run specific scenarios
python delay_benchmark.py --scenarios staggered extreme --iterations 5

# Run all scenarios
python delay_benchmark.py --scenarios all --iterations 3

# Custom delays (Thumbnail, Transform, Filters, Contour in ms)
python delay_benchmark.py --custom 0,1000,2000,5000 --iterations 3

# Skip cold start forcing (faster, less accurate)
python delay_benchmark.py --scenarios staggered --no-cold --iterations 5
```

### How It Works

1. **Configuration**: Sets `ARTIFICIAL_DELAY_MS` environment variable on each branch Lambda
2. **Branch Execution**: Each branch reads the delay and sleeps after completing real work
3. **Measurement**: Compares E2E latency between CLASSIC and FUTURE_BASED modes
4. **Analysis**: Tracks which branch invokes Publisher (slowest in CLASSIC, fastest in FUTURE)

### Expected Results

For the **Staggered** scenario (0, 1000, 2000, 3000ms delays):

```
CLASSIC Mode:
  - Thumbnail completes at ~80ms
  - Transform completes at ~1120ms  
  - Filters completes at ~2180ms
  - Contour completes at ~3300ms     ← Publisher invoked here
  - E2E: ~3400ms

FUTURE_BASED Mode:
  - Thumbnail completes at ~80ms     ← Publisher invoked here (uses futures)
  - Publisher starts, waits for remaining data via futures
  - E2E: ~3400ms BUT Publisher started 3+ seconds earlier
  - Cold start latency eliminated for Publisher
```

### Generate Charts

```bash
# Generate visualization from latest results
python generate_delay_charts.py --latest

# Generate from specific results file
python generate_delay_charts.py delay_benchmark_20260201_123456.json
```

## Execution Modes

### CLASSIC Mode
- Fan-in triggered by the **SLOWEST** branch (Contour)
- Publisher waits for all branches to complete before starting
- Total latency = ImageLoader + Contour + Publisher

### FUTURE_BASED Mode
- Fan-in triggered by the **FASTEST** branch (Thumbnail)
- Publisher starts early, other results pre-resolved in background
- Total latency = ImageLoader + Thumbnail + Publisher
- **Expected improvement: proportional to branch variance**

## Metrics Collected

| Metric | Description |
|--------|-------------|
| E2E Latency | Time from invoke to completion |
| Per-Function Duration | CloudWatch REPORT logs |
| Billed Duration | For cost calculation |
| Cold Start Duration | Init Duration from logs |
| Memory Usage | Max Memory Used |
| Invoker Branch | Which branch triggered Publisher |
| Artificial Delay | Configured delay per branch |
| Theoretical Savings | max(branch_times) - min(branch_times) |

## Usage

### Prerequisites

1. AWS credentials configured with profile `research-profile`
2. Image pipeline deployed to AWS Lambda
3. Test image uploaded to S3

```bash
# Upload test image
python upload_test_image.py

# Install dependencies
pip install boto3 pyyaml matplotlib numpy
```

### Run Complete Benchmark

```bash
# Run both modes (default: 5 iterations each)
python run_benchmark.py --all

# Custom iterations
python run_benchmark.py --all --iterations 10 --cold-iterations 3

# Single mode
python run_benchmark.py --mode CLASSIC --iterations 5
python run_benchmark.py --mode FUTURE_BASED --iterations 5
```

### Generate Charts

```bash
# Generate from results
python generate_charts.py --results-dir results/

# Custom output directory
python generate_charts.py --results-dir results/ --output-dir charts/
```

### Quick Benchmark (No Charts)

```bash
# Quick test with 3 iterations
python quick_benchmark.py
```

## Output Files

### Results Directory
```
results/
├── benchmark_image-pipeline_CLASSIC_<timestamp>_runs.json
├── benchmark_image-pipeline_CLASSIC_<timestamp>_summary.json
├── benchmark_image-pipeline_FUTURE_BASED_<timestamp>_runs.json
├── benchmark_image-pipeline_FUTURE_BASED_<timestamp>_summary.json
└── COMPARISON_REPORT_<timestamp>.md
```

### Charts Directory
```
charts/
├── e2e_latency_comparison.png      # Bar chart with error bars
├── cold_warm_comparison.png        # Cold vs warm performance
├── per_function_breakdown.png      # Stacked duration bars
├── invoker_distribution.png        # Pie charts showing invokers
├── branch_timing_profile.png       # Branch duration comparison
├── improvement_summary.png         # Horizontal improvement bars
├── key_metrics_table.png           # Summary metrics table
└── timeline_diagram.png            # Visual workflow timeline
```

## Expected Results

Based on real PIL computation timing:

| Metric | CLASSIC | FUTURE_BASED | Improvement |
|--------|---------|--------------|-------------|
| E2E Latency | ~1600ms | ~300ms | **~1300ms (80%)** |
| Invoker | Contour | Thumbnail | - |
| Pre-resolved | 0 | 3-4 | - |

### Why FUTURE_BASED is Faster

1. **Early Trigger**: Publisher starts as soon as Thumbnail completes (~100ms)
2. **Pre-resolved Futures**: By the time Publisher accesses Contour result, it's already available
3. **No Waiting**: No need to wait for slowest branch before starting aggregation

```
CLASSIC:    |--Loader--|--Contour (1400ms)--|--Publisher--|  Total: ~1600ms
FUTURE:     |--Loader--|--Thumb(100ms)--|--Publisher--|     Total: ~300ms
                        └─ Others complete in background
```

## Configuration

### Environment Variables
```bash
export AWS_REGION=eu-central-1
export AWS_PROFILE=research-profile
export TEST_BUCKET=unum-benchmark-images
export TEST_KEY=test-images/sample-1920x1080.jpg
```

### Modify in Script
Edit `run_benchmark.py`:
```python
REGION = 'eu-central-1'
PROFILE = 'research-profile'
TEST_BUCKET = 'unum-benchmark-images'
TEST_KEY = 'test-images/sample-1920x1080.jpg'
```

## Troubleshooting

### No CloudWatch Logs
- Increase wait time in `run_benchmark.py` (default: 15s)
- Check Lambda function permissions

### Function Errors
- Verify S3 bucket and image exist
- Check Lambda role has S3 read permissions
- Ensure PIL/Pillow layer is attached

### Cold Start Inconsistency
- Increase cold start wait time
- Force cold starts by updating function config

## Research Context

This benchmark demonstrates the **Future-Based Execution** optimization for serverless workflows with fan-in patterns. The key insight is:

> In fan-in scenarios with varying branch durations, FUTURE_BASED execution allows the aggregator to start early (triggered by the fastest branch) while other results are resolved in the background via parallel polling.

This is particularly beneficial when:
- Branch durations have high variance (σ > 1s)
- Aggregator processing is independent of input order
- Latency is more important than raw compute efficiency
