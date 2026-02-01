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
| Thumbnail | ~100ms | Resize to 128x128 (**FASTEST**) |
| Transform | ~200ms | Rotate + flip |
| Filters | ~350ms | Blur + sharpen |
| Contour | ~1400ms | Edge detection (**SLOWEST**) |
| Publisher | ~100ms | Aggregate results |

## Execution Modes

### CLASSIC Mode
- Fan-in triggered by the **SLOWEST** branch (Contour)
- Publisher waits for all branches to complete before starting
- Total latency = ImageLoader + Contour + Publisher

### FUTURE_BASED Mode
- Fan-in triggered by the **FASTEST** branch (Thumbnail)
- Publisher starts early, other results pre-resolved in background
- Total latency = ImageLoader + Thumbnail + Publisher
- **Expected improvement: 1300ms (Contour - Thumbnail)**

## Metrics Collected

| Metric | Description |
|--------|-------------|
| E2E Latency | Time from invoke to completion |
| Per-Function Duration | CloudWatch REPORT logs |
| Billed Duration | For cost calculation |
| Cold Start Duration | Init Duration from logs |
| Memory Usage | Max Memory Used |
| Invoker Branch | Which branch triggered Publisher |
| Pre-resolved Count | Branches already complete (FUTURE benefit) |
| Branch Variance | Max - Min branch duration |

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
