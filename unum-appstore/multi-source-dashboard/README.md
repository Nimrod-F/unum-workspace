# Multi-Source Data Aggregation Dashboard

A comprehensive UNUM application designed to benchmark and demonstrate the benefits of Future-Based fan-in execution. This application simulates fetching data from multiple sources in parallel with varying latencies, then aggregating the results into a unified dashboard.

## Architecture

```
                              ┌──> FetchSalesData (100-400ms) ────────────┐
                              │                                           │
                              ├──> FetchInventoryData (150-500ms) ────────┤
                              │                                           │
TriggerDashboard ──> Fan-Out ─┼──> FetchMarketingMetrics (200-800ms) ────┼──> MergeDashboardData
                              │                                           │
                              ├──> FetchExternalMarketData (500-3000ms) ──┤  (Fan-In Aggregation)
                              │          ↑ SLOWEST BRANCH                 │
                              ├──> FetchWeatherData (300-1500ms) ─────────┤
                              │                                           │
                              └──> FetchCompetitorPricing (400-2500ms) ───┘
```

### Key Characteristics

- **6 parallel data sources** with realistic, heterogeneous latencies
- **10x latency variance** between fastest (100ms) and slowest (3000ms) branches
- **Fan-in aggregation** with support for CLASSIC, EAGER, and FUTURE_BASED modes
- **Realistic mock data** representing common enterprise dashboard scenarios

## Data Sources

| Source | Description | Latency Range | Purpose |
|--------|-------------|---------------|---------|
| **FetchSalesData** | Internal sales database | 100-400ms | Fast internal query baseline |
| **FetchInventoryData** | Warehouse management system | 150-500ms | Internal system with moderate latency |
| **FetchMarketingMetrics** | Marketing analytics API | 200-800ms | Third-party API with variable latency |
| **FetchExternalMarketData** | External market data provider | 500-3000ms | **SLOWEST** - Critical slow path |
| **FetchWeatherData** | Weather API for operations | 300-1500ms | External API moderate-high latency |
| **FetchCompetitorPricing** | Web scraping competitor prices | 400-2500ms | High-latency scraping operation |

## Execution Modes

### CLASSIC Mode
```
All 6 branches complete → Last finisher invokes aggregator → Cold start → Execute
Total time: ~3500-4000ms (slowest branch + cold start + aggregation)
```

### EAGER Mode (LazyInput)
```
First branch complete → Invoke aggregator → time.sleep() polling → Execute when all ready
Total time: ~3200-3800ms (some cold start overlap)
```

### FUTURE_BASED Mode
```
First branch complete → Invoke aggregator (cold start overlaps) → asyncio.Event() waiting → Execute
Total time: ~3000-3200ms (cold start hidden, non-blocking waits)
Expected improvement: 15-25% over CLASSIC
```

## Project Structure

```
multi-source-dashboard/
├── unum-template.yaml                    # Application configuration
├── unum-step-functions.json              # Workflow definition
│
├── trigger-dashboard/                    # Entry point function
│   ├── app.py
│   └── requirements.txt
│
├── fetch-sales-data/                     # Fast internal source
│   ├── app.py
│   └── requirements.txt
│
├── fetch-inventory-data/                 # Internal system
│   ├── app.py
│   └── requirements.txt
│
├── fetch-marketing-metrics/              # Marketing API
│   ├── app.py
│   └── requirements.txt
│
├── fetch-external-market-data/           # SLOWEST - External API
│   ├── app.py
│   └── requirements.txt
│
├── fetch-weather-data/                   # Weather API
│   ├── app.py
│   └── requirements.txt
│
├── fetch-competitor-pricing/             # Web scraping
│   ├── app.py
│   └── requirements.txt
│
├── merge-dashboard-data/                 # Fan-in aggregation
│   ├── app.py                           # Supports sync + async modes
│   └── requirements.txt
│
└── benchmark/                            # Benchmark scripts
    ├── config.yaml                       # Benchmark configuration
    ├── run_benchmark.py                  # Main benchmark runner
    └── collect_metrics.py                # Metrics analysis
```

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **UNUM CLI** installed and configured
3. **Python 3.13** (or configured runtime)
4. **AWS CLI** configured with credentials

## Deployment

### Step 1: Create DynamoDB Table

```bash
aws dynamodb create-table \
    --table-name unum-intermediary-multi-dashboard \
    --attribute-definitions AttributeName=Name,AttributeType=S \
    --key-schema AttributeName=Name,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region eu-west-1
```

### Step 2: Compile the Application

```bash
cd unum-appstore/multi-source-dashboard

# Compile the Step Functions workflow to UNUM IR
unum-cli compile \
    -p step-functions \
    -w unum-step-functions.json \
    -t unum-template.yaml
```

This generates:
- `unum_config.json` for each function
- `template.yaml` (AWS SAM template)

### Step 3: Build for AWS

```bash
# Build the application with platform-specific templates
unum-cli build -g -p aws
```

This packages all Lambda functions with dependencies.

### Step 4: Deploy to AWS

```bash
# Deploy using SAM
unum-cli deploy -b
```

Or manually with SAM:

```bash
sam deploy \
    --template-file template.yaml \
    --stack-name multi-source-dashboard \
    --capabilities CAPABILITY_IAM \
    --region eu-west-1
```

### Step 5: Verify Deployment

```bash
# List deployed functions
aws cloudformation describe-stack-resources \
    --stack-name multi-source-dashboard \
    --query 'StackResources[?ResourceType==`AWS::Lambda::Function`].[LogicalResourceId,PhysicalResourceId]' \
    --output table
```

## Running the Application

### Manual Test

```bash
# Get the entry function ARN
FUNCTION_ARN=$(aws cloudformation describe-stack-resources \
    --stack-name multi-source-dashboard \
    --query 'StackResources[?LogicalResourceId==`TriggerDashboardFunction`].PhysicalResourceId' \
    --output text)

# Invoke the workflow
aws lambda invoke \
    --function-name $FUNCTION_ARN \
    --invocation-type Event \
    --payload '{
      "Data": {
        "Source": "http",
        "Value": {
          "request_id": "test-001",
          "dashboard_type": "executive",
          "time_range": "24h"
        }
      },
      "Session": "test-session-001"
    }' \
    response.json

# Check CloudWatch Logs for results
aws logs tail /aws/lambda/$FUNCTION_ARN --follow
```

## Benchmarking

### Configure Benchmark Settings

Edit `benchmark/config.yaml`:

```yaml
benchmark:
  iterations: 10
  cold_start_runs: 3

aws:
  region: eu-west-1
  stack_name: multi-source-dashboard
```

### Run Benchmarks

#### Benchmark All Modes

```bash
cd benchmark

python run_benchmark.py --mode all --iterations 10
```

This will:
1. Run 10 iterations for CLASSIC, EAGER, and FUTURE_BASED modes
2. Include 3 cold start runs per mode
3. Save results to `results/benchmark_<timestamp>.json`

#### Benchmark Single Mode

```bash
# CLASSIC mode
python run_benchmark.py --mode CLASSIC --iterations 5

# FUTURE_BASED mode only
python run_benchmark.py --mode FUTURE_BASED --iterations 20 --cold-iterations 5
```

#### Custom Output

```bash
python run_benchmark.py \
    --mode all \
    --iterations 15 \
    --cold-iterations 5 \
    --output-file my_results.json
```

### Analyze Results

```bash
# Analyze specific results file
python collect_metrics.py \
    --results-file results/benchmark_20240115.json \
    --output-csv results.csv \
    --output-chart chart_data.json

# Analyze all results in directory
python collect_metrics.py \
    --analyze-all results/ \
    --output-csv aggregated.csv
```

### Use with Existing Chart Generator

The benchmark output format is compatible with the existing chart generator:

```bash
# Copy results to benchmark directory
cp multi-source-dashboard/benchmark/results/*.json unum-appstore/benchmark/results/

# Generate charts
cd unum-appstore/benchmark
python generate_comparison_charts.py --results-dir results/
```

## Expected Results

### Performance Metrics

| Mode | E2E Latency (mean) | Cold Start Impact | Pre-Resolved Inputs |
|------|-------------------|-------------------|---------------------|
| **CLASSIC** | 3800-4200ms | Full cold start penalty | 0 |
| **EAGER** | 3400-3900ms | Partial overlap | 0 |
| **FUTURE_BASED** | 3000-3400ms | Hidden in wait time | 4-5 (out of 6) |

### Expected Improvement

- **Latency improvement**: 15-25% (FUTURE_BASED vs CLASSIC)
- **Time saved**: 600-900ms per request
- **Cold start hiding**: ~200-300ms cold start overlaps with waiting for slow branch

### Why This Demonstrates Benefits

1. **High latency variance** (10x): Slowest branch is 10x slower than fastest, creating opportunity for futures
2. **Multiple fast branches**: 3-4 branches complete early, demonstrating background polling efficiency
3. **Realistic scenario**: Common enterprise pattern of aggregating from multiple sources
4. **Measurable improvement**: Clear 15-25% latency reduction with futures

## Switching Between Modes

### CLASSIC Mode (All inputs ready before aggregation)

```bash
# Update environment variables via AWS Console or CLI
aws lambda update-function-configuration \
    --function-name <FUNCTION_NAME> \
    --environment Variables={EAGER=false,UNUM_FUTURE_BASED=false}
```

Apply to all functions in the stack.

### EAGER Mode (LazyInput with synchronous polling)

```bash
aws lambda update-function-configuration \
    --function-name <FUNCTION_NAME> \
    --environment Variables={EAGER=true,UNUM_FUTURE_BASED=false}
```

### FUTURE_BASED Mode (Async with asyncio.Event)

```bash
aws lambda update-function-configuration \
    --function-name <FUNCTION_NAME> \
    --environment Variables={EAGER=true,UNUM_FUTURE_BASED=true}
```

**Note**: The benchmark script automates this configuration switching.

## Monitoring and Debugging

### CloudWatch Logs

Each function logs structured JSON for metrics:

```json
{
  "event": "function_complete",
  "function": "FetchSalesData",
  "latency_ms": 245.3,
  "status": "success"
}
```

Aggregator logs pre-resolved metrics:

```json
{
  "event": "async_inputs_retrieved",
  "function": "MergeDashboardData",
  "pre_resolved": 5,
  "total_inputs": 6
}
```

### View Logs

```bash
# View aggregator logs
aws logs tail /aws/lambda/<MergeDashboardData-ARN> --follow --format short

# Filter for specific events
aws logs filter-log-events \
    --log-group-name /aws/lambda/<ARN> \
    --filter-pattern '"pre_resolved"'
```

### DynamoDB Checkpoints

```bash
# View checkpoints
aws dynamodb scan \
    --table-name unum-intermediary-multi-dashboard \
    --max-items 10
```

## Cost Analysis

Estimated costs per 1000 invocations (eu-west-1):

- **Lambda compute**: ~$0.15-0.25
- **DynamoDB**: ~$0.05-0.10
- **CloudWatch Logs**: ~$0.02-0.05

**Total**: ~$0.22-0.40 per 1000 dashboard requests

FUTURE_BASED mode provides 15-25% latency improvement with negligible cost difference (~1-2% memory overhead).

## Troubleshooting

### Functions timeout

- Increase timeout in `unum-template.yaml`
- Check CloudWatch Logs for errors
- Verify DynamoDB table exists and is accessible

### Benchmark fails to discover functions

- Ensure CloudFormation stack name matches `config.yaml`
- Check AWS credentials and permissions
- Verify stack is fully deployed

### No pre-resolved inputs in FUTURE mode

- Verify `UNUM_FUTURE_BASED=true` environment variable is set
- Check that `EAGER=true` is also set
- Ensure UNUM runtime version supports futures

### Inconsistent results

- Run more iterations (20-30) for statistical significance
- Ensure functions are warm before warm runs
- Check for AWS Lambda throttling or cold starts

## Integration with Existing Benchmarks

This application follows the same patterns as other UNUM benchmark apps:

1. Add to `WORKFLOWS` in `unum-appstore/benchmark/run_all_benchmarks.py`:

```python
'multi-source-dashboard': {
    'stack_name': 'multi-source-dashboard',
    'entry_function': 'TriggerDashboard',
    'terminal_function': 'MergeDashboardData',
    'dynamodb_table': 'unum-intermediary-multi-dashboard',
    'fan_in_sizes': [6],
    'expected_delays': [0.25, 0.325, 0.5, 1.75, 0.85, 1.45],
    'description': 'Multi-source dashboard: 6 parallel sources (100ms-3000ms)'
}
```

2. Run unified benchmarks:

```bash
cd unum-appstore/benchmark
python run_all_benchmarks.py --workflow multi-source-dashboard --mode all --iterations 10
```

## Contributing

To extend this application:

1. Add new data sources by creating new function directories
2. Update `unum-step-functions.json` to include new parallel branches
3. Update `unum-template.yaml` with new function definitions
4. Recompile and redeploy

## License

Part of the UNUM project. See main repository for license details.

## References

- [UNUM Architecture and Futures](../../UNUM_ARCHITECTURE_AND_FUTURES.md)
- [Benchmark Results](./benchmark/results/)
- [Chart Generation](../benchmark/generate_comparison_charts.py)
