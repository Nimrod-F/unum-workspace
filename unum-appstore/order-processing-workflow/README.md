# Order Processing Workflow

Benchmark workflow demonstrating the benefits of Unum's Future-Based execution over CLASSIC mode through a realistic e-commerce order processing scenario.

## Workflow Overview

```
TriggerFunction (Entry)
├─> FastProcessor (100ms) ────────────┐
└─> SlowChainStart (500ms)            │
    └─> SlowChainMid (1000ms)         ├─> Aggregator (Terminal)
        └─> SlowChainEnd (1500ms) ────┘
```

### Real-World Scenario: E-Commerce Order Processing

When a customer places an order:

1. **TriggerFunction**: Receives order, initiates parallel processing
2. **FastProcessor**: Quick inventory check (local cache lookup) - **100ms**
3. **SlowChainStart**: Payment validation (external API) - **500ms**
4. **SlowChainMid**: Shipping calculation (routing algorithm) - **1000ms**
5. **SlowChainEnd**: Invoice generation (PDF creation) - **1500ms**
6. **Aggregator**: Finalize order and send confirmation

**Total Sequential Chain Time**: 500ms + 1000ms + 1500ms = **3000ms**

## Execution Modes Comparison

### CLASSIC Mode
- Aggregator invoked by **SlowChainEnd** (last to complete)
- Timing: ~3100ms (after slow chain completes) + Cold start (200ms)
- **Total E2E**: ~3300ms

### FUTURE_BASED Mode
- Aggregator invoked by **FastProcessor** (first to complete)
- Timing: ~100ms (after fast branch) + Cold start overlaps with slow chain
- **Total E2E**: ~3100ms
- **Benefit**: Cold start hidden behind slow chain execution

### Expected Performance Improvement

| Metric | CLASSIC | FUTURE_BASED | Improvement |
|--------|---------|--------------|-------------|
| Aggregator Invocation | ~3100ms | ~100ms | **3000ms (96%)** |
| E2E Latency (cold) | ~3300ms | ~3100ms | **~200ms** |
| Invoker Branch | SlowChainEnd | FastProcessor | ✓ |

## Deployment Instructions

### Prerequisites
- AWS CLI configured (default profile or set `AWS_PROFILE` environment variable)
- Python 3.13
- Unum CLI tools

### Step 1: Create DynamoDB Table

```bash
cd A:\Disertatie\unum-workspace\unum-appstore\order-processing-workflow

aws dynamodb create-table \
  --table-name unum-intermediate-datastore-orders \
  --attribute-definitions AttributeName=id,AttributeType=S \
  --key-schema AttributeName=id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region eu-central-1
```

### Step 2: Compile Step Functions Workflow

```bash
python ..\..\unum\unum-cli\unum-cli.py compile -p step-functions -w unum-step-functions.json -t unum-template.yaml
```

This generates `unum_config.json` files for each function based on the Step Functions definition.

### Step 3: Generate AWS SAM Template

```bash
python ..\..\unum\unum-cli\unum-cli.py template -t unum-template.yaml -p aws
```

This creates `template.yaml` from `unum-template.yaml`.

### Step 4: Build Workflow

```bash
python ..\..\unum\unum-cli\unum-cli.py build -t unum-template.yaml
```

This copies Unum runtime files to each function directory and runs `sam build`.

### Step 5: Deploy to AWS

```bash
python ..\..\unum\unum-cli\unum-cli.py deploy -t unum-template.yaml
```

This deploys all Lambda functions and creates the workflow infrastructure.

## Running Benchmarks

### Complete Benchmark Suite (Recommended)

The complete benchmark script automatically runs both modes, collects detailed metrics, and generates comparison charts:

```bash
cd benchmark
python benchmark_complete.py --iterations 10
```

**Features:**
- Runs both CLASSIC and FUTURE_BASED modes automatically
- Collects comprehensive CloudWatch metrics
- Calculates detailed performance improvements
- Generates comparison charts (PNG files)
- Saves results to JSON for further analysis

**Options:**
- `--iterations N`: Number of iterations per mode (default: 5)
- `--cold-all`: Force cold starts for all iterations (default: only first)
- `--skip-classic`: Skip CLASSIC mode benchmark
- `--skip-future`: Skip FUTURE_BASED mode benchmark
- `--skip-charts`: Skip chart generation

**Charts Generated:**
1. `e2e_latency_comparison.png` - End-to-end latency comparison
2. `aggregator_invocation_timing.png` - When Aggregator is invoked
3. `per_function_duration.png` - Individual function performance
4. `execution_timeline.png` - Visual timeline of both modes

### Basic Benchmark Script (Alternative)

For quick testing of individual modes:

```bash
cd benchmark
python run_benchmark.py --mode CLASSIC --iterations 5
python run_benchmark.py --mode FUTURE_BASED --iterations 5
python run_benchmark.py --all --iterations 5
```

**Options:**
- `--mode`: Choose `CLASSIC` or `FUTURE_BASED`
- `--all`: Run both modes sequentially
- `--iterations`: Number of workflow invocations (default: 5)
- `--cold`: Force cold starts for all iterations (default: only first)

## Verification

### 1. Check Deployed Functions

```bash
aws lambda list-functions --region eu-central-1 | grep order-processing
```

Expected output:
- `order-processing-workflow-TriggerFunction`
- `order-processing-workflow-FastProcessor`
- `order-processing-workflow-SlowChainStart`
- `order-processing-workflow-SlowChainMid`
- `order-processing-workflow-SlowChainEnd`
- `order-processing-workflow-Aggregator`

### 2. Verify DynamoDB Table

```bash
aws dynamodb describe-table --table-name unum-intermediate-datastore-orders --region eu-central-1
```

### 3. Test Workflow Invocation

```bash
aws lambda invoke \
  --function-name order-processing-workflow-TriggerFunction \
  --payload '{"order_id":"test-001","customer_id":"CUST-TEST","items":[{"sku":"ITEM-001","quantity":2}]}' \
  --region eu-central-1 \
  response.json

cat response.json
```

### 4. Monitor CloudWatch Logs

```bash
# Watch Aggregator logs
aws logs tail /aws/lambda/order-processing-workflow-Aggregator --follow --region eu-central-1

# Check for FUTURE_BASED mode indicator
aws logs filter-log-events \
  --log-group-name /aws/lambda/order-processing-workflow-Aggregator \
  --filter-pattern "FUTURE_BASED" \
  --region eu-central-1
```

## Key Metrics to Observe

### In Benchmark Results

1. **E2E Latency**: Total time from workflow start to completion
2. **Aggregator Invocation Time**: When Aggregator starts executing
3. **Invoker Branch**: Which function triggered the Aggregator
   - CLASSIC: Should be `SlowChainEnd`
   - FUTURE_BASED: Should be `FastProcessor`
4. **Cold Start Count**: Number of cold starts
5. **Pre-resolved Count**: How many inputs were ready when Aggregator started

### Expected Observations

**CLASSIC Mode:**
- Aggregator triggered after ~3100ms
- All inputs pre-resolved (blocking wait completed)
- Cold start adds ~200ms to E2E latency

**FUTURE_BASED Mode:**
- Aggregator triggered after ~100ms
- Initially 1 input ready (FastProcessor), awaits SlowChainEnd
- Cold start overlaps with slow chain execution
- ~200ms improvement in E2E latency

## Workflow Architecture

### Function Details

| Function | Role | Delay | Description |
|----------|------|-------|-------------|
| TriggerFunction | Entry | - | Receives order, fans out to 2 branches |
| FastProcessor | Fast Branch | 100ms | Quick inventory check |
| SlowChainStart | Sequential 1/3 | 500ms | Payment validation |
| SlowChainMid | Sequential 2/3 | 1000ms | Shipping calculation |
| SlowChainEnd | Sequential 3/3 | 1500ms | Invoice generation |
| Aggregator | Terminal | - | Merge results, finalize order |

### Data Flow

1. Customer places order
2. TriggerFunction creates payloads for both branches
3. **Parallel Execution:**
   - FastProcessor: Quick cache lookup (100ms)
   - SlowChainStart→Mid→End: Sequential processing (3000ms)
4. Aggregator waits for both branches (async in FUTURE mode)
5. Return order confirmation

## Troubleshooting

### Lambda Functions Not Found

```bash
# List CloudFormation stacks
aws cloudformation list-stacks --region eu-central-1

# Check if stack exists
aws cloudformation describe-stacks --stack-name order-processing-workflow --region eu-central-1
```

### DynamoDB Table Already Exists

```bash
# Delete existing table
aws dynamodb delete-table --table-name unum-intermediate-datastore-orders --region eu-central-1

# Wait for deletion, then recreate
```

### Benchmark Returns Empty Metrics

- Ensure CloudWatch logs are enabled
- Increase wait time in benchmark script (line with `time.sleep(5)`)
- Check Lambda execution role has CloudWatch Logs permissions

### Cold Starts Not Working

```bash
# Manually update a function to force cold start
aws lambda update-function-configuration \
  --function-name order-processing-workflow-Aggregator \
  --environment "Variables={UNUM_FUTURE_BASED=true,FORCE_COLD=$(date +%s)}" \
  --region eu-central-1
```

## Files Structure

```
order-processing-workflow/
├── trigger-function/
│   ├── app.py                    # Entry point lambda
│   └── unum_config.json          # Workflow configuration
├── fast-processor/
│   ├── app.py                    # Fast branch (100ms)
│   └── unum_config.json
├── slow-chain-start/
│   ├── app.py                    # Sequential step 1 (500ms)
│   └── unum_config.json
├── slow-chain-mid/
│   ├── app.py                    # Sequential step 2 (1000ms)
│   └── unum_config.json
├── slow-chain-end/
│   ├── app.py                    # Sequential step 3 (1500ms)
│   └── unum_config.json
├── aggregator/
│   ├── app.py                    # Terminal fan-in function
│   └── unum_config.json
├── benchmark/
│   ├── run_benchmark.py          # Benchmark runner
│   └── results/                  # Benchmark results (JSON)
├── unum-template.yaml            # Unum workflow configuration
├── unum-step-functions.json      # Step Functions definition
├── template.yaml                 # Generated AWS SAM template
├── function-arn.yaml             # Generated function ARNs
└── README.md                     # This file
```

## References

- [Unum Documentation](../../README.md)
- [Future-Based Implementation Guide](../../FUTURE_BASED_IMPLEMENTATION.md)
- [Image Pipeline Benchmark](../image-pipeline/benchmark/)
