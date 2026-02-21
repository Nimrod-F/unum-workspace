# Order Processing Workflow - Deployment Commands

Quick reference for all deployment commands. Copy and paste into your terminal.

## Prerequisites

Ensure you're in the workflow directory:
```bash
cd A:\Disertatie\unum-workspace\unum-appstore\order-processing-workflow
```

## Step-by-Step Deployment

### 1. Create DynamoDB Table

```bash
aws dynamodb create-table --table-name unum-intermediate-datastore-orders --attribute-definitions AttributeName=id,AttributeType=S --key-schema AttributeName=id,KeyType=HASH --billing-mode PAY_PER_REQUEST --region eu-central-1
```

**Note**: If table already exists, you'll get an error. That's OK - skip to step 2.

### 2. Compile Step Functions Workflow

```bash
python ..\..\unum\unum-cli\unum-cli.py compile -p step-functions -w unum-step-functions.json -t unum-template.yaml
```

**What this does**: Generates `unum_config.json` files for each Lambda function based on the Step Functions definition.

### 3. Generate AWS SAM Template

```bash
python ..\..\unum\unum-cli\unum-cli.py template -t unum-template.yaml -p aws
```

**What this does**: Creates `template.yaml` from `unum-template.yaml`.

### 4. Build Workflow

```bash
python ..\..\unum\unum-cli\unum-cli.py build -t unum-template.yaml
```

**What this does**: Copies Unum runtime files to each function directory and runs `sam build`.

### 5. Deploy to AWS

```bash
python ..\..\unum\unum-cli\unum-cli.py deploy -t unum-template.yaml
```

**What this does**: Deploys all Lambda functions to AWS and creates `function-arn.yaml`.

## Verify Deployment

### Check Lambda Functions

```bash
aws lambda list-functions --region eu-central-1 | grep order-processing
```

Expected output:
- order-processing-workflow-TriggerFunction
- order-processing-workflow-FastProcessor
- order-processing-workflow-SlowChainStart
- order-processing-workflow-SlowChainMid
- order-processing-workflow-SlowChainEnd
- order-processing-workflow-Aggregator

### Test Workflow

```bash
aws lambda invoke --function-name order-processing-workflow-TriggerFunction --payload "{\"order_id\":\"test-001\",\"customer_id\":\"CUST-TEST\",\"items\":[{\"sku\":\"ITEM-001\",\"quantity\":2}]}" --region eu-central-1 response.json
```

Then check the response:
```bash
type response.json
```

## Run Benchmarks

### Option 1: Complete Benchmark Suite (Recommended)

Runs both modes, collects metrics, and generates charts:

```bash
cd benchmark
python benchmark_complete.py --iterations 10
```

This will:
- Run both CLASSIC and FUTURE_BASED modes
- Collect detailed CloudWatch metrics
- Calculate performance improvements
- Generate comparison charts (PNG files)
- Save results to JSON files

**Output files** (in `benchmark/results/`):
- `benchmark_summary_<timestamp>.json` - Statistical summary
- `benchmark_runs_<timestamp>.json` - Detailed run data
- `e2e_latency_comparison.png` - Latency chart
- `aggregator_invocation_timing.png` - Invocation timing chart
- `per_function_duration.png` - Function performance chart
- `execution_timeline.png` - Timeline visualization

### Option 2: Quick Individual Mode Tests

```bash
cd benchmark
python run_benchmark.py --mode CLASSIC --iterations 5
python run_benchmark.py --mode FUTURE_BASED --iterations 5
```

### Option 3: Both Modes (Basic Script)

```bash
cd benchmark
python run_benchmark.py --all --iterations 5 --cold
```

## Common Issues

### Issue: "function-arn.yaml not found"

**Solution**: Deploy the workflow first (step 5 above). The `function-arn.yaml` file is created during deployment.

### Issue: "Table already exists"

**Solution**: Skip the DynamoDB table creation step. The table persists across deployments.

### Issue: "Function not found"

**Solution**:
1. Check if functions are deployed: `aws lambda list-functions --region eu-central-1 | grep order-processing`
2. Verify `function-arn.yaml` exists in the workflow directory
3. Re-run deployment: `python ..\..\unum\unum-cli\unum-cli.py deploy -t unum-template.yaml`

### Issue: Benchmark shows "CLASSIC" for both modes

**Solution**: The benchmark script modifies the Aggregator's `UNUM_FUTURE_BASED` environment variable. Wait 5-10 seconds between mode changes for Lambda to update.

## Clean Up (Optional)

To remove all AWS resources:

```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name order-processing-workflow --region eu-central-1

# Wait for deletion
aws cloudformation wait stack-delete-complete --stack-name order-processing-workflow --region eu-central-1

# Delete DynamoDB table
aws dynamodb delete-table --table-name unum-intermediate-datastore-orders --region eu-central-1
```

## Quick Redeploy

If you modify Lambda function code and want to redeploy:

```bash
# From workflow directory
python ..\..\unum\unum-cli\unum-cli.py build -t unum-template.yaml
python ..\..\unum\unum-cli\unum-cli.py deploy -t unum-template.yaml
```

## All Commands (Copy-Paste)

```bash
# Navigate to workflow directory
cd A:\Disertatie\unum-workspace\unum-appstore\order-processing-workflow

# 1. Create DynamoDB table (skip if exists)
aws dynamodb create-table --table-name unum-intermediate-datastore-orders --attribute-definitions AttributeName=id,AttributeType=S --key-schema AttributeName=id,KeyType=HASH --billing-mode PAY_PER_REQUEST --region eu-central-1

# 2. Compile Step Functions
python ..\..\unum\unum-cli\unum-cli.py compile -p step-functions -w unum-step-functions.json -t unum-template.yaml

# 3. Generate SAM template
python ..\..\unum\unum-cli\unum-cli.py template -t unum-template.yaml -p aws

# 4. Build
python ..\..\unum\unum-cli\unum-cli.py build -t unum-template.yaml

# 5. Deploy
python ..\..\unum\unum-cli\unum-cli.py deploy -t unum-template.yaml

# 6. Test
aws lambda invoke --function-name order-processing-workflow-TriggerFunction --payload "{\"order_id\":\"test-001\",\"customer_id\":\"CUST-TEST\",\"items\":[{\"sku\":\"ITEM-001\",\"quantity\":2}]}" --region eu-central-1 response.json

# 7. Run benchmarks
cd benchmark
python run_benchmark.py --all --iterations 5 --cold
```
