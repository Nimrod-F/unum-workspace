# Multi-Source Dashboard - Complete Deployment Guide

This guide provides step-by-step instructions for deploying and benchmarking the multi-source dashboard application in both CLASSIC and FUTURE_BASED modes.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [CLASSIC Mode Deployment](#classic-mode-deployment)
4. [FUTURE_BASED Mode Deployment](#future-based-mode-deployment)
5. [Running Benchmarks](#running-benchmarks)
6. [Interpreting Results](#interpreting-results)
7. [Generating Charts](#generating-charts)

---

## Prerequisites

### Software Requirements

- **UNUM CLI**: Latest version installed
- **AWS CLI**: Version 2.x configured with credentials
- **Python**: 3.8 or higher
- **SAM CLI**: (Optional) For manual deployments

### AWS Requirements

- **IAM Permissions**:
  - Lambda: Create, update, invoke functions
  - DynamoDB: Create tables, read/write items
  - CloudFormation: Create/update/delete stacks
  - CloudWatch: Read logs
  - IAM: Create/attach roles

- **AWS Region**: `eu-west-1` (or update in configs)

### Install Python Dependencies

```bash
cd unum-appstore/multi-source-dashboard/benchmark
pip install boto3 pyyaml
```

---

## Initial Setup

### 1. Create DynamoDB Table

```bash
aws dynamodb create-table \
    --table-name unum-intermediary-multi-dashboard \
    --attribute-definitions AttributeName=Name,AttributeType=S \
    --key-schema AttributeName=Name,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region eu-west-1
```

Verify creation:
```bash
aws dynamodb describe-table \
    --table-name unum-intermediary-multi-dashboard \
    --query 'Table.TableStatus'
```

### 2. Configure AWS Region (if not eu-west-1)

Edit `benchmark/config.yaml`:
```yaml
aws:
  region: us-east-1  # Change to your region
```

---

## CLASSIC Mode Deployment

### Step 1: Configure for CLASSIC Mode

Edit `unum-template.yaml` and ensure the MergeDashboardData function does NOT have UNUM_FUTURE_BASED set:

```yaml
MergeDashboardData:
  Properties:
    CodeUri: merge-dashboard-data/
    Runtime: python3.13
    MemorySize: 512
    Timeout: 120
    # No UNUM_FUTURE_BASED environment variable
```

### Step 2: Compile

```bash
cd unum-appstore/multi-source-dashboard

unum-cli compile \
    -p step-functions \
    -w unum-step-functions.json \
    -t unum-template.yaml
```

Expected output:
- `unum_config.json` files created in each function directory
- Workflow compiled successfully

### Step 3: Build

```bash
unum-cli build -g -p aws
```

This generates:
- `template.yaml` (AWS SAM template)
- Packages all Lambda functions

### Step 4: Deploy

```bash
unum-cli deploy -b
```

Or manually with SAM:
```bash
sam deploy \
    --template-file template.yaml \
    --stack-name multi-source-dashboard \
    --capabilities CAPABILITY_IAM \
    --region eu-west-1 \
    --resolve-s3
```

### Step 5: Verify CLASSIC Deployment

```bash
# List functions
aws cloudformation describe-stack-resources \
    --stack-name multi-source-dashboard \
    --query 'StackResources[?ResourceType==`AWS::Lambda::Function`].[LogicalResourceId,PhysicalResourceId]' \
    --output table

# Check aggregator config
AGG_ARN=$(aws cloudformation describe-stack-resources \
    --stack-name multi-source-dashboard \
    --query 'StackResources[?LogicalResourceId==`MergeDashboardDataFunction`].PhysicalResourceId' \
    --output text)

aws lambda get-function-configuration \
    --function-name $AGG_ARN \
    --query 'Environment.Variables'
```

Should show `EAGER` is not set or is `"false"`.

### Step 6: Test CLASSIC Deployment

```bash
# Get entry function
ENTRY_ARN=$(aws cloudformation describe-stack-resources \
    --stack-name multi-source-dashboard \
    --query 'StackResources[?LogicalResourceId==`TriggerDashboardFunction`].PhysicalResourceId' \
    --output text)

# Invoke
aws lambda invoke \
    --function-name $ENTRY_ARN \
    --invocation-type Event \
    --payload '{
      "Data": {
        "Source": "http",
        "Value": {"request_id": "classic-test-001"}
      },
      "Session": "classic-test-001"
    }' \
    response.json

# Wait 10 seconds, then check aggregator logs
sleep 10
aws logs tail /aws/lambda/$AGG_ARN --since 1m
```

Look for log indicating CLASSIC mode (no pre-resolved inputs).

---

## FUTURE_BASED Mode Deployment

### Option A: Update Existing Deployment

If you already deployed in CLASSIC mode, you can switch to FUTURE mode:

#### Step 1: Update Template

Edit `unum-template.yaml`:

```yaml
MergeDashboardData:
  Properties:
    CodeUri: merge-dashboard-data/
    Runtime: python3.13
    MemorySize: 512
    Timeout: 120
    Environment:
      UNUM_FUTURE_BASED: "true"  # Add this
```

#### Step 2: Rebuild and Redeploy

```bash
unum-cli build -g -p aws
unum-cli deploy -b
```

#### Step 3: Set Environment Variables for All Functions

The benchmark script does this automatically, but for manual setup:

```bash
# Get all function ARNs
aws cloudformation describe-stack-resources \
    --stack-name multi-source-dashboard \
    --query 'StackResources[?ResourceType==`AWS::Lambda::Function`].PhysicalResourceId' \
    --output text | tr '\t' '\n' > functions.txt

# Update each function
while read arn; do
  aws lambda update-function-configuration \
      --function-name $arn \
      --environment Variables={EAGER=true,UNUM_FUTURE_BASED=true}
  echo "Updated $arn"
done < functions.txt

# Wait for updates to complete
sleep 10
```

### Option B: Fresh Deployment

If deploying fresh for FUTURE mode:

1. Edit `unum-template.yaml` to include `UNUM_FUTURE_BASED: "true"` in `Globals.Function.Environment` or per function
2. Follow CLASSIC deployment steps 2-4

### Verify FUTURE_BASED Deployment

```bash
# Check aggregator environment
aws lambda get-function-configuration \
    --function-name $AGG_ARN \
    --query 'Environment.Variables'
```

Should show:
```json
{
  "EAGER": "true",
  "UNUM_FUTURE_BASED": "true",
  ...
}
```

### Test FUTURE_BASED Deployment

```bash
# Invoke
aws lambda invoke \
    --function-name $ENTRY_ARN \
    --invocation-type Event \
    --payload '{
      "Data": {
        "Source": "http",
        "Value": {"request_id": "future-test-001"}
      },
      "Session": "future-test-001"
    }' \
    response.json

# Check logs for pre-resolved metrics
sleep 10
aws logs filter-log-events \
    --log-group-name /aws/lambda/$AGG_ARN \
    --filter-pattern '"pre_resolved"' \
    --start-time $(date -u -d '1 minute ago' +%s)000
```

Look for `"pre_resolved": 4` or similar (should be > 0).

---

## Running Benchmarks

### Complete Benchmark (All Modes)

This is the recommended approach:

```bash
cd benchmark

# Run all modes with 10 iterations each
python run_benchmark.py --mode all --iterations 10

# This will:
# 1. Run CLASSIC mode (10 iterations, 3 cold, 7 warm)
# 2. Switch to EAGER mode (10 iterations, 3 cold, 7 warm)
# 3. Switch to FUTURE_BASED mode (10 iterations, 3 cold, 7 warm)
# 4. Save results to results/benchmark_<timestamp>.json
```

**Time required**: ~15-20 minutes

### Single Mode Benchmark

For testing or quick validation:

```bash
# CLASSIC only
python run_benchmark.py --mode CLASSIC --iterations 5

# FUTURE_BASED only
python run_benchmark.py --mode FUTURE_BASED --iterations 5
```

### Extended Benchmark (Higher Confidence)

For publication-quality results:

```bash
python run_benchmark.py \
    --mode all \
    --iterations 30 \
    --cold-iterations 5
```

**Time required**: ~45-60 minutes

### Custom Configuration

```bash
python run_benchmark.py \
    --mode FUTURE_BASED \
    --iterations 20 \
    --cold-iterations 10 \
    --output-file extended_future_test.json
```

---

## Interpreting Results

### Console Output

During benchmark, you'll see:

```
============================================================
Benchmark: CLASSIC mode
============================================================

Setting mode to CLASSIC...
  Discovered 8 functions

Cold start runs (3):
  Run 1/3... ✓ 4150ms (pre-resolved: 0)
  Run 2/3... ✓ 3980ms (pre-resolved: 0)
  Run 3/3... ✓ 4050ms (pre-resolved: 0)

Warm start runs (7):
  Run 1/7... ✓ 3820ms (pre-resolved: 0)
  Run 2/7... ✓ 3790ms (pre-resolved: 0)
  ...

CLASSIC Summary:
  Mean E2E: 3900ms ± 180ms
  Cold: 4060ms, Warm: 3800ms
  Pre-resolved avg: 0.0

============================================================
Benchmark: FUTURE_BASED mode
============================================================

...

FUTURE_BASED Summary:
  Mean E2E: 3200ms ± 150ms
  Cold: 3350ms, Warm: 3150ms
  Pre-resolved avg: 4.8

============================================================
Results saved to: results/benchmark_20240115_143022.json
============================================================
```

### Understanding Metrics

- **E2E (End-to-End) Latency**: Time from TriggerDashboard invocation to MergeDashboardData completion
- **Cold vs Warm**: Cold includes Lambda cold start overhead
- **Pre-resolved**: Number of inputs already available when accessed in aggregator (should be 0 in CLASSIC, 4-5 in FUTURE)

### Expected Results

| Metric | CLASSIC | FUTURE_BASED | Improvement |
|--------|---------|--------------|-------------|
| E2E Mean | 3800-4200ms | 3000-3400ms | 15-25% |
| Cold Mean | 4000-4500ms | 3300-3600ms | 18-22% |
| Warm Mean | 3700-4000ms | 2900-3200ms | 20-25% |
| Pre-resolved | 0 | 4-5 (of 6) | N/A |

### Analyzing Results

```bash
# Detailed analysis
python collect_metrics.py \
    --results-file results/benchmark_20240115_143022.json

# Output:
#
# Analyzing: results/benchmark_20240115_143022.json
#
# Summary:
#
# CLASSIC:
#   E2E: 3900ms ± 180ms
#   Cold: 4060ms
#   Warm: 3800ms
#   Pre-resolved: 0.0
#
# FUTURE_BASED:
#   E2E: 3200ms ± 150ms
#   Cold: 3350ms
#   Warm: 3150ms
#   Pre-resolved: 4.8
#
# Improvement: 17.9%
# Time saved: 700ms
```

### Export to CSV

```bash
python collect_metrics.py \
    --results-file results/benchmark_20240115_143022.json \
    --output-csv results/analysis.csv
```

CSV format:
```
Mode,Metric,Value
CLASSIC,E2E Mean (ms),3900.00
CLASSIC,E2E Std (ms),180.00
CLASSIC,Pre-resolved,0.00
FUTURE_BASED,E2E Mean (ms),3200.00
...
```

---

## Generating Charts

### Use Existing Chart Generator

```bash
# Copy results to central benchmark directory
cp multi-source-dashboard/benchmark/results/*.json \
   unum-appstore/benchmark/results/

# Generate charts
cd unum-appstore/benchmark
python generate_comparison_charts.py --results-dir results/
```

This generates:
- `e2e_latency_comparison.png` - Bar chart comparing modes
- `cold_warm_comparison.png` - Cold vs warm analysis
- `improvement_chart.png` - Percentage improvement
- `pre_resolved_efficiency.png` - Background polling efficiency

### Manual Chart Data Export

```bash
cd multi-source-dashboard/benchmark

python collect_metrics.py \
    --results-file results/benchmark_20240115_143022.json \
    --output-chart chart_data.json
```

Use `chart_data.json` with your preferred charting library.

---

## Validation Checklist

Before considering benchmark results valid:

- [ ] DynamoDB table exists and is accessible
- [ ] All 8 Lambda functions deployed successfully
- [ ] CLASSIC mode shows pre-resolved = 0
- [ ] FUTURE_BASED mode shows pre-resolved = 4-6
- [ ] At least 10 successful iterations per mode
- [ ] E2E latency variance (std dev) < 15% of mean
- [ ] FUTURE_BASED shows 15-25% improvement over CLASSIC
- [ ] No timeout errors or failures

---

## Cleanup

### Delete Everything

```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name multi-source-dashboard

# Wait for deletion
aws cloudformation wait stack-delete-complete --stack-name multi-source-dashboard

# Delete DynamoDB table
aws dynamodb delete-table --table-name unum-intermediary-multi-dashboard
```

### Clean Local Artifacts

```bash
cd unum-appstore/multi-source-dashboard
rm -rf .aws-sam/ template.yaml
find . -name "unum_config.json" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
```

### Archive Results

```bash
# Create archive
cd benchmark
tar -czf results-$(date +%Y%m%d).tar.gz results/

# Move archive
mv results-*.tar.gz ~/benchmark-archives/
```

---

## Troubleshooting

### Stack Creation Fails

**Error**: `CREATE_FAILED` with IAM permissions

**Solution**:
```bash
# Ensure you have admin permissions or add these policies:
# - AWSLambda_FullAccess
# - AmazonDynamoDBFullAccess
# - IAMFullAccess
# - CloudWatchLogsFullAccess
```

### Benchmark Script Can't Find Functions

**Error**: `Could not find entry function TriggerDashboard`

**Solution**:
```bash
# Verify stack name
aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE

# Update config.yaml if stack name differs
vim benchmark/config.yaml
```

### No Pre-Resolved Inputs in FUTURE Mode

**Error**: Pre-resolved count is 0 even in FUTURE mode

**Solution**:
```bash
# Verify environment variables
aws lambda get-function-configuration \
    --function-name <MergeDashboardData-ARN> \
    --query 'Environment.Variables'

# Should show EAGER=true and UNUM_FUTURE_BASED=true

# If not, update:
aws lambda update-function-configuration \
    --function-name <ARN> \
    --environment Variables={EAGER=true,UNUM_FUTURE_BASED=true}
```

### High Variance in Results

**Issue**: Standard deviation > 20% of mean

**Solutions**:
- Run more iterations (20-30)
- Ensure no other AWS load during benchmarks
- Check for Lambda throttling in CloudWatch
- Wait longer between runs (increase delay in config.yaml)

---

## Next Steps

1. **Integrate with CI/CD**: Add benchmark runs to your pipeline
2. **Extend the Application**: Add more data sources to increase parallelism
3. **Custom Analysis**: Build custom charts and reports from CSV exports
4. **Research Paper**: Use results to demonstrate Future-Based execution benefits

## Support

For issues:
1. Check `README.md` for detailed documentation
2. Review `UNUM_ARCHITECTURE_AND_FUTURES.md` for architecture details
3. Examine CloudWatch logs for errors
4. Verify AWS permissions and quotas
