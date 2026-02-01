# Multi-Source Dashboard - Quick Start Guide

## TL;DR - Get Running in 5 Minutes

### 1. Create DynamoDB Table

```bash
aws dynamodb create-table \
    --table-name unum-intermediary-multi-dashboard \
    --attribute-definitions AttributeName=Name,AttributeType=S \
    --key-schema AttributeName=Name,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region eu-west-1
```

### 2. Deploy the Application

```bash
cd unum-appstore/multi-source-dashboard

# Compile, build, and deploy
unum-cli compile -p step-functions -w unum-step-functions.json -t unum-template.yaml
unum-cli build -g -p aws
unum-cli deploy -b
```

### 3. Run Benchmark

```bash
cd benchmark

# Edit config.yaml to set your AWS region
vim config.yaml  # Change region if needed

# Install dependencies
pip install boto3 pyyaml

# Run benchmark (takes ~15 minutes)
python run_benchmark.py --mode all --iterations 10
```

### 4. View Results

Results are saved to `benchmark/results/benchmark_<timestamp>.json`

Expected output:
```
CLASSIC Summary:
  Mean E2E: 3900ms ± 250ms
  Pre-resolved avg: 0.0

FUTURE_BASED Summary:
  Mean E2E: 3200ms ± 180ms
  Pre-resolved avg: 4.8

Improvement: 18-22%
```

## Command Reference

### Deployment

```bash
# CLASSIC mode (default)
unum-cli deploy -b

# For FUTURE_BASED mode, edit unum-template.yaml first:
# Set Environment: UNUM_FUTURE_BASED: "true" for MergeDashboardData
```

### Benchmark Commands

```bash
# All modes, 10 iterations
python run_benchmark.py --mode all --iterations 10

# Single mode, 20 iterations
python run_benchmark.py --mode FUTURE_BASED --iterations 20

# Custom cold/warm split
python run_benchmark.py --mode all --iterations 15 --cold-iterations 5

# Custom output file
python run_benchmark.py --mode CLASSIC --iterations 10 --output-file my_test.json
```

### Analysis Commands

```bash
# Analyze results
python collect_metrics.py --results-file results/benchmark_20240115.json

# Export to CSV
python collect_metrics.py \
    --results-file results/benchmark_20240115.json \
    --output-csv results.csv

# Export chart data
python collect_metrics.py \
    --results-file results/benchmark_20240115.json \
    --output-chart chart_data.json

# Analyze all results in directory
python collect_metrics.py --analyze-all results/
```

### Manual Testing

```bash
# Get function ARN
FUNCTION_ARN=$(aws cloudformation describe-stack-resources \
    --stack-name multi-source-dashboard \
    --query 'StackResources[?LogicalResourceId==`TriggerDashboardFunction`].PhysicalResourceId' \
    --output text)

# Invoke
aws lambda invoke \
    --function-name $FUNCTION_ARN \
    --invocation-type Event \
    --payload '{"Data":{"Source":"http","Value":{"request_id":"test-001"}},"Session":"test-001"}' \
    response.json

# View logs
aws logs tail /aws/lambda/$FUNCTION_ARN --follow
```

### Mode Switching

The benchmark script handles this automatically, but for manual switching:

```bash
# Get all function names
aws cloudformation describe-stack-resources \
    --stack-name multi-source-dashboard \
    --query 'StackResources[?ResourceType==`AWS::Lambda::Function`].PhysicalResourceId' \
    --output text

# Set to FUTURE_BASED mode (repeat for each function)
aws lambda update-function-configuration \
    --function-name <FUNCTION_ARN> \
    --environment Variables={EAGER=true,UNUM_FUTURE_BASED=true}
```

## Monitoring

### View Aggregator Logs

```bash
# Get aggregator ARN
AGG_ARN=$(aws cloudformation describe-stack-resources \
    --stack-name multi-source-dashboard \
    --query 'StackResources[?LogicalResourceId==`MergeDashboardDataFunction`].PhysicalResourceId' \
    --output text)

# Tail logs
aws logs tail /aws/lambda/$AGG_ARN --follow --format short

# Filter for pre-resolved metrics
aws logs filter-log-events \
    --log-group-name /aws/lambda/$AGG_ARN \
    --filter-pattern '"pre_resolved"' \
    --start-time $(date -u -d '5 minutes ago' +%s)000
```

### Check DynamoDB

```bash
# View recent checkpoints
aws dynamodb scan \
    --table-name unum-intermediary-multi-dashboard \
    --max-items 10 \
    --output table
```

## Cleanup

```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name multi-source-dashboard

# Delete DynamoDB table
aws dynamodb delete-table --table-name unum-intermediary-multi-dashboard

# Clean local build artifacts
cd unum-appstore/multi-source-dashboard
rm -rf .aws-sam/ template.yaml */unum_config.json
```

## Troubleshooting

### "Stack not found" error

Check stack name in `benchmark/config.yaml` matches deployed stack:

```bash
aws cloudformation list-stacks \
    --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE \
    --query 'StackSummaries[].StackName'
```

### Functions timeout

Increase timeout in `unum-template.yaml`:

```yaml
Functions:
  MergeDashboardData:
    Properties:
      Timeout: 180  # Increase from 120
```

Then redeploy:
```bash
unum-cli build -g -p aws
unum-cli deploy -b
```

### No improvement in FUTURE mode

Verify environment variables are set:

```bash
aws lambda get-function-configuration \
    --function-name <MergeDashboardData-ARN> \
    --query 'Environment.Variables'
```

Should show:
```json
{
  "EAGER": "true",
  "UNUM_FUTURE_BASED": "true"
}
```

### Benchmark script errors

```bash
# Install missing dependencies
pip install boto3 pyyaml

# Check AWS credentials
aws sts get-caller-identity

# Verify Python version
python --version  # Should be 3.7+
```

## Next Steps

1. **View detailed results**: Check `benchmark/results/` for JSON files
2. **Generate charts**: Use the existing chart generator in `unum-appstore/benchmark/`
3. **Integrate with CI/CD**: Add benchmark runs to your pipeline
4. **Customize latencies**: Edit `app.py` files to adjust delays
5. **Add more sources**: Extend the workflow with additional parallel branches

## Key Files

- `unum-template.yaml` - Application configuration
- `unum-step-functions.json` - Workflow definition
- `merge-dashboard-data/app.py` - Fan-in aggregation logic
- `benchmark/config.yaml` - Benchmark settings
- `benchmark/run_benchmark.py` - Benchmark runner

## Support

See full documentation in `README.md` or check the UNUM architecture guide:
`UNUM_ARCHITECTURE_AND_FUTURES.md`
