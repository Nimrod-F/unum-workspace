# Region Configuration Fixed

## Issue
The initial configuration had mixed regions:
- ❌ Created DynamoDB table in `eu-west-1`
- ✅ Lambda functions deployed in `eu-central-1`

This caused the benchmark to timeout because Lambdas couldn't reach the DynamoDB table.

## What Was Fixed

### ✅ DynamoDB Table
- **Before**: Created in `eu-west-1` (wrong region)
- **After**: Using existing table in `eu-central-1`
- **Status**: Table `unum-intermediary-multi-dashboard` is ACTIVE in `eu-central-1`

### ✅ Configuration Files Updated

1. **unum-template.yaml**
   - Added `Region: eu-central-1` to Globals section

2. **benchmark/config.yaml**
   - Already correctly set to `region: eu-central-1`

### ✅ Requirements Files Fixed

All 8 `requirements.txt` files updated with:
```
cfn-flip
boto3
```

## Current State

**Everything is now configured for `eu-central-1`:**

✅ DynamoDB table: `unum-intermediary-multi-dashboard` (ACTIVE)
✅ Lambda functions: Deployed in `eu-central-1`
✅ Configuration: `unum-template.yaml` has `Region: eu-central-1`
✅ Benchmark: `config.yaml` has `region: eu-central-1`
✅ Dependencies: All functions have correct `requirements.txt`

## Next Steps

**Rebuild and redeploy** with the fixed configuration:

```bash
cd unum-appstore/multi-source-dashboard

# Rebuild with updated dependencies and region config
unum-cli build -g -p aws

# Redeploy to eu-central-1
unum-cli deploy -b
```

## Verification Commands

After redeployment, verify everything is in the correct region:

```bash
# Check Lambda region
aws lambda get-function-configuration \
    --function-name multi-source-dashboard-TriggerDashboardFunction-XXX \
    --region eu-central-1 \
    --query 'FunctionArn'

# Check DynamoDB region
aws dynamodb describe-table \
    --table-name unum-intermediary-multi-dashboard \
    --region eu-central-1 \
    --query 'Table.TableArn'

# Both should show eu-central-1 in the ARN
```

## What Will Work After Redeployment

✅ Lambdas will access DynamoDB in the same region (low latency)
✅ Checkpoints will be written to DynamoDB
✅ Benchmark will run without timeouts
✅ You'll see items in DynamoDB after running the workflow
