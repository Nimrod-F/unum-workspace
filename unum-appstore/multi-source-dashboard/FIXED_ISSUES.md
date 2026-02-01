# Fixed Issues

## Issue 1: Missing DynamoDB Table ✅ FIXED

**Problem**: The benchmark was timing out because the DynamoDB table `unum-intermediary-multi-dashboard` didn't exist.

**Solution**: Created the table with:
```bash
aws dynamodb create-table \
    --table-name unum-intermediary-multi-dashboard \
    --attribute-definitions AttributeName=Name,AttributeType=S \
    --key-schema AttributeName=Name,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region eu-west-1
```

**Status**: ✅ Table created and ACTIVE

## Issue 2: Missing Python Dependencies ✅ FIXED

**Problem**: Lambda functions were failing with error:
```
Unable to import module 'main': No module named 'cfn_tools'
```

**Root Cause**: The `requirements.txt` files were empty or didn't include the required UNUM dependencies.

**Solution**: Updated all `requirements.txt` files to include:
```
cfn-flip
boto3
```

**Status**: ✅ All requirements.txt files updated

## Next Steps

Now you need to **rebuild and redeploy** the application:

```bash
cd unum-appstore/multi-source-dashboard

# Rebuild with updated dependencies
unum-cli build -g -p aws

# Redeploy
unum-cli deploy -b
```

This will package the functions with the correct dependencies and redeploy them.

## Verification After Redeployment

Once redeployed, test with:

```bash
# Test manual invocation
FUNC_ARN=$(aws cloudformation describe-stack-resources \
    --stack-name multi-source-dashboard \
    --query 'StackResources[?LogicalResourceId==`TriggerDashboardFunction`].PhysicalResourceId' \
    --output text)

aws lambda invoke \
    --function-name "$FUNC_ARN" \
    --payload '{"Data":{"Source":"http","Value":{"request_id":"test-001"}},"Session":"test-001"}' \
    --cli-binary-format raw-in-base64-out \
    response.json

# Check response (should not have errorMessage)
cat response.json

# Check DynamoDB table (should have items)
aws dynamodb scan \
    --table-name unum-intermediary-multi-dashboard \
    --max-items 5
```

## Expected After Fix

✅ Lambda functions should execute without import errors
✅ DynamoDB table should contain checkpoints
✅ Benchmark should run without timeouts
✅ You should see items in DynamoDB like:
  - TriggerDashboard-xxx
  - FetchSalesData-xxx
  - FetchInventoryData-xxx
  - etc.

## Files Modified

- All 8 `requirements.txt` files updated with:
  ```
  cfn-flip
  boto3
  ```
