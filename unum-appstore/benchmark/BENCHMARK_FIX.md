# Benchmark Script Fixed ✅

## Issue Found

The benchmark script crashed with:
```
StatisticsError: mean requires at least one data point
```

When trying to compute memory metrics on empty data.

## Fix Applied

**File**: `run_all_benchmarks.py` (Lines 655-664)

**Before**:
```python
summary.avg_aggregator_memory_mb = statistics.mean([r.aggregator_memory_mb for r in successful if r.aggregator_memory_mb > 0])
# Crashed if list is empty
```

**After**:
```python
agg_mem = [r.aggregator_memory_mb for r in successful if r.aggregator_memory_mb > 0]
summary.avg_aggregator_memory_mb = statistics.mean(agg_mem) if agg_mem else 0
# Returns 0 if no data
```

Applied to all 4 memory metrics:
- `avg_max_memory_mb`
- `avg_total_memory_mb`
- `avg_aggregator_memory_mb`
- `avg_memory_efficiency`

## Status

✅ Benchmark should now complete without crashing

You can re-run:
```bash
python run_all_benchmarks.py --workflow multi-source-dashboard --mode all --iterations 10
```

## Remaining Issues to Investigate

Based on your output, there are still some concerns:

### 1. High Latency (11-14 seconds)

**Expected**: 3-4 seconds for CLASSIC mode
**Actual**: 11-14 seconds

**Possible causes**:
- Lambda functions still have the import error (cfn-flip)
- Functions are timing out and retrying
- Very slow network/DynamoDB access

**Verify**:
```bash
# Check if Lambda has correct dependencies
FUNC_ARN=$(aws cloudformation describe-stack-resources \
    --stack-name multi-source-dashboard \
    --query 'StackResources[?LogicalResourceId==`TriggerDashboardFunction`].PhysicalResourceId' \
    --output text --region eu-central-1)

# Test invoke
aws lambda invoke \
    --function-name "$FUNC_ARN" \
    --region eu-central-1 \
    --payload '{"Data":{"Source":"http","Value":{"request_id":"test"}},"Session":"test"}' \
    test-response.json

# Check for errors
cat test-response.json
```

### 2. Pre-Resolved Shows 6 in CLASSIC Mode

**Expected**: 0 pre-resolved in CLASSIC mode
**Actual**: 6 pre-resolved

**This is incorrect**. In CLASSIC mode, no inputs should be pre-resolved.

**Possible causes**:
- Metric collection bug in the benchmark script
- Logs don't have `pre_resolved` field, so it defaults to 6
- UNUM runtime isn't logging the metric

**Check CloudWatch logs**:
```bash
aws logs filter-log-events \
    --log-group-name /aws/lambda/$FUNC_ARN \
    --filter-pattern '"pre_resolved"' \
    --region eu-central-1 \
    --start-time $(($(date +%s) - 3600))000 \
    --limit 5
```

## Next Steps

1. **Verify Lambda deployment** - Make sure you rebuilt and redeployed with cfn-flip in requirements.txt
2. **Check logs** - Verify functions are executing without errors
3. **Test manually** - Run a single invocation and check the result

If latencies are still high after rebuild, there may be other issues with the workflow execution.
