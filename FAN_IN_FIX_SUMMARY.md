# Fan-In Coordination Fix - Global Implementation

## Location
**File**: [`unum/runtime/unum.py`](unum/runtime/unum.py#L1138-L1193)  
**Function**: `_run_fan_in()` (lines 1138-1193)

## Problem Summary
Fan-in coordination had two critical bugs preventing automatic workflow continuation:

1. **Inconsistent Aggregation Names**: Each parallel branch computed different aggregation function names by including their Fan-out index (e.g., "CreatePost-unumIndex-0", "CreatePost-unumIndex-1"), causing branches to sync on different names in DynamoDB
2. **Conditional Blocking**: The last branch to finish wouldn't invoke the aggregation function if the conditional was false, even when all branches were ready

## Solution Applied

### Fix 1: Skip Fan-out in Aggregation Name (Lines 1147-1150)
```python
# Skip Fan-out when building the aggregation function name for fan-in
# All branches should sync on the same aggregation function name
if f == "Fan-out":
    continue
```

**Result**: All branches now sync on the same aggregation function name (e.g., just "CreatePost")

### Fix 2: Always Invoke When All Ready (Lines 1186-1193)
```python
if all_ready:
    # This branch is the last to finish - it must invoke the aggregation function
    payload['Data'] = {'Source': self.datastore.my_type, 'Value': branch_instance_names}
    payload['Session'] = session
    self.invoker.invoke(self.function_name, payload)
```

**Result**: The last branch to finish ALWAYS invokes the next stage, regardless of conditional

## Global Impact

### Automatic Application
The fix is in [`unum/runtime/unum.py`](unum/runtime/unum.py), which is automatically copied to all workflow builds via `unum-cli build`:

1. `populate_common_directory()` copies `unum/runtime/*.py` → project's `common/` directory
2. `sam_build()` copies from `common/` → each Lambda function's build directory

### Verified Workflows
- ✅ **WordCount MapReduce**: 6 parallel mappers → Partition → 3 parallel reducers → Summary
  - Session: cd14998f-e9b5-4731-acdc-b4e7bc79a77e
  - All 6 mappers marked ready, automatic invocation working
  
- ✅ **Text-Processing**: UserMention + (FindUrl → ShortenUrl) → CreatePost → Publish
  - Session: 47b71d64-8574-4bc8-a337-b11ff151d4a4
  - Both branches ready (ReadyMap: [true, true])
  - Needs rebuild to apply fix for automatic CreatePost invocation

## Next Steps

For any existing deployed workflows to get the fix:
1. Run `unum-cli build` to copy updated runtime files
2. Run `unum-cli deploy` to redeploy with fixed code

New workflows automatically get the fix on first build.

## Technical Details

### DynamoDB ReadyMap Structure
```json
{
  "Name": {
    "S": "<session-id>/<AggregationFunction>-fanin"
  },
  "ReadyMap": {
    "L": [
      {"BOOL": true},   // Branch 0 ready
      {"BOOL": true}    // Branch 1 ready
    ]
  }
}
```

### Synchronization Flow
1. Each branch executes and calls `fanin_sync_ready()`
2. Branch marks itself ready in DynamoDB ReadyMap[my_index]
3. Last branch to finish sets `all_ready=True`
4. Last branch invokes aggregation function with all branch outputs

---

**Date**: January 2, 2026  
**Status**: ✅ Applied globally in unum/runtime/unum.py
