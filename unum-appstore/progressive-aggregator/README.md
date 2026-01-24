# Progressive Aggregator - Future-Based Demo

This example demonstrates when Future-Based mode is **faster** than Classic mode.

## The Scenario

5 data sources with **different latencies**:

- Source 1: 1 second
- Source 2: 2 seconds
- Source 3: 3 seconds
- Source 4: 4 seconds
- Source 5: 5 seconds

Each source returns data that must be aggregated.

## Why Future-Based Wins Here

### Classic Mode

1. Fan-out triggers 5 sources
2. Each source writes to DynamoDB when done
3. Aggregator polls DynamoDB until ALL 5 are ready
4. **Total wait: 5 seconds (slowest source) + polling overhead**

### Future-Based Mode

1. Fan-out triggers 5 sources
2. Each source returns a Future inline
3. Aggregator receives futures and can process incrementally:
   - After 1s: Process Source 1's data
   - After 2s: Process Source 2's data
   - ...
4. **Aggregator starts processing immediately as each future resolves**
5. **No DynamoDB writes/reads = lower latency**

## Key Difference

The Aggregator uses **incremental processing** - it doesn't need ALL inputs
to start working. It processes each input as it arrives, accumulating results.

```python
# Aggregator processes inputs as they resolve
total = 0
for i, future_input in enumerate(inputs):
    # This blocks only until THIS input is ready
    data = future_input  # Future resolves here
    total += process(data)
    print(f"Processed source {i+1}")
```

## Expected Results

| Mode         | E2E Latency | Why                                         |
| ------------ | ----------- | ------------------------------------------- |
| Classic      | ~6-7s       | Wait for slowest (5s) + DynamoDB polling    |
| Future-Based | ~5-5.5s     | Start processing at 1s, parallel resolution |

Future-Based saves:

- DynamoDB round-trip latency
- Polling overhead
- Can overlap processing with waiting
