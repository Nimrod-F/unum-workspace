# Graph Analysis Benchmark Results

## Configuration
- **Graph Size**: 200 nodes
- **Edge Probability**: 0.4
- **Iterations**: 3 per mode (cold starts)
- **Region**: eu-central-1
- **Date**: 2026-01-29

## Execution Modes

| Mode | Description |
|------|-------------|
| **CLASSIC** | Synchronous fan-in - last branch to checkpoint invokes Aggregator |
| **FUTURE_BASED** | Async fan-in with eager claiming - first branch to finish claims and invokes Aggregator |

---

## Key Results

| Metric | CLASSIC | FUTURE_BASED | Improvement |
|--------|---------|--------------|-------------|
| **E2E Latency (mean)** | 1975.6ms | 1927.7ms | **+2.4%** ✅ |
| **E2E Latency (min)** | 1938.1ms | 1876.2ms | +3.2% |
| **E2E Latency (max)** | 2018.9ms | 1984.7ms | +1.7% |
| **E2E Latency (std)** | 40.7ms | 54.4ms | - |
| **Total Cold Starts** | 15 | 15 | 0% |
| **Total Init Time** | 15212ms | 14807ms | **+2.7%** ✅ |

---

## Resource Metrics

| Metric | CLASSIC | FUTURE_BASED | Difference |
|--------|---------|--------------|------------|
| **Avg Billed Duration** | 8037.3ms | 8047.7ms | +0.1% |
| **Max Memory Used** | 92MB | 92MB | 0% |
| **Total Memory (all funcs)** | 455MB | 447MB | -1.8% |
| **Aggregator Memory** | 89MB | 87MB | **-3.0%** ✅ |
| **Memory Efficiency** | 71.1% | 69.8% | -1.3% |
| **Avg Cost per Run** | $0.0000177 | $0.0000178 | +0.1% |
| **Total Cost** | $0.0000532 | $0.0000533 | +0.1% |

---

## Invoker Distribution

Shows which branch triggered the Aggregator function:

| Mode | Run 1 | Run 2 | Run 3 | Pattern |
|------|-------|-------|-------|---------|
| **CLASSIC** | BFS | MST | MST | Varies (last to finish) |
| **FUTURE_BASED** | PageRank | PageRank | PageRank | Consistent (first to claim) |

---

## Per-Function Duration (Run 1)

| Function | CLASSIC | FUTURE_BASED |
|----------|---------|--------------|
| GraphGenerator | 713.6ms | 713.6ms |
| PageRank | 608.4ms | 794.8ms |
| BFS | 728.1ms | 578.2ms |
| MST | 508.8ms | 580.2ms |
| Aggregator | 510.6ms | 433.8ms |

---

## Charts

### E2E Latency Comparison
![E2E Latency](https://mdn.alipayobjects.com/one_clip/afts/img/R1sIR7s3o_4AAAAAReAAAAgAoEACAQFr/original)

### Mean Latency
![Mean Latency](https://mdn.alipayobjects.com/one_clip/afts/img/VaNFQ6Vlm28AAAAARLAAAAgAoEACAQFr/original)

### Memory Usage
![Memory](https://mdn.alipayobjects.com/one_clip/afts/img/-bj1TpP4PYIAAAAARZAAAAgAoEACAQFr/original)

### Per-Function Duration
![Per-Function](https://mdn.alipayobjects.com/one_clip/afts/img/IeFjQIYwS2gAAAAARnAAAAgAoEACAQFr/original)

### FUTURE_BASED Improvements
![Improvements](https://mdn.alipayobjects.com/one_clip/afts/img/BygwTIz1-H0AAAAARJAAAAgAoEACAQFr/original)

### Performance Radar
![Radar](https://mdn.alipayobjects.com/one_clip/afts/img/WaWiQpvf5tUAAAAASnAAAAgAoEACAQFr/original)

---

## Conclusions

1. **FUTURE_BASED mode shows +2.4% latency improvement** over CLASSIC mode with 200-node graphs
2. **Aggregator memory usage is 3% lower** in FUTURE mode (87MB vs 89MB)
3. **Invoker pattern differs significantly**:
   - CLASSIC: Variable (whichever branch finishes last)
   - FUTURE: Consistent (first branch to claim wins)
4. **Cost difference is negligible** (~0.1%)
5. The improvement would be **more pronounced with heterogeneous task durations** (e.g., one fast task, one slow task)

---

## Workflow Architecture

```
                    ┌─────────────┐
                    │   Input     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ GraphGen    │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │  PageRank   │  │    BFS      │  │    MST      │
   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Aggregator  │  ← Fan-in point
                    └─────────────┘
```

**Fan-in Behavior:**
- **CLASSIC**: Aggregator invoked by the LAST branch to checkpoint
- **FUTURE_BASED**: Aggregator invoked by the FIRST branch to claim (others skip)
