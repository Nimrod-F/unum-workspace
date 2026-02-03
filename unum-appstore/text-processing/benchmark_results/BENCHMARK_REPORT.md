# Text-Processing Benchmark Results

## Workflow Description
- **Branch 0**: UserMention (1 step - FAST)
- **Branch 1**: FindUrl → ShortenUrl (2 steps - SLOWER)  
- **Fan-in**: CreatePost
- **Final**: Publish

## Summary

| Metric | CLASSIC | FUTURE_BASED | Improvement |
|--------|---------|--------------|-------------|
| Cold Start E2E (mean) | 4320ms | 3432ms | **20.6%** |
| Warm Start E2E (mean) | 4375ms | 2374ms | **45.7%** |
| CreatePost Delay (mean) | 676ms | 553ms | **18.2%** |

## Key Findings

### 1. Significant Warm Start Improvement (45.7%)
In FUTURE_BASED mode, warm start E2E latency is reduced by nearly half compared to CLASSIC mode. This is because:
- CreatePost is invoked immediately when the first branch (UserMention) completes
- The aggregator starts processing/waiting before the slower branch finishes

### 2. Cold Start Improvement (20.6%)
Cold start performance also improves because:
- CreatePost container starts warming up earlier
- Parallel initialization with the slower branch execution

### 3. CreatePost Invocation Delay (18.2% faster)
In FUTURE_BASED mode:
- **CLASSIC**: CreatePost waits for ALL branches → starts after ShortenUrl completes
- **FUTURE_BASED**: CreatePost starts when FIRST branch (UserMention) completes → starts much earlier

## Execution Pattern Comparison

### CLASSIC Mode
```
UserMention ──────────────────┐
                              ├──► CreatePost ──► Publish
FindUrl ──► ShortenUrl ───────┘
           (waits for both)
```

### FUTURE_BASED Mode
```
UserMention ──────────────────► CreatePost ──► Publish
                                (starts immediately, polls for missing)
FindUrl ──► ShortenUrl ────────► (data arrives while CreatePost running)
```

## Generated Charts

1. **cold_warm_comparison.png** - Side-by-side comparison of Cold vs Warm start latency
2. **createpost_delay.png** - Time from workflow start to CreatePost invocation
3. **improvement_chart.png** - Percentage improvement for each metric
4. **latency_distribution.png** - Box plot showing latency distribution

## Raw Data Files
- `classic_results.json` - Detailed per-iteration results for CLASSIC mode
- `future_results.json` - Detailed per-iteration results for FUTURE_BASED mode

---
**Conclusion**: Future-Based execution provides significant performance improvements, especially for warm starts where the improvement reaches **45.7%**. The key benefit is that the aggregator (CreatePost) starts processing earlier, reducing overall end-to-end latency.

---
Generated: 2026-02-03
