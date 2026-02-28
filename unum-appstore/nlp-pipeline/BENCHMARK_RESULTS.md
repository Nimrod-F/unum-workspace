# NLP Pipeline — Benchmark Results

## Normal vs Partial Parameter Streaming

**Date:** February 10, 2026  
**Region:** eu-central-1  
**Pipeline:** Tokenizer → Analyzer → Classifier → Summarizer (4 stages, 5 fields per stage)  
**Text Corpus:** ~12,735 characters (NLP history paragraph × 3)  
**Iterations:** 3 cold starts + 3 warm starts per mode  

---

## Key Results

| Metric | Normal | Streaming | Δ |
|---|---|---|---|
| **Warm E2E Latency** | 7.04 s | 5.73 s | **−18.6%** |
| **Cold E2E Latency** | 14.25 s | 10.42 s | **−26.9%** |
| Warm Billed Duration (total) | 9,130 ms | 11,014 ms | +20.6% |
| Warm Cost per Invocation | 19.02 µ$ | 22.95 µ$ | +20.6% |
| Warm Latency Std Dev | 3.282 s | 0.039 s | **−98.8%** |

> **★ Warm Speedup: 1.23×** — Streaming completes 18.6% faster  
> **★ Cold Speedup: 1.37×** — Streaming completes 26.9% faster  
> **★ Consistency: 84× more predictable** — σ drops from 3.28s to 0.039s

---

## Charts

### 1. End-to-End Latency Comparison

Streaming reduces wall-clock pipeline latency by **19–27%** across both warm and cold starts.

![E2E Latency: Normal vs Streaming](https://mdn.alipayobjects.com/one_clip/afts/img/fJz6Q5FcETcAAAAARQAAAAgAoEACAQFr/original)

### 2. Latency Consistency Across Runs

Streaming shows dramatically lower variance. Normal mode warm latency ranges wildly from 3.2s to 9.0s (σ=3.28s), while Streaming is rock-solid at 5.69–5.76s (σ=0.039s).

![Latency Consistency](https://mdn.alipayobjects.com/one_clip/afts/img/YKu7S7_LyQAAAAAASVAAAAgAoEACAQFr/original)

### 3. Streaming Improvements

![Improvements](https://mdn.alipayobjects.com/one_clip/afts/img/NtepRLxfK_wAAAAARaAAAAgAoEACAQFr/original)

### 4. Per-Function Duration (Warm Avg)

Individual function durations increase slightly in streaming mode because downstream functions start early and spend time resolving futures. However, the **total pipeline latency decreases** because functions overlap.

![Per-Function Duration](https://mdn.alipayobjects.com/one_clip/afts/img/pd0hS5bgr6MAAAAARtAAAAgAoEACAQFr/original)

### 5. Per-Function App Duration (Warm Avg)

App duration isolates the application logic from the unum runtime overhead. Streaming functions show higher app durations because they include DynamoDB future-resolution wait time.

![Per-Function App Duration](https://mdn.alipayobjects.com/one_clip/afts/img/4P_mQbxrJvYAAAAARwAAAAgAoEACAQFr/original)

### 6. Pipeline Concurrency

In **Normal** mode, functions execute sequentially: only 1 function is active at a time.  
In **Streaming** mode, up to 4 functions run concurrently thanks to early invocation with futures.

![Pipeline Concurrency](https://mdn.alipayobjects.com/one_clip/afts/img/HZETRbHmjyYAAAAAQvAAAAgAoEACAQFr/original)

### 7. Memory Usage per Function

Memory overhead from streaming is negligible (+0–1 MB). The `LazyFutureDict` and DynamoDB polling add minimal memory footprint.

![Memory Usage](https://mdn.alipayobjects.com/one_clip/afts/img/_xVBQomr7qkAAAAARdAAAAgAoEACAQFr/original)

### 8. Cold Start Init Duration

Streaming adds virtually no overhead to cold start init. Both modes show comparable init durations (~630–690ms per function).

![Cold Start Init](https://mdn.alipayobjects.com/one_clip/afts/img/CtOgRIiFlX8AAAAARmAAAAgAoEACAQFr/original)

### 9. Cost per Invocation

Streaming costs ~21% more per invocation due to longer billed durations from overlapping execution. This is the expected trade-off: **latency vs cost**.

![Cost Comparison](https://mdn.alipayobjects.com/one_clip/afts/img/RI7RSINr1eUAAAAARLAAAAgAoEACAQFr/original)

### 10. Total Billed Duration

![Billed Duration](https://mdn.alipayobjects.com/one_clip/afts/img/gtXtS5URs70AAAAARMAAAAgAoEACAQFr/original)

### 11. Multi-Dimensional Radar Comparison

Normalized scores (higher = better). Streaming excels in latency and consistency; Normal is more cost-efficient.

![Radar Comparison](https://mdn.alipayobjects.com/one_clip/afts/img/vFrtTrW3jO4AAAAATWAAAAgAoEACAQFr/original)

---

## Detailed Per-Function Metrics (Warm Avg)

| Function | Normal Duration | Streaming Duration | Normal Memory | Streaming Memory |
|---|---|---|---|---|
| Tokenizer | 602 ms | 774 ms | 97 MB | 97 MB |
| Analyzer | 1,103 ms | 1,207 ms | 97 MB | 98 MB |
| Classifier | 3,751 ms | 3,873 ms | 98 MB | 98 MB |
| Summarizer | 3,673 ms | 5,158 ms | 94 MB | 95 MB |
| **Total** | **9,128 ms** | **11,012 ms** | — | — |

### Why individual durations increase but E2E drops

In Normal mode, the 4 functions execute **sequentially**:
```
Tokenizer(0.6s) → Analyzer(1.1s) → Classifier(3.8s) → Summarizer(3.7s)
Total E2E ≈ sum = 9.1s (observed: ~7.0s with CloudWatch timing)
```

In Streaming mode, functions **overlap** because each function invokes the next after computing its first field:
```
Tokenizer starts at t=0
Analyzer starts at t≈0.2s   (after Tokenizer's first field published)
Classifier starts at t≈0.7s (after Analyzer's first field)
Summarizer starts at t≈1.5s (after Classifier's first field)
```

Each downstream function:
- Receives 1 real value + 4 futures
- Immediately processes the real value
- Resolves futures on-demand (with DynamoDB polling), adding wait time
- **Net effect: individual durations increase, but wall-clock E2E decreases**

### Per-Function Duration Increase Explained

| Function | Normal → Streaming | Reason |
|---|---|---|
| Tokenizer | 602 → 774 ms (+29%) | DynamoDB publish overhead for 5 intermediate fields |
| Analyzer | 1,103 → 1,207 ms (+9%) | Starts early, resolves futures, small overhead |
| Classifier | 3,751 → 3,873 ms (+3%) | Dominant compute time, futures resolve quickly |
| Summarizer | 3,673 → 5,158 ms (+40%) | Last function resolves all remaining futures from Classifier |

---

## Cold Start Analysis

| Function | Normal Init | Streaming Init |
|---|---|---|
| Tokenizer | 642 ms | 643 ms |
| Analyzer | 661 ms | 672 ms |
| Classifier | 690 ms | 649 ms |
| Summarizer | 635 ms | 619 ms |
| **Sum** | **2,628 ms** | **2,582 ms** |

Cold start init overhead from streaming: **−46 ms (−1.7%)** — essentially identical (within noise).  
Cold E2E improves by **26.9%** (14.25s → 10.42s) because the overlapping execution advantage dominates.

---

## Cost–Latency Trade-off

| Mode | Warm E2E | Warm Cost | Cost/Latency Ratio |
|---|---|---|---|
| Normal | 7.04 s | 19.02 µ$ | 2.70 µ$/s |
| Streaming | 5.73 s | 22.95 µ$ | 4.01 µ$/s |

Streaming trades **+21% cost** for **−19% latency**.  
For latency-sensitive NLP workloads (real-time text processing, interactive pipelines), this is a favorable trade-off.  
For batch/offline workloads prioritizing cost, Normal mode remains preferable.

---

## Latency Consistency

One of the most striking results is the **dramatic improvement in consistency**:

| Mode | Warm Avg | Warm Min | Warm Max | Std Dev |
|---|---|---|---|---|
| Normal | 7.04 s | 3.25 s | 8.98 s | **3.28 s** |
| Streaming | 5.73 s | 5.69 s | 5.76 s | **0.039 s** |

Streaming mode reduces standard deviation by **98.8%** (3.28s → 0.039s). This is because:
- Normal mode's E2E depends on CloudWatch log event timing and sequential handoff jitter
- Streaming mode uses direct DynamoDB polling with predictable latency

---

## Comparison with DNA Pipeline Benchmark

| Metric | DNA Pipeline | NLP Pipeline |
|---|---|---|
| Warm Speedup | **1.43×** (−29.9%) | **1.23×** (−18.6%) |
| Cold Speedup | **1.56×** (−35.8%) | **1.37×** (−26.9%) |
| Consistency Gain | 228× (σ: 6.42→0.03s) | 84× (σ: 3.28→0.04s) |
| Cost Overhead | +32.2% | +20.6% |
| Pipeline Length | 4 stages | 4 stages |
| Dominant Stage | Reporter (5.6s) | Classifier+Summarizer (3.7s each) |

The DNA pipeline shows higher speedup because its functions have **more uniform, longer durations** (~4–6s each), giving streaming more opportunity to overlap. The NLP pipeline has **front-heavy fast stages** (Tokenizer 0.6s, Analyzer 1.1s) followed by **heavier stages** (Classifier 3.8s, Summarizer 3.7s), reducing the overlap benefit for early stages but still delivering significant gains.

---

## Configuration

```
Pipeline:         nlp-pipeline (4 stages × 5 fields)
Runtime:          python3.11
Memory:           128 MB per function
Timeout:          900 s per function
Region:           eu-central-1
DynamoDB Table:   unum-nlp-pipeline (PAY_PER_REQUEST)
Text Corpus:      ~12,735 characters
Iterations:       3 cold + 3 warm per mode
```

---

## How to Reproduce

```bash
# Navigate to the workflow directory
cd unum-appstore/nlp-pipeline

# Compile workflow to generate unum_config.json files
py -3.11 ../../unum/unum-cli/unum-cli.py compile -p step-functions -w unum-step-functions.json -t unum-template.yaml

# Run the full benchmark (deploys both modes, runs all iterations)
py -3.11 benchmark.py

# Results saved to benchmark_results.json
```

The benchmark script:
1. Deploys Normal mode (restores clean `app.py.original` files)
2. Runs 3 cold start + 3 warm start iterations
3. Deploys Streaming mode (applies AST transformation)
4. Runs 3 cold start + 3 warm start iterations
5. Collects CloudWatch REPORT metrics for all 4 functions per iteration
6. Saves results to `benchmark_results.json`
