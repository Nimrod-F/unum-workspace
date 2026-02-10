# Streaming Demo - Partial Parameter Streaming Benchmark

This workflow demonstrates the benefits of **Partial Parameter Streaming** in serverless workflows, showing how streaming output fields as they're computed can dramatically reduce end-to-end latency.

## 📊 Key Results

| Metric               | Normal Mode | Streaming Mode | Improvement       |
| -------------------- | ----------- | -------------- | ----------------- |
| **Warm Start E2E**   | 10.3s       | 4.4s           | **57% faster**    |
| **Cold Start E2E**   | 12.1s       | 8.2s           | **32% faster**    |
| **Throughput**       | 0.097 req/s | 0.227 req/s    | **134% increase** |
| **Peak Parallelism** | 1 function  | 4 functions    | **4x parallel**   |

### Latency Comparison

![E2E Latency Comparison](https://mdn.alipayobjects.com/one_clip/afts/img/kNk7QrZg4t0AAAAARSAAAAgAoEACAQFr/original)

### Improvement Summary

![Streaming Improvements](https://mdn.alipayobjects.com/one_clip/afts/img/QUc2RbnmeiUAAAAARLAAAAgAoEACAQFr/original)

---

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Generator  │───▶│  Processor  │───▶│  Analyzer   │───▶│  Reporter   │
│  (5 items)  │    │  (5 items)  │    │  (5 items)  │    │  (summary)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     2.5s              2.5s              2.5s              2.5s
```

### Data Flow

![Data Flow](https://mdn.alipayobjects.com/one_clip/afts/img/EczVTbERpmoAAAAAQlAAAAgAoEACAQFr/original)

### Key Design Principles

1. **Independent Fields**: Each stage produces 5 independent fields (item_1 through item_5)
2. **One-to-One Dependencies**: Each output depends on exactly ONE input field
   - `Processor.processed_1` depends only on `Generator.item_1`
   - `Analyzer.analyzed_1` depends only on `Processor.processed_1`
3. **Early Invocation**: After computing first field, invoke next stage with futures

---

## ⏱️ Execution Timeline

### Normal Mode (Sequential)

```
Time:  0s ─────────────────────────────────────────────────────▶ 10.3s

Generator:  ████████████████████
                              2.5s
Processor:                        ████████████████████
                                                    5.0s
Analyzer:                                               ████████████████████
                                                                          7.5s
Reporter:                                                                     ████████████████████
                                                                                               10.0s
```

### Streaming Mode (Parallel Pipeline)

```
Time:  0s ─────────────────────────────────▶ 4.4s

Generator:  ████████████████████
                              2.5s
Processor:   ████████████████████
                              3.0s
Analyzer:     ████████████████████
                               3.5s
Reporter:      ████████████████████
                                4.4s
```

### Active Functions Over Time

![Active Functions](https://mdn.alipayobjects.com/one_clip/afts/img/WTWgQJCojQgAAAAARsAAAAgAoEACAQFr/original)

### Concurrency Comparison

![Concurrency](https://mdn.alipayobjects.com/one_clip/afts/img/DCupT6mSGPsAAAAARPAAAAgAoEACAQFr/original)

---

## 💰 Cost Analysis

### Lambda Pricing (256 MB, eu-central-1)

- **Per 1ms**: $0.000000417
- **Per request**: $0.0000002

| Component    | Normal Mode | Streaming Mode | Difference |
| ------------ | ----------- | -------------- | ---------- |
| Generator    | $4.20e-6    | $4.31e-6       | +2.6%      |
| Processor    | $4.31e-6    | $4.35e-6       | +0.9%      |
| Analyzer     | $4.25e-6    | $4.32e-6       | +1.6%      |
| Reporter     | $4.19e-6    | $4.22e-6       | +0.7%      |
| DynamoDB I/O | $0          | $5e-7          | +$0.5µ     |
| **Total**    | **$17.2µ**  | **$17.7µ**     | **+2.9%**  |

### Cost Breakdown

![Cost Comparison](https://mdn.alipayobjects.com/one_clip/afts/img/DDFcTrU3ZhUAAAAARVAAAAgAoEACAQFr/original)

> **Key Insight**: Streaming adds only ~3% cost overhead while delivering 57% latency improvement.

---

## 🧠 Memory Usage

| Function    | Normal Mode  | Streaming Mode | Overhead |
| ----------- | ------------ | -------------- | -------- |
| Generator   | 68 MB        | 72 MB          | +6%      |
| Processor   | 65 MB        | 75 MB          | +15%     |
| Analyzer    | 66 MB        | 74 MB          | +12%     |
| Reporter    | 64 MB        | 64 MB          | 0%       |
| **Average** | **65.75 MB** | **71.25 MB**   | **+8%**  |

### Memory Comparison

![Memory Usage](https://mdn.alipayobjects.com/one_clip/afts/img/c41tSol1XZwAAAAARRAAAAgAoEACAQFr/original)

> **Note**: Streaming mode uses ~8% more memory due to:
>
> - LazyFutureDict wrapper objects
> - DynamoDB client (boto3)
> - Future resolution caching

---

## 🌐 Network Overhead

### Per-Operation Latency

| Operation           | Avg Latency | P99 Latency |
| ------------------- | ----------- | ----------- |
| DynamoDB Write      | 15 ms       | 45 ms       |
| DynamoDB Read       | 12 ms       | 35 ms       |
| Future Resolution   | 18 ms       | 50 ms       |
| Lambda Async Invoke | 45 ms       | 120 ms      |

### Network Overhead Breakdown

![Network Overhead](https://mdn.alipayobjects.com/one_clip/afts/img/DHDpTY-80lgAAAAARHAAAAgAoEACAQFr/original)

### Total Network Time per Workflow

| Mode      | Network I/O | % of E2E |
| --------- | ----------- | -------- |
| Normal    | ~45 ms      | 0.4%     |
| Streaming | ~120 ms     | 2.7%     |

> **Key Insight**: Network overhead in streaming mode is minimal (~2.7% of E2E) and is far outweighed by the parallelization benefits.

---

## ⏳ Per-Function Execution Time

Each function executes for approximately the same duration in both modes - the difference is in **when** they start.

### Function Duration Comparison

![Function Duration](https://mdn.alipayobjects.com/one_clip/afts/img/VQZWQaOIp7kAAAAARSAAAAgAoEACAQFr/original)

### Billed Duration

| Mode      | Total Billed | E2E Latency | Efficiency |
| --------- | ------------ | ----------- | ---------- |
| Normal    | 41,200 ms    | 10,300 ms   | 4.0x       |
| Streaming | 42,400 ms    | 4,400 ms    | 9.6x       |

![Billed Duration](https://mdn.alipayobjects.com/one_clip/afts/img/J3-OTJRbJL4AAAAARJAAAAgAoEACAQFr/original)

> **Key Insight**: While total billed duration is similar, streaming mode achieves 2.4x better efficiency (work done per wall-clock second).

---

## 📈 Streaming Mode Breakdown

### Latency Waterfall

![Latency Waterfall](https://mdn.alipayobjects.com/one_clip/afts/img/gneMQKAsseAAAAAARfAAAAgAoEACAQFr/original)

### Time Breakdown

![Time Breakdown](https://mdn.alipayobjects.com/one_clip/afts/img/sh1CQburomIAAAAARWAAAAgAoEACAQFr/original)

---

## 🎯 Multi-Metric Comparison

![Radar Comparison](https://mdn.alipayobjects.com/one_clip/afts/img/sKx8TaYQm90AAAAATLAAAAgAoEACAQFr/original)

| Metric      | Normal  | Streaming | Winner       |
| ----------- | ------- | --------- | ------------ |
| Latency     | 10.3s   | 4.4s      | ✅ Streaming |
| Throughput  | 0.097/s | 0.227/s   | ✅ Streaming |
| Parallelism | 1x      | 4x        | ✅ Streaming |
| Efficiency  | 60%     | 90%       | ✅ Streaming |
| Simplicity  | 100%    | 85%       | ⚠️ Normal    |
| Cost        | $17.2µ  | $17.7µ    | ⚠️ Normal    |
| Memory      | 66 MB   | 71 MB     | ⚠️ Normal    |

---

## 🚀 Quick Start

### Deploy Normal Mode

```bash
cd streaming-demo
unum-cli build
sam build && sam deploy --guided
```

### Run Normal Benchmark

```bash
python quick_benchmark.py
```

### Deploy Streaming Mode

```bash
unum-cli build --streaming
sam build && sam deploy
```

### Run Streaming Benchmark

```bash
python quick_benchmark.py
```

---

## 📁 Files

| File                 | Description                              |
| -------------------- | ---------------------------------------- |
| `Generator/app.py`   | Stage 1: Produces 5 items (0.5s each)    |
| `Processor/app.py`   | Stage 2: Processes each item (0.5s each) |
| `Analyzer/app.py`    | Stage 3: Analyzes each item (0.5s each)  |
| `Reporter/app.py`    | Stage 4: Produces final summary          |
| `template.yaml`      | SAM deployment template                  |
| `unum-template.yaml` | Unum workflow configuration              |
| `quick_benchmark.py` | Benchmark script                         |
| `charts/`            | Generated benchmark charts               |

---

## 🔬 How Streaming Works

### 1. Normal Mode (Sequential)

```python
# Each stage waits for complete input
def lambda_handler(event, context):
    item_1 = event["item_1"]  # Available immediately
    item_2 = event["item_2"]  # Available immediately
    # ... process all items
    return {"processed_1": ..., "processed_2": ...}
```

### 2. Streaming Mode (Pipeline Parallel)

```python
# Stage starts with futures, resolves on-demand
def lambda_handler(event, context):
    # Event contains futures for items not yet computed
    input_data = LazyFutureDict(event)  # Wraps futures

    item_1 = input_data["item_1"]  # Resolves immediately or waits
    processed_1 = process(item_1)
    publish("processed_1", processed_1)  # Next stage can start!

    item_2 = input_data["item_2"]  # May wait for Generator
    processed_2 = process(item_2)
    publish("processed_2", processed_2)
    # ...
```

### Key Mechanism: LazyFutureDict

- Wraps input event containing futures
- Transparently resolves futures when accessed
- Polls DynamoDB for future values
- Returns computed values immediately if available

---

## 📊 When to Use Streaming

### ✅ Good Fit

- Multi-stage pipelines with independent outputs
- Compute-heavy stages (>100ms per item)
- Latency-sensitive applications
- Pipelines with 3+ stages

### ❌ Not Ideal

- Single-stage functions
- All outputs depend on all inputs
- Very fast stages (<10ms per item)
- Cost-critical with minimal latency requirements

---

## 🏆 Conclusion

Partial Parameter Streaming achieves **57% latency improvement** with only:

- **+3% cost overhead**
- **+8% memory overhead**
- **+2.7% network overhead**

The key enabler is **pipeline parallelism**: downstream functions start processing as soon as the first output field is ready, rather than waiting for the entire upstream function to complete.

---

_Benchmark conducted on AWS Lambda (256 MB, Python 3.11, eu-central-1)_
_DynamoDB table: unum-streaming-demo (on-demand capacity)_
