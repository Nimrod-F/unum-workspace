# ML Training Pipeline

A serverless workflow that trains multiple machine learning models in parallel and aggregates their results, demonstrating fan-in/fan-out patterns with **highly varied execution times**.

## Workflow Structure

```
                    ┌─────────────────┐
                    │  DataGenerator  │
                    │   (Entry Point) │
                    └────────┬────────┘
                             │ Fan-Out (4 branches)
         ┌───────────────────┼───────────────────┐
         │           ┌───────┴───────┐           │
         ▼           ▼               ▼           ▼
    ┌─────────┐ ┌─────────┐   ┌─────────┐ ┌─────────┐
    │ TrainLR │ │TrainSVM │   │ TrainRF │ │TrainGB  │
    │ ~100ms  │ │  ~2-5s  │   │ ~8-12s  │ │ ~5-8s   │
    │ (FAST)  │ │(MEDIUM) │   │ (SLOW)  │ │(MED-HI) │
    └────┬────┘ └────┬────┘   └────┬────┘ └────┬────┘
         │           │               │           │
         └───────────┴───────┬───────┴───────────┘
                             │ Fan-In
                    ┌────────▼────────┐
                    │ ModelAggregator │
                    │   (Terminal)    │
                    └─────────────────┘
```

## Functions

| Function | Duration | Description |
|----------|----------|-------------|
| `DataGenerator` | ~200ms | Generates synthetic training dataset, creates payloads for 4 models |
| `TrainLR` | ~100-200ms | Trains Linear Regression (fastest - simple model) |
| `TrainSVM` | ~2-5s | Trains Support Vector Machine (medium - kernel computation) |
| `TrainRF` | ~8-12s | Trains Random Forest (slowest - many decision trees) |
| `TrainGB` | ~5-8s | Trains Gradient Boosting (medium-high - sequential boosting) |
| `ModelAggregator` | ~50ms | Aggregates results, selects best model, reports metrics |

## FUTURE_BASED Benefits

With such varied training times (100ms to 12s), FUTURE_BASED execution provides significant benefits:

- **Linear Regression** results available almost immediately
- **SVM** results pre-resolved while waiting for slower models
- By the time **Random Forest** completes, 3 other models are already aggregated

Expected improvement: **25-30%** latency reduction vs CLASSIC mode.

## Research Origin

Inspired by ML ensemble training patterns from:
- SeBS-Flow (EuroSys'25) ML training benchmarks
- FunctionBench distributed training workloads

## Usage

```bash
# Deploy
cd ml-training-pipeline
unum-cli deploy

# Invoke
aws lambda invoke --function-name ml-training-pipeline-DataGenerator \
  --payload '{"Data":{"Value":{"dataset_size":1000}}}' response.json
```
