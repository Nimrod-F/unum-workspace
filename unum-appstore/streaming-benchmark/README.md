# Streaming Benchmark Application

This application benchmarks **Partial Parameter Streaming** vs **Normal Execution**.

## Architecture

```
Producer → Consumer → Aggregator
```

- **Producer**: Computes 3 fields sequentially (each takes ~1 second)
- **Consumer**: Receives fields and processes them
- **Aggregator**: Final aggregation

## Key Insight

With **Normal Execution**:
- Producer computes all 3 fields (3 seconds)
- Then invokes Consumer
- Consumer starts processing after 3 seconds

With **Partial Parameter Streaming**:
- Producer computes field1 (1 second) → immediately invokes Consumer
- Consumer receives field1 + futures for field2, field3
- Consumer starts processing field1 while Producer computes field2, field3
- Consumer resolves futures when it needs field2, field3

**Expected improvement**: Consumer starts ~2 seconds earlier!

## Deployment

```bash
# Build with streaming
cd streaming-benchmark
unum-cli template
unum-cli build --streaming
unum-cli deploy

# Build without streaming (baseline)
unum-cli build
unum-cli deploy
```

## Benchmarking

```bash
python benchmark.py --mode streaming --runs 5
python benchmark.py --mode normal --runs 5
python benchmark.py --compare
```
