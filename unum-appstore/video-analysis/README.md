# Video Analysis Pipeline

A serverless workflow that decodes video into frame batches and performs parallel object detection, demonstrating fan-in patterns with **scene complexity-dependent processing times**.

## Workflow Structure

```
                    ┌──────────────────┐
                    │   VideoDecoder   │
                    │   (Entry Point)  │
                    └────────┬─────────┘
                             │ Fan-Out (6 batches)
    ┌────────────────────────┼────────────────────────┐
    │        ┌───────────────┼───────────────┐        │
    ▼        ▼               ▼               ▼        ▼
┌───────┐┌───────┐     ┌───────┐     ┌───────┐┌───────┐
│Batch 0││Batch 1│     │Batch 2│     │Batch 3││Batch 4│ ...
│ 0.3s  ││ 1.5s  │     │ 4.0s  │     │ 2.0s  ││ 0.5s  │
│(simple)│(medium)│    │(complex)│   │(medium)│(simple)│
└───┬───┘└───┬───┘     └───┬───┘     └───┬───┘└───┬───┘
    │        │               │               │        │
    └────────┴───────────────┼───────────────┴────────┘
                             │ Fan-In
                    ┌────────▼─────────┐
                    │ ResultAccumulator│
                    │    (Terminal)    │
                    └──────────────────┘
```

## Functions

| Function | Duration | Description |
|----------|----------|-------------|
| `VideoDecoder` | ~500ms | Decodes video, splits into 6 frame batches based on scene complexity |
| `FrameDetector` | 0.3s - 6.0s | Object detection per batch (varies by scene complexity) |
| `ResultAccumulator` | ~100ms | Aggregates detections, generates timeline, tracks confidence |

## Scene Complexity Map

| Batch | Complexity | Duration | Example Content |
|-------|------------|----------|-----------------|
| 0 | Simple | ~0.3s | Static background, minimal objects |
| 1 | Medium | ~1.5s | Moderate motion, few objects |
| 2 | Complex | ~4.0s | Dense scene, many objects |
| 3 | Medium | ~2.0s | Moderate activity |
| 4 | Simple | ~0.5s | Low activity scene |
| 5 | Very Complex | ~6.0s | Action sequence, many moving objects |

## FUTURE_BASED Benefits

Video analysis naturally has varied processing times based on scene content:

- Simple scenes (batches 0, 4) complete in under 1 second
- Complex scenes (batches 2, 5) take 4-6 seconds
- Background polling resolves fast batches while slow ones process

Expected improvement: **20-25%** latency reduction vs CLASSIC mode.

## Research Origin

Inspired by video processing pipelines from:
- ExCamera (NSDI'17) parallel video encoding
- DataFlower (ASPLOS'24) video analytics workflows

## Usage

```bash
# Deploy
cd video-analysis
unum-cli deploy

# Invoke
aws lambda invoke --function-name video-analysis-VideoDecoder \
  --payload '{"Data":{"Value":{"video_id":"sample","duration_s":60}}}' response.json
```
