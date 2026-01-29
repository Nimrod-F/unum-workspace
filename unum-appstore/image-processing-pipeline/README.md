# Image Processing Pipeline

A serverless workflow that performs multiple image operations in parallel, demonstrating fan-in patterns with **operation-specific processing times** ranging from milliseconds to seconds.

## Workflow Structure

```
                    ┌─────────────────┐
                    │   ImageLoader   │
                    │  (Entry Point)  │
                    └────────┬────────┘
                             │ Fan-Out (5 operations)
    ┌────────────────────────┼────────────────────────┐
    │        ┌───────────────┼───────────────┐        │
    ▼        ▼               ▼               ▼        ▼
┌───────┐┌───────┐     ┌───────┐     ┌───────┐┌───────┐
│Extract││Generate│    │ Resize │    │ Apply ││Detect │
│Metadata│Thumbnail│   │ Image  │    │Filters││ Faces │
│ ~50ms ││ ~150ms │    │ ~400ms │    │ ~1.5s ││ ~3.5s │
│(FAST) ││ (FAST) │    │(MEDIUM)│    │(MEDIUM)│(SLOW) │
└───┬───┘└───┬───┘     └───┬───┘     └───┬───┘└───┬───┘
    │        │               │               │        │
    └────────┴───────────────┼───────────────┴────────┘
                             │ Fan-In
                    ┌────────▼────────┐
                    │ ImageAggregator │
                    │   (Terminal)    │
                    └─────────────────┘
```

## Functions

| Function | Duration | Description |
|----------|----------|-------------|
| `ImageLoader` | ~200ms | Loads image, creates payloads for 5 parallel operations |
| `ExtractMetadata` | ~50-100ms | Reads EXIF data, file info (FASTEST) |
| `GenerateThumbnail` | ~100-200ms | Creates 150x150 thumbnail |
| `ResizeImage` | ~300-500ms | Resizes to 1920px width |
| `ApplyFilters` | ~1-2s | Applies enhancement filters (sharpen, denoise, color) |
| `DetectFaces` | ~2-5s | ML-based face detection and analysis (SLOWEST) |
| `ImageAggregator` | ~50ms | Combines all results into final output |

## Processing Time Comparison

```
ExtractMetadata  ▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  50ms
GenerateThumbnail ▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  150ms
ResizeImage       ▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  400ms
ApplyFilters      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░  1.5s
DetectFaces       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  3.5s
```

## FUTURE_BASED Benefits

This workflow has the **widest range** of processing times (50ms to 3.5s = 70x difference):

- Metadata and Thumbnail available almost instantly
- Resize completes before filters start processing
- By aggregator access time, 3-4 operations typically pre-resolved

Expected improvement: **25-30%** latency reduction vs CLASSIC mode.

## Research Origin

Inspired by image processing pipelines from:
- Serverless image processing benchmarks
- AWS Lambda image resizing patterns
- Computer vision preprocessing workflows

## Usage

```bash
# Deploy
cd image-processing-pipeline
unum-cli deploy

# Invoke
aws lambda invoke --function-name image-processing-pipeline-ImageLoader \
  --payload '{"Data":{"Value":{"image_id":"test","width":4000,"height":3000}}}' response.json
```
