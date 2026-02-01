# Image Pipeline - Real Computation Benchmark

A fan-out/fan-in workflow using **real image processing operations** (no artificial delays).
Designed to demonstrate Unum's FUTURE mode benefits with naturally varying execution times.

## Workflow Architecture

```
                      ┌────────────────┐
                      │  ImageLoader   │
                      │  (S3 Download) │
                      └───────┬────────┘
                              │ Fan-out (Broadcast)
          ┌───────────┬───────┴───────┬───────────┐
          ▼           ▼               ▼           ▼
   ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
   │ Thumbnail │ │  Filters  │ │  Contour  │ │ Transform │
   │  ~50ms ⚡ │ │ ~150-200ms│ │~200-400ms │ │ ~100-150ms│
   │ (resize)  │ │(blur+sharp)│ │  (edge)   │ │(rotate+flip)│
   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
         │             │             │             │
         └─────────────┴──────┬──────┴─────────────┘
                              │ Fan-in
                      ┌───────▼────────┐
                      │   Publisher    │
                      │  (Aggregate)   │
                      └────────────────┘
```

## Real Computation Details

| Function      | Operations                         | Complexity             | Expected Time    |
| ------------- | ---------------------------------- | ---------------------- | ---------------- |
| **Thumbnail** | `image.thumbnail((128,128))`       | O(n) simple resize     | **~50-80ms** ⚡  |
| **Transform** | `ROTATE_90 + ROTATE_180 + FLIP_LR` | O(n) pixel remap       | ~100-150ms       |
| **Filters**   | `BLUR + SHARPEN + SMOOTH`          | O(n×k) convolution     | ~150-200ms       |
| **Contour**   | `CONTOUR + FIND_EDGES + EMBOSS`    | O(n×k²) edge detection | **~200-400ms** ★ |

**Timing variance: ~5-8x (50ms vs 400ms) - all from REAL computation!**

## Why This Demonstrates FUTURE Mode

- **CLASSIC mode**: Contour (slowest, ~400ms) triggers Publisher
- **FUTURE mode**: Thumbnail (fastest, ~50ms) triggers Publisher first
- **Expected improvement**: ~350ms per workflow execution

## Prerequisites

1. S3 bucket with test images
2. PIL/Pillow installed in Lambda layer

## Test Images

Upload test images to S3:

- Small: 640x480 (~50KB)
- Medium: 1920x1080 (~500KB)
- Large: 4000x3000 (~2MB)

Larger images = more pronounced timing differences.

## Deployment

```bash
cd image-pipeline
unum-cli build
unum-cli deploy
```

## Benchmark

```bash
python benchmark/run_benchmark.py
```
