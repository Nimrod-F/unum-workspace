# parallel-chains-merge

## Workflow Diagram

```
A (start) --> B --> C --> D ---------\
          \-> E --> F --> G ----------> H (aggregator) --> I
```

## Functions (9)

- **A**: receive image, split into processing and metadata paths
- **B**: resize image to target dimensions
- **C**: apply color correction and white balance
- **D**: apply artistic filters and finalize image
- **E**: extract EXIF and metadata from original
- **F**: analyze image content with ML model
- **G**: generate descriptive tags from analysis
- **H**: combine processed image with enriched metadata
- **I**: publish final result to storage

