# branching-pipeline

## Workflow Diagram

```
A (start) --> B --> C ----\
          \-> D --> E ----> F (aggregator)
```

## Functions (6)

- **A**: receive request and split into fast/slow processing paths
- **B**: fast path: initial processing
- **C**: fast path: finalize and prepare for aggregation
- **D**: slow path: initial processing with heavier computation
- **E**: slow path: finalize and prepare for aggregation
- **F**: aggregate results from both processing paths

