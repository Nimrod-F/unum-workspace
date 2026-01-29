# Graph Analysis Pipeline

A serverless workflow inspired by **SeBS (Serverless Benchmark Suite)** scientific benchmarks that demonstrates fan-out/fan-in patterns with parallel graph algorithms.

## Workflow Structure

```
                    ┌─────────────────┐
                    │ GraphGenerator  │
                    │   (Entry Point) │
                    └────────┬────────┘
                             │ Fan-Out (3 branches)
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │ PageRank │       │   BFS    │       │   MST    │
    │  ~500ms  │       │  ~300ms  │       │  ~400ms  │
    └────┬─────┘       └────┬─────┘       └────┬─────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │ Fan-In
                    ┌────────▼────────┐
                    │   Aggregator    │
                    │   (Terminal)    │
                    └─────────────────┘
```

## Functions

| Function         | Description                                                                        |
| ---------------- | ---------------------------------------------------------------------------------- |
| `GraphGenerator` | Generates a random Barabási-Albert graph and serializes it for parallel processing |
| `PageRank`       | Computes the PageRank algorithm on the graph (measures node importance)            |
| `BFS`            | Performs Breadth-First Search traversal from a source node                         |
| `MST`            | Computes the Minimum Spanning Tree of the graph                                    |
| `Aggregator`     | Collects results from all algorithms and produces a summary report                 |

## Research Origin

Inspired by:

- **SeBS (Serverless Benchmark Suite)** - Scientific graph benchmarks (501.graph-pagerank, 502.graph-mst, 503.graph-bfs)
- **Unum** - Decentralized serverless orchestration (NSDI'23)

## Usage

```bash
# Build and deploy
cd graph-analysis
unum-cli build -p aws -g
unum-cli deploy -p aws

# Invoke with custom graph size
aws lambda invoke --function-name graph-analysis-GraphGenerator \
  --payload '{"Data":{"Source":"http","Value":{"size":500,"seed":42}}}' \
  --cli-binary-format raw-in-base64-out response.json
```

## Expected Output

The Aggregator produces a summary like:

```json
{
  "graph_info": {
    "nodes": 500,
    "edges": 4960
  },
  "pagerank": {
    "top_nodes": [42, 15, 7],
    "compute_time_ms": 487
  },
  "bfs": {
    "visited_nodes": 500,
    "max_depth": 4,
    "compute_time_ms": 312
  },
  "mst": {
    "total_weight": 1247.5,
    "edges_in_mst": 499,
    "compute_time_ms": 398
  }
}
```
