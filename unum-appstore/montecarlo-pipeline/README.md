# Monte Carlo Analysis Pipeline

A diamond-topology scientific computing workflow for benchmarking Unum's
three enhancement techniques: **Future-Based Execution**, **Partial
Parameter Streaming**, and **Intelligent Function Fusion**.

Inspired by:

- Malawski et al., _"Serverless execution of scientific workflows"_,
  Future Generation Computer Systems, 2020
- SeBS (Serverless Benchmark Suite), Copik et al., ACM Computing Surveys, 2021
- Ristov et al., _"DAG-based serverless computing"_, IEEE CLOUD, 2021

## Topology

```
DataGenerator ──┬─> Transform → Estimate → Validate ─┬─> Aggregator → Reporter
                └─> Simulate ─────────────────────────┘
```

**7 functions, 2 branches, diamond pattern.**

| Branch     | Functions                       | Pattern                           | Approx. Duration |
| ---------- | ------------------------------- | --------------------------------- | ---------------- |
| A (chain)  | Transform → Estimate → Validate | 3-node sequential chain (fusible) | 400–800 ms       |
| B (single) | Simulate                        | Monte Carlo simulation            | 200–500 ms       |
| Fan-in     | Aggregator                      | Merges both branches              | 10–50 ms         |
| Tail       | Reporter                        | Generates summary report          | 10–30 ms         |

## Enhancement Coverage

| Enhancement   | Where it helps                                  | Mechanism                                                                                 |
| ------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Fusion**    | Transform → Estimate → Validate                 | Collapses 3 cold starts into 1; eliminates inter-function serialisation                   |
| **Futures**   | Aggregator fan-in                               | Invoked when Simulate completes (faster branch); uses futures for Validate (slower chain) |
| **Streaming** | DataGenerator → Transform, Transform → Estimate | Downstream starts processing partial fields before upstream finishes                      |

## Functions

### DataGenerator (entry)

Generates a random 150×150 matrix (Box-Muller normal distribution) and
Monte Carlo simulation parameters. Pure Python, no dependencies.

### Transform (chain step 1)

Z-score normalisation (center + scale) and Gram matrix computation
(X^T × X). The matrix multiplication is O(n³) — the heaviest pure-Python
step.

### Estimate (chain step 2)

Power iteration for dominant eigenvalues of the covariance matrix.
Ordinary least-squares regression via Gaussian elimination with partial
pivoting.

### Validate (chain step 3 / fan-in terminal)

K-fold cross-validation, chi-squared goodness-of-fit test on eigenvalue
distribution, bootstrap confidence intervals (200 resamples).

### Simulate (parallel branch / fan-in terminal)

Monte Carlo π estimation (200K samples), geometric Brownian motion random
walks (100 walks × 1000 steps), Black-Scholes option pricing.

### Aggregator (fan-in)

Merges chain validation results with Monte Carlo simulation outputs.
Computes cross-method consistency metrics.

### Reporter (terminal)

Convergence diagnostics, overall quality scoring, and human-readable
report generation.

## Usage

```bash
# Compile Step Functions → unum_config.json
unum-cli compile -p step-functions -w unum-step-functions.json -t unum-template.yaml

# Generate SAM template
unum-cli template -t unum-template.yaml -p aws

# Build and deploy
unum-cli build -p aws -g -t unum-template.yaml
unum-cli deploy

# (Optional) Apply fusion
unum-cli fuse --config fusion.yaml -t unum-template.yaml
```

## Configurable Parameters

Pass as input payload to DataGenerator:

```json
{
  "matrix_size": 150,
  "n_simulations": 200000,
  "n_walks": 100,
  "seed": 42
}
```

| Parameter       | Default | Effect on runtime                                   |
| --------------- | ------- | --------------------------------------------------- |
| `matrix_size`   | 150     | O(n³) impact on Transform; O(n²) on Estimate        |
| `n_simulations` | 200000  | Linear impact on Simulate                           |
| `n_walks`       | 100     | Linear impact on Simulate                           |
| `seed`          | None    | Reproducibility (fixed seed = deterministic output) |
