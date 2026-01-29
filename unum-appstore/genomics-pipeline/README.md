# Genomics Pipeline (1000Genomes-inspired)

A serverless scientific workflow that processes genomic variant data across multiple individuals and performs parallel analyses, featuring **two fan-in points** with coverage-dependent processing times.

## Workflow Structure

```
                         ┌──────────────┐
                         │ DataSplitter │
                         │ (Entry Point)│
                         └──────┬───────┘
                                │ Fan-Out #1 (6 individuals)
    ┌───────────────────────────┼───────────────────────────┐
    │       ┌───────────────────┼───────────────────────────┤
    ▼       ▼           ▼       ▼       ▼           ▼       │
┌───────┐┌───────┐┌───────┐┌───────┐┌───────┐┌───────┐      │
│HG00096││HG00097││NA12891││NA12892││HG00099││NA12878│      │
│ 0.5s  ││ 0.4s  ││ 1.5s  ││ 1.8s  ││ 2.5s  ││ 3.5s  │      │
│(15x)  ││(15x)  ││(30x)  ││(30x)  ││(45x)  ││(60x)  │      │
└───┬───┘└───┬───┘└───┬───┘└───┬───┘└───┬───┘└───┬───┘      │
    └────────┴────────┴────┬───┴────────┴────────┘          │
                           │ Fan-In #1                      │
                  ┌────────▼────────┐                       │
                  │ IndividualsMerge│                       │
                  └────────┬────────┘                       │
                           │                                │
                  ┌────────▼────────┐                       │
                  │ SiftingVariants │                       │
                  └────────┬────────┘                       │
                           │ Fan-Out #2 (2 analyses)        │
              ┌────────────┴────────────┐                   │
              ▼                         ▼                   │
       ┌─────────────┐          ┌─────────────┐            │
       │MutationOverlap│        │FrequencyAnalysis│        │
       │   ~3.0s     │          │    ~0.4s    │            │
       │   (SLOW)    │          │   (FAST)    │            │
       └──────┬──────┘          └──────┬──────┘            │
              └────────────┬───────────┘                   │
                           │ Fan-In #2                     │
                  ┌────────▼────────┐                      │
                  │ FinalAggregator │                      │
                  │   (Terminal)    │                      │
                  └─────────────────┘                      │
```

## Functions

### Phase 1: Individual Processing

| Function | Coverage | Duration | Description |
|----------|----------|----------|-------------|
| `DataSplitter` | - | ~300ms | Splits VCF data by individual |
| `IndividualsProcessor` | 15x | ~0.4-0.5s | Low coverage - fast |
| `IndividualsProcessor` | 30x | ~1.5-1.8s | Medium coverage |
| `IndividualsProcessor` | 45x | ~2.5s | High-medium coverage |
| `IndividualsProcessor` | 60x | ~3.5s | High coverage - slow |
| `IndividualsMerge` | - | ~200ms | Merges all variant calls |

### Phase 2: Analysis

| Function | Duration | Description |
|----------|----------|-------------|
| `SiftingVariants` | ~500ms | Filters and categorizes variants |
| `MutationOverlap` | ~2-4s | Compares against ClinVar, COSMIC, gnomAD (SLOW) |
| `FrequencyAnalysis` | ~0.3-0.5s | Calculates allele frequencies (FAST) |
| `FinalAggregator` | ~100ms | Combines analysis results |

## Coverage → Processing Time

Higher sequencing coverage = more data = longer processing:

```
HG00096 (15x)  ▓▓▓▓░░░░░░░░░░░░░░░░░░░░  0.5s
HG00097 (15x)  ▓▓▓░░░░░░░░░░░░░░░░░░░░░  0.4s
NA12891 (30x)  ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░  1.5s
NA12892 (30x)  ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░  1.8s
HG00099 (45x)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░  2.5s
NA12878 (60x)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░  3.5s
```

## FUTURE_BASED Benefits

This workflow has **two fan-in points**, both benefiting from FUTURE_BASED:

**Fan-In #1 (Individuals):**
- Low coverage samples (HG00096, HG00097) complete quickly
- Pre-resolved before high coverage samples finish

**Fan-In #2 (Analyses):**
- Frequency analysis (~0.4s) completes 7x faster than mutation overlap (~3s)
- Frequency results available immediately at aggregator

Expected improvement: **20-28%** latency reduction vs CLASSIC mode.

## Research Origin

Inspired by genomics workflows from:
- 1000 Genomes Project analysis pipelines
- SeBS-Flow scientific workflow benchmarks
- GATK variant calling best practices

## Usage

```bash
# Deploy
cd genomics-pipeline
unum-cli deploy

# Invoke
aws lambda invoke --function-name genomics-pipeline-DataSplitter \
  --payload '{"Data":{"Value":{"chromosome":22,"num_individuals":6}}}' response.json
```
