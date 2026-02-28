# DNA Sequence Analysis Pipeline

A **4-stage sequential chain** performing real bioinformatics computation on DNA sequences.
No artificial delays — all latency comes from genuine sequence processing.

Inspired by the **SeBS benchmark 504.dna-visualisation** (Middleware '21, ETH Zurich)
and the **DNAVisualization.org** project by Benjamin Lee.

## Academic References

- **SeBS 504.dna-visualisation** (Copik et al., Middleware '21): DNA processing as serverless benchmark
- **DNAVisualization.org** (Lee, 2019): Squiggle visualization method for DNA sequences
- **BLAST pipeline patterns**: Common in bioinformatics serverless workflows

## Workflow Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Reader     │───▶│   Analyzer   │───▶│  Comparator  │───▶│  Reporter    │
│  (parse +    │    │  (GC content │    │  (alignment  │    │  (stats +    │
│   k-mers)    │    │   + codons)  │    │   + motifs)  │    │   visualize) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Stages

| Stage          | Operations                                              | Real Compute Time |
|----------------|---------------------------------------------------------|-------------------|
| **Reader**     | FASTA parse, k-mer extraction, base composition, ORFs   | ~200-600ms        |
| **Analyzer**   | GC content windows, codon usage, CpG islands, repeats   | ~300-700ms        |
| **Comparator** | Sequence alignment scoring, motif search, palindromes   | ~200-500ms        |
| **Reporter**   | Statistical summary, squiggle coordinates, final report | ~150-400ms        |

### Why Partial Streaming Helps

Each stage produces **5 independent output fields** with one-to-one dependencies.
Once `Reader` computes `kmers`, `Analyzer` can start working on `gc_content`
while `Reader` still computes `base_composition`, `open_reading_frames`, etc.

## Data

Uses programmatically generated DNA sequences with realistic properties
(GC content ~40-60%, realistic codon distribution). No external files needed.

## Deployment

```bash
cd dna-analysis
unum-cli build --streaming
unum-cli deploy
```
