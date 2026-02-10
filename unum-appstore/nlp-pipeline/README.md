# NLP Pipeline - Natural Language Processing Benchmark

A **4-stage sequential chain** performing real NLP operations on text documents.
No artificial delays — all latency comes from genuine computation.

This workflow is inspired by common NLP pipelines found in academic serverless
benchmarks (SeBS, ServerlessBench) and real-world text analytics applications.

## Academic References

- **SeBS** (Middleware '21): Text processing workloads for serverless
- **ExCamera** (NSDI '17): Video/data processing pipelines with fine-grained parallelism
- **Serverless text analytics**: Common pattern in enterprise serverless architectures

## Workflow Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Tokenizer   │───▶│  Analyzer    │───▶│  Classifier  │───▶│  Summarizer  │
│  (tokenize   │    │  (NER +      │    │  (sentiment  │    │  (extractive │
│   + POS tag) │    │   features)  │    │   + topics)  │    │   summary)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Stages

| Stage          | Operations                                          | Real Compute Time |
|----------------|-----------------------------------------------------|-------------------|
| **Tokenizer**  | Sentence split, word tokenize, POS tagging, n-grams | ~200-500ms        |
| **Analyzer**   | Named entity recognition, frequency analysis, dependency features | ~300-600ms |
| **Classifier** | TF-IDF vectors, sentiment scoring, topic classification | ~200-400ms    |
| **Summarizer** | TextRank sentence scoring, extractive summarization | ~150-350ms        |

### Why Partial Streaming Helps

Each stage produces **multiple independent output fields**:
- `Tokenizer` → `sentences`, `pos_tags`, `ngrams`, `vocab_stats`, `token_matrix`
- `Analyzer` → `entities`, `freq_dist`, `dep_features`, `collocations`, `readability`
- `Classifier` → `sentiment`, `tfidf_vectors`, `topics`, `text_features`, `classification`

Each field in the next stage depends on only **one** field from the previous stage,
enabling streaming: once `sentences` is computed, `Analyzer` can start working
on `entities` while `Tokenizer` still computes `ngrams`, `vocab_stats`, etc.

## Data

Uses real English text — public domain excerpts from Project Gutenberg and
Wikipedia. The input size controls computational load.

## Deployment

```bash
cd nlp-pipeline
unum-cli build --streaming
unum-cli deploy
```

## Benchmark

```bash
python benchmark.py --compare
```
