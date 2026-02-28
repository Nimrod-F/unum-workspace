#!/usr/bin/env python3
"""
Local end-to-end test for the NLP Pipeline.

Runs all 4 stages sequentially to verify correctness and measure compute times.
"""
import json
import time
import sys
import os

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer.app import lambda_handler as tokenize
from analyzer.app import lambda_handler as analyze
from classifier.app import lambda_handler as classify
from summarizer.app import lambda_handler as summarize


# ─── Input text corpus (public domain — Project Gutenberg style) ─────────

INPUT_TEXT = """
Natural language processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language, in particular how to program computers to process and analyze large
amounts of natural language data. The result is a computer capable of understanding
the contents of documents, including the contextual nuances of the language within
them. Challenges in natural language processing frequently involve speech recognition,
natural language understanding, and natural language generation.

Natural language processing has its roots in computational linguistics and has a
history of more than 50 years. In the 1950s, Alan Turing published an article titled
Computing Machinery and Intelligence which proposed what is now called the Turing test
as a criterion of intelligence. The Georgetown experiment in 1954 involved fully
automatic translation of more than sixty Russian sentences into English. The authors
claimed that within three or five years, machine translation would be a solved problem.
However, real progress was much slower, and after the ALPAC report in 1966, which
found that ten-year-long research had failed to fulfill the expectations, funding for
machine translation was dramatically reduced.

Little further research in machine translation was conducted until the late 1980s
when the first statistical machine translation systems were developed. Some notably
successful natural language processing systems developed in the 1960s were SHRDLU, a
natural language system working in restricted blocks worlds with restricted
vocabularies, and ELIZA, a simulation of a Rogerian psychotherapist, written by
Joseph Weizenbaum. Using almost no information about human thought or emotion, ELIZA
sometimes provided a startlingly human-like interaction.

During the 1970s, many programmers began to write conceptual ontologies, which
structured real-world information into computer-understandable data. In the 1980s and
early 1990s, most natural language processing systems were based on complex sets of
hand-written rules. Starting in the late 1980s, however, there was a revolution in
natural language processing with the introduction of machine learning algorithms for
language processing. This was due to both the steady increase in computational power
and the gradual lessening of the dominance of Chomskyan theories of linguistics,
whose theoretical underpinnings discouraged the sort of corpus linguistics that
underlies the machine-learning approach to language processing.

Modern deep learning techniques for NLP include word embedding, transformer models,
and large language models. Recurrent neural networks and long short-term memory
networks were popular in the 2010s, but have largely been replaced by transformer
architectures. The attention mechanism, introduced in the Transformer model by
Vaswani et al. in 2017, revolutionized NLP by enabling models to process sequences
in parallel rather than sequentially. BERT, GPT, and their successors have achieved
state-of-the-art results on virtually all NLP benchmarks, including question answering,
named entity recognition, sentiment analysis, and machine translation.

Transfer learning through pre-trained language models has become the dominant paradigm
in NLP. These models are first trained on massive corpora of unlabeled text using
self-supervised objectives like masked language modeling or next-token prediction.
They are then fine-tuned on specific downstream tasks using much smaller labeled
datasets. This approach has dramatically improved performance across a wide range of
NLP applications, from information extraction and text summarization to dialogue
systems and code generation.

The field continues to evolve rapidly with new architectures and training methods
being proposed regularly. Mixture of experts models, retrieval-augmented generation,
and multimodal models that combine text with images and audio represent the current
frontier of research. The scalability of these approaches and their ability to
generalize across tasks and languages make them particularly attractive for real-world
applications in healthcare, finance, legal analysis, and scientific research.
""" * 3  # Triple for realistic workload


def run_pipeline(text, doc_id='benchmark'):
    """Run the complete NLP pipeline and collect timing."""
    print('=' * 70)
    print(f'NLP Pipeline - Local Test (doc_id={doc_id})')
    print(f'Input: {len(text)} characters')
    print('=' * 70)

    pipeline_start = time.time()
    stage_times = {}

    # Stage 1: Tokenizer
    t0 = time.time()
    tok_result = tokenize({'text': text, 'doc_id': doc_id}, None)
    stage_times['tokenizer'] = int((time.time() - t0) * 1000)
    print()

    # Stage 2: Analyzer
    t0 = time.time()
    ana_result = analyze(tok_result, None)
    stage_times['analyzer'] = int((time.time() - t0) * 1000)
    print()

    # Stage 3: Classifier
    t0 = time.time()
    cls_result = classify(ana_result, None)
    stage_times['classifier'] = int((time.time() - t0) * 1000)
    print()

    # Stage 4: Summarizer
    t0 = time.time()
    sum_result = summarize(cls_result, None)
    stage_times['summarizer'] = int((time.time() - t0) * 1000)

    pipeline_time = int((time.time() - pipeline_start) * 1000)

    print()
    print('=' * 70)
    print('RESULTS')
    print('=' * 70)
    print(f'  Tokenizer:   {stage_times["tokenizer"]:>6}ms')
    print(f'  Analyzer:    {stage_times["analyzer"]:>6}ms')
    print(f'  Classifier:  {stage_times["classifier"]:>6}ms')
    print(f'  Summarizer:  {stage_times["summarizer"]:>6}ms')
    print(f'  ─────────────────────')
    print(f'  Total E2E:   {pipeline_time:>6}ms')
    print()

    # Per-field timings
    print('Per-field compute times:')
    for stage_name, result in [('Tokenizer', tok_result), ('Analyzer', ana_result),
                                ('Classifier', cls_result), ('Summarizer', sum_result)]:
        fields = [(k, v.get('compute_ms', 0)) for k, v in result.items() if isinstance(v, dict)]
        print(f'  {stage_name}:')
        for field, ms in fields:
            print(f'    {field}: {ms}ms')

    # Key results
    report = sum_result.get('final_report', {})
    abstract = sum_result.get('abstract', {})
    keywords = sum_result.get('keywords', {})
    print()
    print('Key Findings:')
    print(f'  Reading level: {report.get("reading_level")}')
    print(f'  Target audience: {report.get("target_audience")}')
    print(f'  Primary topic: {abstract.get("primary_topic")}')
    print(f'  Top keywords: {[kw["word"] for kw in keywords.get("top_keywords", [])[:10]]}')

    return pipeline_time, stage_times


if __name__ == '__main__':
    pipeline_time, stage_times = run_pipeline(INPUT_TEXT)
