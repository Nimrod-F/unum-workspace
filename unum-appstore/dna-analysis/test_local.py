#!/usr/bin/env python3
"""
Local end-to-end test for the DNA Sequence Analysis Pipeline.

Runs all 4 stages sequentially to verify correctness and measure compute times.
"""
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reader.app import lambda_handler as read_seq
from dna_analyzer.app import lambda_handler as analyze
from comparator.app import lambda_handler as compare
from reporter.app import lambda_handler as report


def run_pipeline(seq_length=50000, gc_content=0.5, seq_id='benchmark_seq'):
    """Run the complete DNA analysis pipeline and collect timing."""
    print('=' * 70)
    print(f'DNA Analysis Pipeline - Local Test')
    print(f'Sequence: {seq_length}bp, GC={gc_content}, id={seq_id}')
    print('=' * 70)

    pipeline_start = time.time()
    stage_times = {}

    # Stage 1: Reader
    t0 = time.time()
    read_result = read_seq({
        'sequence_length': seq_length,
        'gc_content': gc_content,
        'seq_id': seq_id
    }, None)
    stage_times['reader'] = int((time.time() - t0) * 1000)
    print()

    # Stage 2: Analyzer
    t0 = time.time()
    ana_result = analyze(read_result, None)
    stage_times['analyzer'] = int((time.time() - t0) * 1000)
    print()

    # Stage 3: Comparator
    t0 = time.time()
    cmp_result = compare(ana_result, None)
    stage_times['comparator'] = int((time.time() - t0) * 1000)
    print()

    # Stage 4: Reporter
    t0 = time.time()
    rep_result = report(cmp_result, None)
    stage_times['reporter'] = int((time.time() - t0) * 1000)

    pipeline_time = int((time.time() - pipeline_start) * 1000)

    print()
    print('=' * 70)
    print('RESULTS')
    print('=' * 70)
    print(f'  Reader:      {stage_times["reader"]:>6}ms')
    print(f'  Analyzer:    {stage_times["analyzer"]:>6}ms')
    print(f'  Comparator:  {stage_times["comparator"]:>6}ms')
    print(f'  Reporter:    {stage_times["reporter"]:>6}ms')
    print(f'  ─────────────────────')
    print(f'  Total E2E:   {pipeline_time:>6}ms')
    print()

    # Per-field timings
    print('Per-field compute times:')
    for stage_name, result in [('Reader', read_result), ('Analyzer', ana_result),
                                ('Comparator', cmp_result), ('Reporter', rep_result)]:
        fields = [(k, v.get('compute_ms', 0)) for k, v in result.items() if isinstance(v, dict)]
        print(f'  {stage_name}:')
        for field, ms in fields:
            print(f'    {field}: {ms}ms')

    # Key results
    quality = rep_result.get('quality_scores', {})
    annotation = rep_result.get('annotation', {})
    final = rep_result.get('final_report', {})
    print()
    print('Key Findings:')
    print(f'  Quality grade: {quality.get("grade")}')
    print(f'  Classification: {annotation.get("classification")}')
    print(f'  Genome type: {final.get("assessment", {}).get("genome_type")}')
    print(f'  Evolutionary status: {final.get("evolutionary_status", "unknown")}')

    return pipeline_time, stage_times


if __name__ == '__main__':
    # Default: 50kb sequence
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    gc = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    run_pipeline(length, gc)
