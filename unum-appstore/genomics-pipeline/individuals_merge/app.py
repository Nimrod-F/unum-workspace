"""
Individuals Merge - Merges results from all individual variant calls.

First fan-in point in the genomics pipeline. Combines variant calls
from all individuals before sifting analysis.
"""
import time
import json
import os


def lambda_handler(event, context):
    """
    Merge variant calls from all individuals.
    
    This is the FIRST FAN-IN point with highly varied input times:
    - HG00096, HG00097: ~0.4-0.5s (FAST)
    - NA12891, NA12892: ~1.5-1.8s (MEDIUM)
    - HG00099: ~2.5s (MEDIUM-HIGH)
    - NA12878: ~3.5s (SLOW)
    """
    start_time = time.time()
    
    inputs = event
    
    eager_mode = os.environ.get('EAGER', 'false').lower() in ('true', '1', 'yes')
    future_mode = os.environ.get('UNUM_FUTURE_BASED', 'false').lower() in ('true', '1', 'yes')
    mode = 'FUTURE_BASED' if (eager_mode and future_mode) else ('EAGER' if eager_mode else 'CLASSIC')
    
    print(f'')
    print(f'=' * 70)
    print(f'[MERGE] Starting in {mode} mode')
    print(f'=' * 70)
    
    merged_variants = {
        'total_snps': 0,
        'total_indels': 0,
        'individuals': [],
    }
    
    access_times = []
    
    for i in range(len(inputs) if hasattr(inputs, '__len__') else 6):
        before_access = time.time()
        elapsed_before = int((before_access - start_time) * 1000)
        
        already_resolved = False
        if hasattr(inputs, '_resolved_values'):
            already_resolved = i in inputs._resolved_values
        
        print(f'[MERGE] Accessing individual {i} at t={elapsed_before}ms (pre-resolved: {already_resolved})')
        
        individual_result = inputs[i]
        
        after_access = time.time()
        wait_time = int((after_access - before_access) * 1000)
        
        access_times.append({
            'index': i,
            'wait_time_ms': wait_time,
            'pre_resolved': already_resolved
        })
        
        if isinstance(individual_result, dict):
            variants = individual_result.get('variants', {})
            merged_variants['total_snps'] += variants.get('snps', 0)
            merged_variants['total_indels'] += variants.get('indels', 0)
            merged_variants['individuals'].append({
                'id': individual_result.get('individual_id', f'ind_{i}'),
                'variants': variants.get('total', 0),
                'coverage': individual_result.get('coverage_depth', 0)
            })
        
        print(f'[MERGE] Received individual {i} in {wait_time}ms')
    
    pre_resolved_count = sum(1 for a in access_times if a['pre_resolved'])
    total_time = int((time.time() - start_time) * 1000)
    
    result = {
        'phase': 'individuals_merge',
        'mode': mode,
        'merged_variants': merged_variants,
        'total_variants': merged_variants['total_snps'] + merged_variants['total_indels'],
        'num_individuals': len(merged_variants['individuals']),
        'pre_resolved_count': pre_resolved_count,
        'processing_time_ms': total_time,
        'timestamp': time.time()
    }
    
    print(f'[MERGE] Merged {result["num_individuals"]} individuals, {result["total_variants"]} variants')
    print(f'[MERGE] Pre-resolved: {pre_resolved_count}/{len(access_times)}')
    
    return result


if __name__ == '__main__':
    mock_inputs = [
        {'individual_id': 'NA12878', 'coverage_depth': 60, 'variants': {'snps': 5000, 'indels': 800, 'total': 5800}},
        {'individual_id': 'NA12891', 'coverage_depth': 30, 'variants': {'snps': 3000, 'indels': 450, 'total': 3450}},
    ]
    result = lambda_handler(mock_inputs, None)
    print(json.dumps(result, indent=2))
