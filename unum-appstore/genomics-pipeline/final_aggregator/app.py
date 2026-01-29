"""
Final Aggregator - Combines results from mutation overlap and frequency analysis.

SECOND FAN-IN point with very different input times:
- Frequency Analysis: ~0.3-0.5s (FAST)
- Mutation Overlap: ~2-4s (SLOW)

Perfect demonstration of Future-Based execution benefits!
"""
import time
import json
import os
import boto3

SQS_QUEUE_URL = os.environ.get('SQS_RESULT_QUEUE',
    'https://sqs.eu-west-1.amazonaws.com/528757807812/genomics-results')
try:
    sqs_client = boto3.client('sqs', region_name='eu-west-1')
except:
    sqs_client = None


def lambda_handler(event, context):
    """
    Aggregate final genomics analysis results.
    
    EXPECTED TIMING:
    - inputs[0]: Mutation Overlap - ~2-4s (SLOW)
    - inputs[1]: Frequency Analysis - ~0.3-0.5s (FAST)
    
    With Future-Based: Frequency results available IMMEDIATELY
    while waiting for slower mutation overlap!
    """
    start_time = time.time()
    
    inputs = event
    
    eager_mode = os.environ.get('EAGER', 'false').lower() in ('true', '1', 'yes')
    future_mode = os.environ.get('UNUM_FUTURE_BASED', 'false').lower() in ('true', '1', 'yes')
    mode = 'FUTURE_BASED' if (eager_mode and future_mode) else ('EAGER' if eager_mode else 'CLASSIC')
    
    print(f'')
    print(f'=' * 70)
    print(f'[FINAL_AGG] Starting in {mode} mode')
    print(f'[FINAL_AGG] Second fan-in point: mutation_overlap vs frequency_analysis')
    print(f'=' * 70)
    
    analysis_results = {}
    access_times = []
    
    analysis_names = ['mutation_overlap', 'frequency_analysis']
    
    for i in range(len(inputs) if hasattr(inputs, '__len__') else 2):
        before_access = time.time()
        elapsed_before = int((before_access - start_time) * 1000)
        
        already_resolved = False
        if hasattr(inputs, '_resolved_values'):
            already_resolved = i in inputs._resolved_values
        
        name = analysis_names[i] if i < len(analysis_names) else f'analysis_{i}'
        print(f'')
        print(f'[FINAL_AGG] Accessing {name} at t={elapsed_before}ms')
        if already_resolved:
            print(f'[FINAL_AGG] ✓ Pre-resolved from background polling!')
        else:
            print(f'[FINAL_AGG] ⏳ Waiting for result...')
        
        analysis_result = inputs[i]
        
        after_access = time.time()
        wait_time = int((after_access - before_access) * 1000)
        elapsed_after = int((after_access - start_time) * 1000)
        
        access_times.append({
            'analysis': name,
            'index': i,
            'wait_time_ms': wait_time,
            'total_elapsed_ms': elapsed_after,
            'pre_resolved': already_resolved
        })
        
        print(f'[FINAL_AGG] ✓ Received at t={elapsed_after}ms (waited {wait_time}ms)')
        
        if isinstance(analysis_result, dict):
            analysis_type = analysis_result.get('analysis_type', name)
            analysis_results[analysis_type] = {
                'processing_time_ms': analysis_result.get('processing_time_ms', 0),
                'success': True
            }
            
            if analysis_type == 'mutation_overlap':
                analysis_results[analysis_type]['summary'] = analysis_result.get('summary', {})
            elif analysis_type == 'frequency_analysis':
                analysis_results[analysis_type]['statistics'] = analysis_result.get('statistics', {})
    
    pre_resolved_count = sum(1 for a in access_times if a['pre_resolved'])
    total_wait_time = sum(a['wait_time_ms'] for a in access_times)
    total_time = int((time.time() - start_time) * 1000)
    
    print(f'')
    print(f'=' * 70)
    print(f'[FINAL_AGG] GENOMICS PIPELINE COMPLETE')
    print(f'=' * 70)
    print(f'  Mode: {mode}')
    print(f'  Analyses completed: {len(analysis_results)}')
    print(f'  Pre-resolved: {pre_resolved_count}/{len(access_times)}')
    print(f'  Total wait time: {total_wait_time}ms')
    print(f'  Total aggregation time: {total_time}ms')
    
    result = {
        'mode': mode,
        'pipeline_complete': True,
        'analysis_results': analysis_results,
        'num_analyses': len(analysis_results),
        'pre_resolved_count': pre_resolved_count,
        'total_wait_time_ms': total_wait_time,
        'aggregation_time_ms': total_time,
        'access_times': access_times,
        'timestamp': time.time()
    }
    
    if sqs_client:
        try:
            sqs_client.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(result)
            )
        except Exception as e:
            print(f'[SQS] Failed: {e}')
    
    return result


if __name__ == '__main__':
    mock_inputs = [
        {
            'analysis_type': 'mutation_overlap',
            'processing_time_ms': 3200,
            'summary': {'known_variants': 8500, 'clinically_relevant': 12}
        },
        {
            'analysis_type': 'frequency_analysis',
            'processing_time_ms': 380,
            'statistics': {'mean_af': 0.18, 'pi_diversity': 0.001}
        }
    ]
    result = lambda_handler(mock_inputs, None)
    print(json.dumps({k: v for k, v in result.items() if k != 'access_times'}, indent=2))
