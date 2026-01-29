"""
Result Accumulator - Aggregates detection results from all frame batches.

Demonstrates Future-Based mode benefits:
- Fast batches (simple scenes) complete early and are immediately available
- While waiting for slow batches (complex scenes), fast results are processed
- Background polling caches results as they become available
"""
import time
import json
import os
import boto3

SQS_QUEUE_URL = os.environ.get('SQS_RESULT_QUEUE', 
    'https://sqs.eu-west-1.amazonaws.com/528757807812/video-analysis-results')
try:
    sqs_client = boto3.client('sqs', region_name='eu-west-1')
except:
    sqs_client = None


def lambda_handler(event, context):
    """
    Accumulate object detection results from all frame batches.
    
    EXPECTED TIMING (with varied complexity):
    - Batch 0: ~0.3s (simple scene) - returns FAST
    - Batch 1: ~1.5s (medium scene)
    - Batch 2: ~4.0s (complex scene) 
    - Batch 3: ~2.0s (medium scene)
    - Batch 4: ~0.5s (simple scene) - returns FAST
    - Batch 5: ~6.0s (very complex) - SLOWEST
    
    Future-Based benefit: Process simple scenes while complex scenes still running!
    """
    start_time = time.time()
    
    inputs = event
    
    # Get mode
    eager_mode = os.environ.get('EAGER', 'false').lower() in ('true', '1', 'yes')
    future_mode = os.environ.get('UNUM_FUTURE_BASED', 'false').lower() in ('true', '1', 'yes')
    mode = 'FUTURE_BASED' if (eager_mode and future_mode) else ('EAGER' if eager_mode else 'CLASSIC')
    
    input_type = type(inputs).__name__
    num_inputs = len(inputs) if hasattr(inputs, '__len__') else 'unknown'
    
    print(f'')
    print(f'=' * 70)
    print(f'[ACCUMULATOR] Starting in {mode} mode')
    print(f'[ACCUMULATOR] Input type: {input_type}, Count: {num_inputs}')
    print(f'=' * 70)
    
    if hasattr(inputs, 'get_ready_count'):
        ready, pending = inputs.get_ready_count()
        print(f'[ACCUMULATOR] Initial state: {ready} ready, {pending} pending')
    
    # Aggregate results
    all_class_counts = {}
    total_detections = 0
    total_frames = 0
    batch_results = []
    access_times = []
    
    print(f'')
    print(f'-' * 70)
    print(f'[DEMO] Accessing batch results ONE BY ONE')
    print(f'[DEMO] Expected: Fast batches (0, 4) ready first, slow batch (5) last')
    print(f'-' * 70)
    
    for i in range(len(inputs) if hasattr(inputs, '__len__') else 6):
        before_access = time.time()
        elapsed_before = int((before_access - start_time) * 1000)
        
        already_resolved = False
        if hasattr(inputs, '_resolved_values'):
            already_resolved = i in inputs._resolved_values
        elif hasattr(inputs, '_inputs'):
            already_resolved = inputs._inputs[i].is_resolved
        
        print(f'')
        print(f'[BATCH {i}] === Accessing batch {i} at t={elapsed_before}ms ===')
        if already_resolved:
            print(f'[BATCH {i}] ✓ Already resolved from BACKGROUND POLLING!')
        else:
            print(f'[BATCH {i}] ⏳ Not yet resolved, will wait...')
        
        # ----- ACTUAL ACCESS -----
        batch_result = inputs[i]
        # -------------------------
        
        after_access = time.time()
        elapsed_after = int((after_access - start_time) * 1000)
        wait_time = int((after_access - before_access) * 1000)
        
        access_times.append({
            'batch_index': i,
            'wait_time_ms': wait_time,
            'total_elapsed_ms': elapsed_after,
            'was_pre_resolved': already_resolved
        })
        
        print(f'[BATCH {i}] ✓ Received at t={elapsed_after}ms (waited {wait_time}ms)')
        
        if isinstance(batch_result, dict):
            # Aggregate detections
            n_detections = batch_result.get('total_detections', 0)
            n_frames = batch_result.get('n_frames_processed', 0)
            class_counts = batch_result.get('class_counts', {})
            
            total_detections += n_detections
            total_frames += n_frames
            
            for cls, count in class_counts.items():
                all_class_counts[cls] = all_class_counts.get(cls, 0) + count
            
            batch_results.append({
                'batch_id': batch_result.get('batch_id'),
                'n_frames': n_frames,
                'n_detections': n_detections,
                'processing_time_ms': batch_result.get('processing_time_ms')
            })
            
            print(f'[BATCH {i}]   Frames: {n_frames}, Detections: {n_detections}')
            print(f'[BATCH {i}]   Processing time: {batch_result.get("processing_time_ms", 0)}ms')
    
    # Sort class counts for top objects
    sorted_classes = sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True)
    top_5_classes = dict(sorted_classes[:5])
    
    pre_resolved_count = sum(1 for a in access_times if a['was_pre_resolved'])
    total_wait_time = sum(a['wait_time_ms'] for a in access_times)
    
    total_time_ms = int((time.time() - start_time) * 1000)
    
    print(f'')
    print(f'=' * 70)
    print(f'[ACCUMULATOR] SUMMARY')
    print(f'=' * 70)
    print(f'  Mode: {mode}')
    print(f'  Total frames: {total_frames}')
    print(f'  Total detections: {total_detections}')
    print(f'  Top 5 objects: {top_5_classes}')
    print(f'  Pre-resolved (background polling): {pre_resolved_count}/{len(access_times)}')
    print(f'  Total wait time: {total_wait_time}ms')
    print(f'  Total aggregation time: {total_time_ms}ms')
    
    result = {
        'mode': mode,
        'video_analysis_complete': True,
        'total_frames': total_frames,
        'total_detections': total_detections,
        'top_5_objects': top_5_classes,
        'all_class_counts': all_class_counts,
        'batch_results': batch_results,
        'pre_resolved_count': pre_resolved_count,
        'total_wait_time_ms': total_wait_time,
        'aggregation_time_ms': total_time_ms,
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
        {'batch_id': 0, 'total_detections': 15, 'n_frames_processed': 5, 'class_counts': {'person': 8, 'car': 7}, 'processing_time_ms': 320},
        {'batch_id': 1, 'total_detections': 28, 'n_frames_processed': 5, 'class_counts': {'person': 12, 'car': 10, 'dog': 6}, 'processing_time_ms': 1580},
        {'batch_id': 2, 'total_detections': 45, 'n_frames_processed': 5, 'class_counts': {'person': 20, 'car': 15, 'bicycle': 10}, 'processing_time_ms': 4120},
    ]
    result = lambda_handler(mock_inputs, None)
    print(json.dumps({k: v for k, v in result.items() if k not in ['all_class_counts', 'batch_results', 'access_times']}, indent=2))
