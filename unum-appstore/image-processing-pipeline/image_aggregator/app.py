"""
Image Aggregator - Combines results from all image processing operations.

Demonstrates Future-Based benefits with HIGHLY VARIED processing times:
- Metadata: ~50-100ms (INSTANT)
- Thumbnail: ~100-200ms (FAST)
- Resize: ~300-500ms (MEDIUM-FAST)
- Filters: ~1-2s (MEDIUM)
- Face Detection: ~2-5s (SLOW)

With Future-Based, fast operations are immediately available while slow ones still run!
"""
import time
import json
import os
import boto3

SQS_QUEUE_URL = os.environ.get('SQS_RESULT_QUEUE', 
    'https://sqs.eu-west-1.amazonaws.com/528757807812/image-proc-results')
try:
    sqs_client = boto3.client('sqs', region_name='eu-west-1')
except:
    sqs_client = None


def lambda_handler(event, context):
    """
    Aggregate all image processing results.
    
    EXPECTED TIMING:
    - inputs[0]: Metadata - ~50-100ms (INSTANT availability)
    - inputs[1]: Thumbnail - ~100-200ms (pre-resolved likely)
    - inputs[2]: Resize - ~300-500ms (pre-resolved likely)
    - inputs[3]: Filters - ~1-2s (may be pre-resolved)
    - inputs[4]: Face Detection - ~2-5s (SLOWEST)
    
    Future-Based benefit: By the time we access inputs[4], 
    inputs[0-3] results are already being processed!
    """
    start_time = time.time()
    
    inputs = event
    
    eager_mode = os.environ.get('EAGER', 'false').lower() in ('true', '1', 'yes')
    future_mode = os.environ.get('UNUM_FUTURE_BASED', 'false').lower() in ('true', '1', 'yes')
    mode = 'FUTURE_BASED' if (eager_mode and future_mode) else ('EAGER' if eager_mode else 'CLASSIC')
    
    input_type = type(inputs).__name__
    num_inputs = len(inputs) if hasattr(inputs, '__len__') else 'unknown'
    
    print(f'')
    print(f'=' * 70)
    print(f'[IMAGE_AGGREGATOR] Starting in {mode} mode')
    print(f'[IMAGE_AGGREGATOR] Input type: {input_type}, Count: {num_inputs}')
    print(f'=' * 70)
    
    if hasattr(inputs, 'get_ready_count'):
        ready, pending = inputs.get_ready_count()
        print(f'[IMAGE_AGGREGATOR] Initial state: {ready} ready, {pending} pending')
    
    operation_names = ['metadata', 'thumbnail', 'resize', 'filters', 'faces']
    operation_results = {}
    access_times = []
    
    print(f'')
    print(f'-' * 70)
    print(f'[DEMO] Accessing operation results ONE BY ONE')
    print(f'[DEMO] Expected order of completion: metadata → thumbnail → resize → filters → faces')
    print(f'-' * 70)
    
    for i in range(len(inputs) if hasattr(inputs, '__len__') else 5):
        before_access = time.time()
        elapsed_before = int((before_access - start_time) * 1000)
        
        already_resolved = False
        if hasattr(inputs, '_resolved_values'):
            already_resolved = i in inputs._resolved_values
        elif hasattr(inputs, '_inputs'):
            already_resolved = inputs._inputs[i].is_resolved
        
        op_name = operation_names[i] if i < len(operation_names) else f'op_{i}'
        print(f'')
        print(f'[OP {i}] === Accessing {op_name} result at t={elapsed_before}ms ===')
        if already_resolved:
            print(f'[OP {i}] ✓ Already resolved from BACKGROUND POLLING!')
        else:
            print(f'[OP {i}] ⏳ Not yet resolved, will wait...')
        
        # ----- ACTUAL ACCESS -----
        op_result = inputs[i]
        # -------------------------
        
        after_access = time.time()
        elapsed_after = int((after_access - start_time) * 1000)
        wait_time = int((after_access - before_access) * 1000)
        
        access_times.append({
            'operation_index': i,
            'operation_name': op_name,
            'wait_time_ms': wait_time,
            'total_elapsed_ms': elapsed_after,
            'was_pre_resolved': already_resolved
        })
        
        print(f'[OP {i}] ✓ Received at t={elapsed_after}ms (waited {wait_time}ms)')
        
        if isinstance(op_result, dict):
            operation_results[op_name] = {
                'operation': op_result.get('operation', op_name),
                'processing_time_ms': op_result.get('processing_time_ms', 0),
                'success': True
            }
            
            # Extract key info based on operation type
            if 'num_faces' in op_result:
                operation_results[op_name]['num_faces'] = op_result['num_faces']
            if 'thumbnail_url' in op_result:
                operation_results[op_name]['thumbnail_url'] = op_result['thumbnail_url']
            if 'resized_size' in op_result:
                operation_results[op_name]['resized_size'] = op_result['resized_size']
            if 'num_filters' in op_result:
                operation_results[op_name]['num_filters'] = op_result['num_filters']
            if 'exif' in op_result:
                operation_results[op_name]['has_exif'] = True
            
            print(f'[OP {i}]   Operation: {op_result.get("operation", op_name)}')
            print(f'[OP {i}]   Processing time: {op_result.get("processing_time_ms", 0)}ms')
    
    pre_resolved_count = sum(1 for a in access_times if a['was_pre_resolved'])
    total_wait_time = sum(a['wait_time_ms'] for a in access_times)
    total_time_ms = int((time.time() - start_time) * 1000)
    
    print(f'')
    print(f'=' * 70)
    print(f'[IMAGE_AGGREGATOR] SUMMARY')
    print(f'=' * 70)
    print(f'  Mode: {mode}')
    print(f'  Operations completed: {len(operation_results)}')
    print(f'  Pre-resolved (background polling): {pre_resolved_count}/{len(access_times)}')
    print(f'  Total wait time: {total_wait_time}ms')
    print(f'  Total aggregation time: {total_time_ms}ms')
    
    result = {
        'mode': mode,
        'image_processing_complete': True,
        'operations': operation_results,
        'num_operations': len(operation_results),
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
        {'operation': 'extract_metadata', 'processing_time_ms': 80, 'exif': {}},
        {'operation': 'generate_thumbnail', 'processing_time_ms': 150, 'thumbnail_url': 's3://...'},
        {'operation': 'resize_image', 'processing_time_ms': 420, 'resized_size': '1920x1440'},
        {'operation': 'apply_filters', 'processing_time_ms': 1650, 'num_filters': 4},
        {'operation': 'detect_faces', 'processing_time_ms': 3800, 'num_faces': 3},
    ]
    result = lambda_handler(mock_inputs, None)
    print(json.dumps({k: v for k, v in result.items() if k != 'access_times'}, indent=2))
