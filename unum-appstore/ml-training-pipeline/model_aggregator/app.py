"""
Model Aggregator - Collects and compares trained models.

This demonstrates Future-Based mode with parallel background polling:
- When accessing inputs[0] (Linear Regression result), polls ALL inputs
- LR finishes first (~100ms), immediately available
- While processing LR, background polling caches SVM, RF, GB results
- By time we need other models, they may already be resolved!
"""
import time
import json
import os
import boto3

# SQS for completion signaling
SQS_QUEUE_URL = os.environ.get('SQS_RESULT_QUEUE', 
    'https://sqs.eu-west-1.amazonaws.com/528757807812/ml-training-results')
try:
    sqs_client = boto3.client('sqs', region_name='eu-west-1')
except:
    sqs_client = None


def lambda_handler(event, context):
    """
    Aggregate model training results and select best model.
    
    EXPECTED TIMING (with Future-Based):
    - inputs[0]: Linear Regression - ready in ~100ms (returns immediately)
    - inputs[1]: SVM - ready in ~2-5s (may be pre-resolved)
    - inputs[2]: Random Forest - ready in ~8-12s (slowest)
    - inputs[3]: Gradient Boosting - ready in ~5-8s (may be pre-resolved)
    
    Future-Based benefit: While waiting for RF, LR/SVM/GB processing continues!
    """
    start_time = time.time()
    
    inputs = event
    
    # Get mode from environment
    eager_mode = os.environ.get('EAGER', 'false').lower() in ('true', '1', 'yes')
    future_mode = os.environ.get('UNUM_FUTURE_BASED', 'false').lower() in ('true', '1', 'yes')
    mode = 'FUTURE_BASED' if (eager_mode and future_mode) else ('EAGER' if eager_mode else 'CLASSIC')
    
    input_type = type(inputs).__name__
    num_inputs = len(inputs) if hasattr(inputs, '__len__') else 'unknown'
    
    print(f'')
    print(f'=' * 70)
    print(f'[MODEL_AGGREGATOR] Starting in {mode} mode')
    print(f'[MODEL_AGGREGATOR] Input type: {input_type}, Count: {num_inputs}')
    print(f'=' * 70)
    
    # Check initial ready state
    if hasattr(inputs, 'get_ready_count'):
        ready, pending = inputs.get_ready_count()
        print(f'[MODEL_AGGREGATOR] Initial state: {ready} ready, {pending} pending')
    
    models = []
    processing_log = []
    access_times = []
    model_names = ['LinearRegression', 'SVM', 'RandomForest', 'GradientBoosting']
    
    print(f'')
    print(f'-' * 70)
    print(f'[DEMO] Accessing model results ONE BY ONE')
    print(f'[DEMO] Expected order of completion: LR(fast) → SVM → GB → RF(slow)')
    print(f'-' * 70)
    
    for i in range(len(inputs) if hasattr(inputs, '__len__') else 4):
        before_access = time.time()
        elapsed_before = int((before_access - start_time) * 1000)
        
        # Check if already resolved
        already_resolved = False
        if hasattr(inputs, '_resolved_values'):
            already_resolved = i in inputs._resolved_values
        elif hasattr(inputs, '_inputs'):
            already_resolved = inputs._inputs[i].is_resolved
        
        expected_model = model_names[i] if i < len(model_names) else f'Model_{i}'
        print(f'')
        print(f'[MODEL {i}] === Accessing {expected_model} result at t={elapsed_before}ms ===')
        if already_resolved:
            print(f'[MODEL {i}] ✓ Already resolved from BACKGROUND POLLING!')
        else:
            print(f'[MODEL {i}] ⏳ Not yet resolved, will wait...')
        
        # ----- ACTUAL ACCESS -----
        model_result = inputs[i]
        # -------------------------
        
        after_access = time.time()
        elapsed_after = int((after_access - start_time) * 1000)
        wait_time = int((after_access - before_access) * 1000)
        
        access_times.append({
            'model_index': i,
            'model_name': model_result.get('model_name', expected_model) if isinstance(model_result, dict) else expected_model,
            'wait_time_ms': wait_time,
            'total_elapsed_ms': elapsed_after,
            'was_pre_resolved': already_resolved
        })
        
        print(f'[MODEL {i}] ✓ Received result at t={elapsed_after}ms (waited {wait_time}ms)')
        
        if isinstance(model_result, dict):
            models.append(model_result)
            print(f'[MODEL {i}]   Model: {model_result.get("model_name", "unknown")}')
            print(f'[MODEL {i}]   Accuracy: {model_result.get("accuracy", 0):.4f}')
            print(f'[MODEL {i}]   Training time: {model_result.get("training_time_ms", 0)}ms')
            
            processing_log.append({
                'model_name': model_result.get('model_name'),
                'accuracy': model_result.get('accuracy'),
                'training_time_ms': model_result.get('training_time_ms')
            })
    
    # Find best model
    best_model = max(models, key=lambda m: m.get('accuracy', 0)) if models else None
    
    # Calculate pre-resolved count
    pre_resolved_count = sum(1 for a in access_times if a['was_pre_resolved'])
    total_wait_time = sum(a['wait_time_ms'] for a in access_times)
    
    total_time_ms = int((time.time() - start_time) * 1000)
    
    print(f'')
    print(f'=' * 70)
    print(f'[MODEL_AGGREGATOR] SUMMARY')
    print(f'=' * 70)
    print(f'  Mode: {mode}')
    print(f'  Models processed: {len(models)}')
    print(f'  Pre-resolved (background polling): {pre_resolved_count}/{len(access_times)}')
    print(f'  Total wait time: {total_wait_time}ms')
    print(f'  Total processing time: {total_time_ms}ms')
    if best_model:
        print(f'  Best model: {best_model.get("model_name")} (accuracy: {best_model.get("accuracy", 0):.4f})')
    
    result = {
        'mode': mode,
        'num_models': len(models),
        'models': processing_log,
        'best_model': best_model.get('model_name') if best_model else None,
        'best_accuracy': best_model.get('accuracy') if best_model else None,
        'pre_resolved_count': pre_resolved_count,
        'total_wait_time_ms': total_wait_time,
        'aggregation_time_ms': total_time_ms,
        'access_times': access_times,
        'timestamp': time.time()
    }
    
    # Send to SQS if available
    if sqs_client:
        try:
            sqs_client.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(result)
            )
        except Exception as e:
            print(f'[SQS] Failed to send result: {e}')
    
    return result


if __name__ == '__main__':
    # Test locally with mock inputs
    mock_inputs = [
        {"model_name": "LinearRegression", "accuracy": 0.82, "training_time_ms": 150},
        {"model_name": "SVM", "accuracy": 0.88, "training_time_ms": 3500},
        {"model_name": "RandomForest", "accuracy": 0.91, "training_time_ms": 9200},
        {"model_name": "GradientBoosting", "accuracy": 0.89, "training_time_ms": 6100},
    ]
    result = lambda_handler(mock_inputs, None)
    print(json.dumps(result, indent=2))
