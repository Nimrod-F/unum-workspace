"""
Publisher - Fan-in aggregator (Terminal function)

Aggregates results from all 4 processing branches:
- Thumbnail (~50ms - FASTEST)
- Transform (~100ms)
- Filters (~150ms)
- Contour (~300ms - SLOWEST)

CLASSIC mode: Triggered by Contour (slowest, ~300ms)
FUTURE mode: Triggered by Thumbnail (fastest, ~50ms), waits for futures
"""
import json
import time


def lambda_handler(event, context):
    """
    Aggregate results from all image processing branches.
    
    In CLASSIC mode: Triggered by LAST function to complete (Contour)
    In FUTURE mode: Triggered by FIRST function to complete (Thumbnail)
    
    Input: Array of results from all 4 branches
    Output: Aggregated summary
    """
    start_time = time.time()
    
    print(f'[Publisher] Starting - Fan-in aggregator')
    print(f'[Publisher] Received {len(event) if hasattr(event, "__len__") else 1} results')
    
    # Handle different input formats
    all_results = []
    if isinstance(event, list):
        all_results = event
    elif isinstance(event, dict):
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
            all_results = body if isinstance(body, list) else [body]
        else:
            all_results = [event]
    else:
        # Handle LazyInputList, AsyncFutureInputList, or other list-like objects
        # These support iteration and indexing transparently
        all_results = list(event)
    
    # Parse results from each branch
    branch_results = {}
    for result in all_results:
        if isinstance(result, dict):
            branch = result.get('branch', result.get('operation', 'unknown'))
            branch_results[branch] = {
                'compute_time_ms': result.get('compute_time_ms', 0),
                'total_time_ms': result.get('total_time_ms', 0),
                'input_size': result.get('input_size', [0, 0]),
                'output_bytes': result.get('output_bytes', 0)
            }
            print(f'[Publisher] - {branch}: compute={result.get("compute_time_ms", 0)}ms, total={result.get("total_time_ms", 0)}ms')
    
    # Calculate aggregate statistics
    compute_times = {k: v['compute_time_ms'] for k, v in branch_results.items()}
    total_times = {k: v['total_time_ms'] for k, v in branch_results.items()}
    
    # Find fastest and slowest
    if compute_times:
        fastest_branch = min(compute_times.keys(), key=lambda k: compute_times[k])
        slowest_branch = max(compute_times.keys(), key=lambda k: compute_times[k])
        fastest_time = compute_times[fastest_branch]
        slowest_time = compute_times[slowest_branch]
    else:
        fastest_branch = slowest_branch = 'unknown'
        fastest_time = slowest_time = 0
    
    aggregation_time = (time.time() - start_time) * 1000
    
    result = {
        'status': 'complete',
        'branches_received': len(all_results),
        'branch_compute_times': compute_times,
        'branch_total_times': total_times,
        'fastest': {
            'branch': fastest_branch,
            'compute_time_ms': fastest_time
        },
        'slowest': {
            'branch': slowest_branch,
            'compute_time_ms': slowest_time
        },
        'timing_variance_ms': slowest_time - fastest_time,
        'timing_ratio': round(slowest_time / fastest_time, 2) if fastest_time > 0 else 0,
        'aggregation_time_ms': int(aggregation_time),
        'timestamp': time.time()
    }
    
    print(f'[Publisher] COMPLETE')
    print(f'[Publisher] Fastest: {fastest_branch} ({fastest_time}ms)')
    print(f'[Publisher] Slowest: {slowest_branch} ({slowest_time}ms)')
    print(f'[Publisher] Variance: {result["timing_variance_ms"]}ms ({result["timing_ratio"]}x)')
    print(f'[Publisher] FUTURE mode benefit: ~{result["timing_variance_ms"]}ms saved per execution')
    
    return result


if __name__ == '__main__':
    # Simulate aggregation
    test_results = [
        {'branch': 'Thumbnail', 'compute_time_ms': 52, 'total_time_ms': 65, 'output_bytes': 5000},
        {'branch': 'Transform', 'compute_time_ms': 120, 'total_time_ms': 140, 'output_bytes': 50000},
        {'branch': 'Filters', 'compute_time_ms': 180, 'total_time_ms': 200, 'output_bytes': 48000},
        {'branch': 'Contour', 'compute_time_ms': 320, 'total_time_ms': 350, 'output_bytes': 45000},
    ]
    
    result = lambda_handler(test_results, None)
    print(json.dumps(result, indent=2))
