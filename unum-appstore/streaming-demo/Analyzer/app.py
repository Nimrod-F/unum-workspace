"""
Stage 3: Analyzer Function

Analyzes 5 processed items. Each analyzed item depends on
ONLY ONE input item (one-to-one dependency).

- analyzed_1 depends only on processed_1
- analyzed_2 depends only on processed_2
- etc.
"""

import json
import time
from unum_streaming import StreamingPublisher, set_streaming_output
import math

ITEM_TIME = 0.5  # Each item takes 0.5 seconds to analyze


def analyze_item(item_id, item_data):
    """Analyze a single item - simulates 0.5s of computation."""
    start = time.time()
    
    # Extract value from input item
    input_value = item_data.get("processed_value", 0) if isinstance(item_data, dict) else 0
    
    result = {
        "id": item_id,
        "source": "analyzer",
        "input_value": input_value,
        "analyzed_value": int(math.sqrt(input_value) * 10),
        "score": min(100, input_value // 10),
        "category": "high" if input_value > 500 else "medium" if input_value > 200 else "low",
        "analyzed_at": time.time()
    }
    
    # Ensure minimum processing time
    elapsed = time.time() - start
    if elapsed < ITEM_TIME:
        time.sleep(ITEM_TIME - elapsed)
    
    return result


def lambda_handler(event, context):
    """Analyzer handler - analyzes items as they arrive."""
    

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = event.get('Session', '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="AnalyzerFunction",
        field_names=["analyzed_1", "analyzed_2", "analyzed_3", "analyzed_4", "analyzed_5"]
    )
    start_time = time.time()
    print(f"[Analyzer] Starting")
    
    # Analyze processed_1
    t1 = time.time()
    processed_1 = event.get("processed_1")
    wait_1 = time.time() - t1
    print(f"[Analyzer] Got processed_1 in {wait_1:.3f}s")
    analyzed_1 = analyze_item(1, processed_1)
    _streaming_publisher.publish('analyzed_1', analyzed_1)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()
    print(f"[Analyzer] analyzed_1 ready at {time.time() - start_time:.3f}s")
    
    # Analyze processed_2
    t2 = time.time()
    processed_2 = event.get("processed_2")
    wait_2 = time.time() - t2
    print(f"[Analyzer] Got processed_2 in {wait_2:.3f}s")
    analyzed_2 = analyze_item(2, processed_2)
    _streaming_publisher.publish('analyzed_2', analyzed_2)
    print(f"[Analyzer] analyzed_2 ready at {time.time() - start_time:.3f}s")
    
    # Analyze processed_3
    t3 = time.time()
    processed_3 = event.get("processed_3")
    wait_3 = time.time() - t3
    print(f"[Analyzer] Got processed_3 in {wait_3:.3f}s")
    analyzed_3 = analyze_item(3, processed_3)
    _streaming_publisher.publish('analyzed_3', analyzed_3)
    print(f"[Analyzer] analyzed_3 ready at {time.time() - start_time:.3f}s")
    
    # Analyze processed_4
    t4 = time.time()
    processed_4 = event.get("processed_4")
    wait_4 = time.time() - t4
    print(f"[Analyzer] Got processed_4 in {wait_4:.3f}s")
    analyzed_4 = analyze_item(4, processed_4)
    _streaming_publisher.publish('analyzed_4', analyzed_4)
    print(f"[Analyzer] analyzed_4 ready at {time.time() - start_time:.3f}s")
    
    # Analyze processed_5
    t5 = time.time()
    processed_5 = event.get("processed_5")
    wait_5 = time.time() - t5
    print(f"[Analyzer] Got processed_5 in {wait_5:.3f}s")
    analyzed_5 = analyze_item(5, processed_5)
    _streaming_publisher.publish('analyzed_5', analyzed_5)
    print(f"[Analyzer] analyzed_5 ready at {time.time() - start_time:.3f}s")
    
    total_time = time.time() - start_time
    print(f"[Analyzer] Completed in {total_time:.3f}s")
    
    return {
        "analyzed_1": analyzed_1,
        "analyzed_2": analyzed_2,
        "analyzed_3": analyzed_3,
        "analyzed_4": analyzed_4,
        "analyzed_5": analyzed_5
    }
