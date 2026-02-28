"""
Stage 1: Generator Function

Produces 5 INDEPENDENT items sequentially.
Each item takes ~0.5 seconds to generate.

With streaming: After item_1 is ready (0.5s), Processor can start
                while Generator continues with item_2, item_3, etc.
"""

import json
import time
from unum_streaming import StreamingPublisher, set_streaming_output
import hashlib

ITEM_TIME = 0.5  # Each item takes 0.5 seconds to generate


def generate_item(item_id, input_data):
    """Generate a single item - simulates 0.5s of computation."""
    start = time.time()
    
    # Simulate CPU-intensive work
    result = {
        "id": item_id,
        "source": "generator",
        "input": input_data,
        "hash": hashlib.md5(f"{input_data}_{item_id}".encode()).hexdigest()[:8],
        "value": hash(f"{input_data}_{item_id}") % 1000,
        "generated_at": time.time()
    }
    
    # Ensure minimum processing time
    elapsed = time.time() - start
    if elapsed < ITEM_TIME:
        time.sleep(ITEM_TIME - elapsed)
    
    return result


def lambda_handler(event, context):
    """Generator handler - produces 5 independent items."""
    

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = event.get('Session', '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="GeneratorFunction",
        field_names=["item_1", "item_2", "item_3", "item_4", "item_5"]
    )
    start_time = time.time()
    input_data = event.get("input", "benchmark_run")
    
    print(f"[Generator] Starting with input: {input_data}")
    
    # Generate 5 items sequentially
    item_1 = generate_item(1, input_data)
    _streaming_publisher.publish('item_1', item_1)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()
    print(f"[Generator] item_1 ready at {time.time() - start_time:.3f}s")
    
    item_2 = generate_item(2, input_data)
    _streaming_publisher.publish('item_2', item_2)
    print(f"[Generator] item_2 ready at {time.time() - start_time:.3f}s")
    
    item_3 = generate_item(3, input_data)
    _streaming_publisher.publish('item_3', item_3)
    print(f"[Generator] item_3 ready at {time.time() - start_time:.3f}s")
    
    item_4 = generate_item(4, input_data)
    _streaming_publisher.publish('item_4', item_4)
    print(f"[Generator] item_4 ready at {time.time() - start_time:.3f}s")
    
    item_5 = generate_item(5, input_data)
    _streaming_publisher.publish('item_5', item_5)
    print(f"[Generator] item_5 ready at {time.time() - start_time:.3f}s")
    
    total_time = time.time() - start_time
    print(f"[Generator] Completed in {total_time:.3f}s")
    
    return {
        "item_1": item_1,
        "item_2": item_2,
        "item_3": item_3,
        "item_4": item_4,
        "item_5": item_5
    }
