"""
Stage 2: Processor Function

Processes 5 items from Generator. Each processed item depends on
ONLY ONE input item (one-to-one dependency).

- processed_1 depends only on item_1
- processed_2 depends only on item_2
- etc.

With streaming: Can start processing item_1 immediately when it arrives,
                while waiting for item_2, item_3, etc. to be computed.
"""

import json
from unum_streaming import StreamingPublisher, set_streaming_output
import time

ITEM_TIME = 0.5  # Each item takes 0.5 seconds to process


def process_item(item_id, item_data):
    """Process a single item - simulates 0.5s of computation."""
    start = time.time()
    
    # Extract value from input item
    input_value = item_data.get("value", 0) if isinstance(item_data, dict) else 0
    
    result = {
        "id": item_id,
        "source": "processor",
        "input_value": input_value,
        "processed_value": input_value * 2 + 100,
        "status": "processed",
        "processed_at": time.time()
    }
    
    # Ensure minimum processing time
    elapsed = time.time() - start
    if elapsed < ITEM_TIME:
        time.sleep(ITEM_TIME - elapsed)
    
    return result


def lambda_handler(event, context):
    """Processor handler - processes items as they arrive."""
    

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = event.get('Session', '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="ProcessorFunction",
        field_names=["processed_1", "processed_2", "processed_3", "processed_4", "processed_5"]
    )
    start_time = time.time()
    print(f"[Processor] Starting")
    
    # Process item_1 (may block on future in streaming mode)
    t1 = time.time()
    item_1 = event.get("item_1")
    wait_1 = time.time() - t1
    print(f"[Processor] Got item_1 in {wait_1:.3f}s")
    processed_1 = process_item(1, item_1)
    _streaming_publisher.publish('processed_1', processed_1)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()
    print(f"[Processor] processed_1 ready at {time.time() - start_time:.3f}s")
    
    # Process item_2
    t2 = time.time()
    item_2 = event.get("item_2")
    wait_2 = time.time() - t2
    print(f"[Processor] Got item_2 in {wait_2:.3f}s")
    processed_2 = process_item(2, item_2)
    _streaming_publisher.publish('processed_2', processed_2)
    print(f"[Processor] processed_2 ready at {time.time() - start_time:.3f}s")
    
    # Process item_3
    t3 = time.time()
    item_3 = event.get("item_3")
    wait_3 = time.time() - t3
    print(f"[Processor] Got item_3 in {wait_3:.3f}s")
    processed_3 = process_item(3, item_3)
    _streaming_publisher.publish('processed_3', processed_3)
    print(f"[Processor] processed_3 ready at {time.time() - start_time:.3f}s")
    
    # Process item_4
    t4 = time.time()
    item_4 = event.get("item_4")
    wait_4 = time.time() - t4
    print(f"[Processor] Got item_4 in {wait_4:.3f}s")
    processed_4 = process_item(4, item_4)
    _streaming_publisher.publish('processed_4', processed_4)
    print(f"[Processor] processed_4 ready at {time.time() - start_time:.3f}s")
    
    # Process item_5
    t5 = time.time()
    item_5 = event.get("item_5")
    wait_5 = time.time() - t5
    print(f"[Processor] Got item_5 in {wait_5:.3f}s")
    processed_5 = process_item(5, item_5)
    _streaming_publisher.publish('processed_5', processed_5)
    print(f"[Processor] processed_5 ready at {time.time() - start_time:.3f}s")
    
    total_time = time.time() - start_time
    print(f"[Processor] Completed in {total_time:.3f}s")
    
    return {
        "processed_1": processed_1,
        "processed_2": processed_2,
        "processed_3": processed_3,
        "processed_4": processed_4,
        "processed_5": processed_5
    }
