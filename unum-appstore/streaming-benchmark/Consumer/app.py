"""
Consumer Function - Processes fields from Producer

This function receives the Producer's output fields.
With streaming, some fields may be future references that
are resolved lazily when accessed.

The key demonstration:
- field1 is available immediately (Producer computed it first)
- field2 and field3 might be futures (resolve when accessed)
- Consumer can start processing field1 while waiting for others
"""

import json
import time


def process_statistical(stats):
    """Process statistical data"""
    start = time.time()
    print(f"[Consumer] Processing statistical data: {stats}")
    
    # Simulate processing
    time.sleep(0.2)
    
    result = {
        "mean_category": "high" if stats["mean"] > 100 else "low",
        "volatility": "high" if stats["std_dev"] > 10 else "low",
        "processed_at": time.time()
    }
    
    print(f"[Consumer] Statistical processed in {time.time() - start:.3f}s")
    return result


def process_trend(trend):
    """Process trend data"""
    start = time.time()
    print(f"[Consumer] Processing trend data: {trend}")
    
    # Simulate processing
    time.sleep(0.2)
    
    result = {
        "direction": trend["trend"],
        "strength": "strong" if abs(trend["slope"]) > 0.1 else "weak",
        "processed_at": time.time()
    }
    
    print(f"[Consumer] Trend processed in {time.time() - start:.3f}s")
    return result


def process_fingerprint(fp):
    """Process fingerprint data"""
    start = time.time()
    print(f"[Consumer] Processing fingerprint data: {fp}")
    
    # Simulate processing
    time.sleep(0.2)
    
    result = {
        "short_hash": fp["fingerprint"][:8],
        "size_category": "large" if fp["data_size"] > 50 else "small",
        "processed_at": time.time()
    }
    
    print(f"[Consumer] Fingerprint processed in {time.time() - start:.3f}s")
    return result


def lambda_handler(event, context):
    """
    Consumer handler - processes Producer's output.
    
    With Partial Parameter Streaming:
    - Invoked as soon as Producer has field1 ready
    - field1 is a real value, field2/field3 are futures
    - When we access event["trend"], it blocks until field2 is ready
    - This allows us to START earlier, even if total time is similar
    
    The key metric is TIME TO FIRST BYTE of processing, not total time.
    """
    
    start_time = time.time()
    print(f"[Consumer] Invoked at {start_time}")
    
    # Record when we received the invocation
    invocation_time = time.time()
    
    # =========================================
    # Process field1 (statistical) - Available immediately!
    # =========================================
    t1 = time.time()
    statistical = event["statistical"]  # Immediate if streaming worked
    statistical_latency = time.time() - t1
    print(f"[Consumer] Got statistical in {statistical_latency:.3f}s")
    
    processed_stats = process_statistical(statistical)
    
    # =========================================
    # Process field2 (trend) - May block waiting for future
    # =========================================
    t2 = time.time()
    trend = event["trend"]  # May block if future
    trend_latency = time.time() - t2
    print(f"[Consumer] Got trend in {trend_latency:.3f}s (blocking indicates future resolution)")
    
    processed_trend = process_trend(trend)
    
    # =========================================
    # Process field3 (fingerprint) - May block waiting for future
    # =========================================
    t3 = time.time()
    fingerprint = event["fingerprint"]  # May block if future
    fingerprint_latency = time.time() - t3
    print(f"[Consumer] Got fingerprint in {fingerprint_latency:.3f}s")
    
    processed_fp = process_fingerprint(fingerprint)
    
    # =========================================
    # Assemble result
    # =========================================
    result = {
        "processed_stats": processed_stats,
        "processed_trend": processed_trend,
        "processed_fingerprint": processed_fp,
        "timing": {
            "invocation_time": invocation_time,
            "statistical_latency": round(statistical_latency, 4),
            "trend_latency": round(trend_latency, 4),
            "fingerprint_latency": round(fingerprint_latency, 4),
            "consumer_total_time": round(time.time() - start_time, 4)
        }
    }
    
    print(f"[Consumer] Completed in {time.time() - start_time:.3f}s")
    
    return result
