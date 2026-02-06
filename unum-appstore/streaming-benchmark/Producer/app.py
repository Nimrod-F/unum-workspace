"""
Producer Function - Computes 3 fields sequentially

This function demonstrates Partial Parameter Streaming by computing
3 independent fields, each taking approximately 1 second.

With streaming enabled:
- After field1 is computed, Consumer is invoked immediately
- field2 and field3 are sent as future references
- Producer continues computing while Consumer processes field1

Without streaming:
- All 3 fields computed (3 seconds total)
- Then Consumer is invoked
"""

import json
import time
import math
import hashlib
import random


def compute_field1(data):
    """
    Compute statistical analysis - takes ~1 second
    """
    start = time.time()
    
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = math.sqrt(variance)
    
    # Simulate heavy computation
    for _ in range(100000):
        _ = math.sin(mean) * math.cos(variance)
    
    # Add controlled delay to ensure ~1 second
    elapsed = time.time() - start
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    
    result = {
        "mean": round(mean, 4),
        "std_dev": round(std_dev, 4),
        "computed_at": time.time()
    }
    
    print(f"[Producer] field1 computed in {time.time() - start:.3f}s")
    return result


def compute_field2(data):
    """
    Compute trend analysis - takes ~1 second
    """
    start = time.time()
    
    n = len(data)
    
    # Linear regression (trend)
    x_mean = (n - 1) / 2
    y_mean = sum(data) / n
    
    numerator = sum((i - x_mean) * (data[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    
    slope = numerator / denominator if denominator != 0 else 0
    intercept = y_mean - slope * x_mean
    
    # Simulate heavy computation
    for _ in range(100000):
        _ = math.tan(slope + 0.001) * math.log(abs(intercept) + 1)
    
    # Add controlled delay
    elapsed = time.time() - start
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    
    result = {
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "trend": "up" if slope > 0 else "down",
        "computed_at": time.time()
    }
    
    print(f"[Producer] field2 computed in {time.time() - start:.3f}s")
    return result


def compute_field3(data):
    """
    Compute fingerprint/hash - takes ~1 second
    """
    start = time.time()
    
    # Create data fingerprint
    data_str = json.dumps(data, sort_keys=True)
    
    # Intensive hashing
    hash_val = data_str
    for _ in range(50000):
        hash_val = hashlib.sha256(hash_val.encode()).hexdigest()
    
    # Add controlled delay
    elapsed = time.time() - start
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    
    result = {
        "fingerprint": hash_val[:32],
        "data_size": len(data),
        "computed_at": time.time()
    }
    
    print(f"[Producer] field3 computed in {time.time() - start:.3f}s")
    return result


def lambda_handler(event, context):
    """
    Producer handler - computes 3 fields sequentially.
    
    With Partial Parameter Streaming:
    - After field1, Consumer is invoked with {field1: value, field2: future, field3: future}
    - Producer continues to compute field2 and field3
    - Each field is published to datastore when ready
    - Consumer resolves futures on-demand
    """
    
    start_time = time.time()
    print(f"[Producer] Starting at {start_time}")
    
    # Extract or generate input data
    if isinstance(event, dict) and "data" in event:
        data = event["data"]
    else:
        # Generate sample data
        random.seed(42)
        data = [random.gauss(100, 15) for _ in range(100)]
    
    # =========================================
    # Compute field1 - Statistical Analysis
    # =========================================
    statistical = compute_field1(data)
    
    # =========================================
    # Compute field2 - Trend Analysis
    # =========================================
    trend = compute_field2(data)
    
    # =========================================
    # Compute field3 - Fingerprint
    # =========================================
    fingerprint = compute_field3(data)
    
    # =========================================
    # Assemble result
    # =========================================
    result = {
        "statistical": statistical,
        "trend": trend,
        "fingerprint": fingerprint,
        "producer_total_time": round(time.time() - start_time, 4)
    }
    
    print(f"[Producer] Completed in {time.time() - start_time:.3f}s")
    
    return result
