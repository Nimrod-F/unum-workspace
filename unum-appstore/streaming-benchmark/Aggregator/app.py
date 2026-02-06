"""
Aggregator Function - Final aggregation

This is the final function in the pipeline.
It receives processed data from Consumer and produces the final result.
"""

import json
import time


def lambda_handler(event, context):
    """
    Aggregator handler - produces final result.
    """
    
    start_time = time.time()
    print(f"[Aggregator] Invoked at {start_time}")
    
    # Extract timing from Consumer
    timing = event.get("timing", {})
    
    # Create final summary
    result = {
        "summary": {
            "stats_category": event.get("processed_stats", {}).get("mean_category", "unknown"),
            "volatility": event.get("processed_stats", {}).get("volatility", "unknown"),
            "trend_direction": event.get("processed_trend", {}).get("direction", "unknown"),
            "trend_strength": event.get("processed_trend", {}).get("strength", "unknown"),
            "data_hash": event.get("processed_fingerprint", {}).get("short_hash", "unknown")
        },
        "pipeline_timing": {
            "consumer_invocation_time": timing.get("invocation_time", 0),
            "consumer_total_time": timing.get("consumer_total_time", 0),
            "aggregator_time": round(time.time() - start_time, 4),
            "aggregator_end_time": time.time()
        },
        "workflow_complete": True
    }
    
    print(f"[Aggregator] Completed in {time.time() - start_time:.3f}s")
    print(f"[Aggregator] Final result: {json.dumps(result, indent=2)}")
    
    return result
