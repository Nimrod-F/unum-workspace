"""
Fan-Out - Triggers multiple data sources with different latencies.

This simulates a real-world scenario where you query multiple 
backend services that have different response times.
"""
import json


def lambda_handler(event, context):
    """
    Create fan-out payloads for 5 sources with increasing latencies.
    
    Returns a list of payloads, one for each source.
    Unum will invoke the Source function for each payload.
    """
    
    # Define sources with different latencies
    # Sources 3 and 4 return EARLY to demonstrate background polling caching
    # When we wait for inputs[0], inputs[3] and [4] will already resolve in background
    sources = [
        {"source_id": 1, "delay_seconds": 2.0, "data_size": 100},   # index 0 → 2 sec
        {"source_id": 2, "delay_seconds": 3.0, "data_size": 100},   # index 1 → 3 sec
        {"source_id": 3, "delay_seconds": 4.0, "data_size": 100},   # index 2 → 4 sec
        {"source_id": 4, "delay_seconds": 0.3, "data_size": 100},   # index 3 → 0.3 sec (EARLY!)
        {"source_id": 5, "delay_seconds": 0.5, "data_size": 100},   # index 4 → 0.5 sec (EARLY!)
    ]
    
    return sources


if __name__ == '__main__':
    result = lambda_handler({}, None)
    print(json.dumps(result, indent=2))
