"""
Data Source - Simulates a data source with configurable latency.

Each source sleeps for a different amount of time to simulate
varying network/processing latencies, then returns data.
"""
import time
import json


def lambda_handler(event, context):
    """
    Process a data source request.
    
    Expected event:
    {
        "source_id": 1,       # Which source (1-5)
        "delay_seconds": 1.0, # How long to sleep
        "data_size": 100      # Size of data to return
    }
    """
    source_id = event.get('source_id', 1)
    delay_seconds = event.get('delay_seconds', 1.0)
    data_size = event.get('data_size', 100)
    
    start_time = time.time()
    
    # Simulate work/network latency
    time.sleep(delay_seconds)
    
    # Generate some data
    data = {
        "source_id": source_id,
        "values": list(range(data_size)),
        "sum": sum(range(data_size)),
        "processing_time_ms": int((time.time() - start_time) * 1000),
        "timestamp": time.time()
    }
    
    return data


if __name__ == '__main__':
    # Test locally
    result = lambda_handler({
        "source_id": 1,
        "delay_seconds": 0.5,
        "data_size": 10
    }, None)
    print(json.dumps(result, indent=2))
