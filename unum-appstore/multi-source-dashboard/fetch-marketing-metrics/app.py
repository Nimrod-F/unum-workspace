"""
Fetch Marketing Metrics - Simulates fetching data from marketing analytics
Fixed Latency: 5.0s (Staircase Benchmark: Step 3/6)
"""
import time
import random
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Fetch marketing metrics with FIXED latency for benchmarking.
    """
    start_time = time.time()
    function_name = "FetchMarketingMetrics"

    # --- BENCHMARK CONFIGURATION ---
    # This is Step 3 in the Staircase.
    # It finishes at T+5s.
    # In Future Mode, the Aggregator will pick this up mid-stream,
    # proving it can handle tasks arriving at varied intervals.
    delay = 5.0
    time.sleep(delay)
    # -------------------------------

    # Generate realistic mock data
    result = {
        "source": "marketing",
        "status": "success",
        "data": {
            "impressions": random.randint(50000, 500000),
            "clicks": random.randint(2000, 15000),
            "conversions": random.randint(100, 800),
            "ctr": round(random.uniform(2.5, 5.5), 2),
            "conversion_rate": round(random.uniform(3.0, 8.0), 2),
            "campaign_performance": [
                {
                    "campaign": "Summer Sale",
                    "spend": round(random.uniform(5000, 15000), 2),
                    "conversions": random.randint(50, 300),
                    "roi": round(random.uniform(1.5, 4.5), 2)
                },
                {
                    "campaign": "Product Launch",
                    "spend": round(random.uniform(8000, 20000), 2),
                    "conversions": random.randint(80, 400),
                    "roi": round(random.uniform(2.0, 5.0), 2)
                }
            ],
            "channel_breakdown": {
                "email": {"impressions": random.randint(10000, 50000), "conversions": random.randint(50, 200)},
                "social": {"impressions": random.randint(20000, 100000), "conversions": random.randint(80, 300)},
                "search": {"impressions": random.randint(15000, 80000), "conversions": random.randint(100, 400)},
                "display": {"impressions": random.randint(5000, 40000), "conversions": random.randint(20, 100)}
            }
        },
        "timestamp": time.time(),
        "latency_ms": (time.time() - start_time) * 1000
    }

    # Log for metrics collection
    logger.info(json.dumps({
        "event": "function_complete",
        "function": function_name,
        "latency_ms": result["latency_ms"],
        "simulated_delay_ms": delay * 1000,
        "status": "success",
        "note": "BENCHMARK_STEP_3"
    }))

    return result