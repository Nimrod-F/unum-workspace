"""
Fetch Sales Data - Simulates fetching sales data from internal database
Latency: 100-400ms (fast internal query)
"""
import time
import random
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Fetch sales data with simulated latency (100-400ms).

    Returns realistic sales metrics including revenue, transactions, and order values.
    """
    start_time = time.time()
    function_name = "FetchSalesData"

    # Simulate realistic latency (100-400ms for internal DB query)
    delay = random.uniform(0.1, 0.4)
    time.sleep(delay)

    # Generate realistic mock data
    result = {
        "source": "sales",
        "status": "success",
        "data": {
            "total_revenue": round(random.uniform(50000, 250000), 2),
            "transactions": random.randint(500, 2500),
            "avg_order_value": round(random.uniform(75, 180), 2),
            "top_products": [
                {"id": "P001", "name": "Product A", "sales": random.randint(100, 500)},
                {"id": "P002", "name": "Product B", "sales": random.randint(80, 400)},
                {"id": "P003", "name": "Product C", "sales": random.randint(60, 300)}
            ],
            "sales_by_region": {
                "north": round(random.uniform(10000, 50000), 2),
                "south": round(random.uniform(10000, 50000), 2),
                "east": round(random.uniform(10000, 50000), 2),
                "west": round(random.uniform(10000, 50000), 2)
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
        "status": "success"
    }))

    return result
