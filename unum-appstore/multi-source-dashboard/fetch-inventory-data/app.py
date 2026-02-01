"""
Fetch Inventory Data - Simulates fetching inventory data from warehouse system
Latency: 150-500ms (internal system with moderate latency)
"""
import time
import random
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Fetch inventory data with simulated latency (150-500ms).

    Returns stock levels, warehouse status, and reorder alerts.
    """
    start_time = time.time()
    function_name = "FetchInventoryData"

    # Simulate realistic latency (150-500ms for warehouse system)
    delay = random.uniform(0.15, 0.5)
    time.sleep(delay)

    # Generate realistic mock data
    result = {
        "source": "inventory",
        "status": "success",
        "data": {
            "total_items": random.randint(5000, 15000),
            "low_stock_items": random.randint(50, 200),
            "out_of_stock_items": random.randint(10, 50),
            "warehouse_status": {
                "warehouse_a": {
                    "capacity_pct": round(random.uniform(60, 95), 1),
                    "items": random.randint(2000, 6000)
                },
                "warehouse_b": {
                    "capacity_pct": round(random.uniform(50, 90), 1),
                    "items": random.randint(1500, 5000)
                },
                "warehouse_c": {
                    "capacity_pct": round(random.uniform(40, 85), 1),
                    "items": random.randint(1000, 4000)
                }
            },
            "reorder_alerts": [
                {"sku": f"SKU{i:04d}", "current_stock": random.randint(1, 20), "reorder_point": 50}
                for i in range(random.randint(3, 8))
            ]
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
