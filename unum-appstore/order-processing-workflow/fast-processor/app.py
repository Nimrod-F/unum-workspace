"""
FastProcessor - FAST branch (100ms)

Quick inventory check - completes first in FUTURE mode.
This branch triggers the Aggregator immediately, hiding cold start behind slow chain.
"""
import json
import time
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Fast operation: 100ms
PROCESSING_DELAY_MS = 100


def lambda_handler(event, context):
    """
    Perform quick inventory check.

    This is the FASTEST branch and should trigger the Aggregator first
    in FUTURE_BASED mode, allowing cold start to overlap with slow chain execution.
    """
    start_time = time.time()

    # Extract order details
    order_id = event.get("order_id", "unknown")
    customer_id = event.get("customer_id", "unknown")
    items = event.get("items", [])

    logger.info(json.dumps({
        "event": "fast_processor_start",
        "order_id": order_id,
        "branch": "fast_processor"
    }))

    print(f"[FastProcessor] Starting - FASTEST branch ({PROCESSING_DELAY_MS}ms)")
    print(f"[FastProcessor] Order: {order_id}, Items: {len(items)}")

    # Simulate quick inventory lookup (local cache check)
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    # Check inventory availability
    # SIMULATE_NO_STOCK env var allows benchmark to test short-circuit scenario
    simulate_no_stock = os.environ.get('SIMULATE_NO_STOCK', 'false').lower() == 'true'

    if simulate_no_stock:
        inventory_status = {
            "available": False,
            "reason": "out_of_stock",
            "warehouse": "WH-EU-01",
            "restock_date": "2024-02-01"
        }
        print(f"[FastProcessor] SIMULATE_NO_STOCK=true - returning unavailable")
    else:
        inventory_status = {
            "available": True,
            "warehouse": "WH-EU-01",
            "estimated_ship_date": "2024-01-15"
        }

    total_time = (time.time() - start_time) * 1000

    result = {
        "function": "FastProcessor",
        "branch": "fast_processor",
        "order_id": order_id,
        "customer_id": customer_id,
        "operation": "inventory_check",
        "status": "complete",
        "inventory_status": inventory_status,
        "processing_time_ms": int(total_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "fast_processor_complete",
        "order_id": order_id,
        "processing_time_ms": int(total_time),
        "inventory_available": inventory_status["available"]
    }))

    print(f"[FastProcessor] COMPLETE in {total_time:.2f}ms")
    print(f"[FastProcessor] Inventory: {inventory_status['available']}")
    print(f"[FastProcessor] This should trigger Aggregator FIRST in FUTURE mode")

    return result


if __name__ == '__main__':
    # Local test
    test_event = {
        "order_id": "ORD-TEST-001",
        "customer_id": "CUST-TEST",
        "items": [{"sku": "ITEM-001", "quantity": 2}]
    }

    result = lambda_handler(test_event, None)
    print(f"\nResult: {json.dumps(result, indent=2)}")
