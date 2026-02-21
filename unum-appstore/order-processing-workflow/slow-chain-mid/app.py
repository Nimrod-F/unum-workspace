"""
SlowChainMid - Sequential chain step 2 (1000ms)

Shipping calculation - middle step in slow sequential chain.
Total chain time: 500ms + 1000ms + 1500ms = 3000ms
"""
import json
import time
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Slow operation: 1000ms (shipping calculation)
PROCESSING_DELAY_MS = 1000


def lambda_handler(event, context):
    """
    Calculate shipping costs and delivery estimates.

    This is step 2 of the sequential slow chain.
    """
    start_time = time.time()

    # Extract order details from previous step
    order_id = event.get("order_id", "unknown")
    customer_id = event.get("customer_id", "unknown")
    items = event.get("items", [])
    payment_result = event.get("payment_result", {})

    logger.info(json.dumps({
        "event": "slow_chain_mid_begin",
        "order_id": order_id,
        "chain_step": 2,
        "operation": "shipping_calculation"
    }))

    print(f"[SlowChainMid] Starting - Chain step 2/3 ({PROCESSING_DELAY_MS}ms)")
    print(f"[SlowChainMid] Order: {order_id}")
    print(f"[SlowChainMid] Operation: Shipping calculation")

    # Simulate complex shipping calculation (routing algorithm)
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    # Shipping calculation result
    shipping_result = {
        "shipping_method": "EXPRESS",
        "carrier": "DHL",
        "estimated_delivery": "2024-01-18",
        "shipping_cost": 12.99,
        "tracking_number": f"TRK-{int(time.time())}"
    }

    total_time = (time.time() - start_time) * 1000

    result = {
        "function": "SlowChainMid",
        "branch": "slow_chain",
        "chain_step": 2,
        "order_id": order_id,
        "customer_id": customer_id,
        "items": items,
        "operation": "shipping_calculation",
        "status": "complete",
        "payment_result": payment_result,
        "shipping_result": shipping_result,
        "processing_time_ms": int(total_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "slow_chain_mid_complete",
        "order_id": order_id,
        "chain_step": 2,
        "processing_time_ms": int(total_time),
        "shipping_method": shipping_result["shipping_method"]
    }))

    print(f"[SlowChainMid] COMPLETE in {total_time:.2f}ms")
    print(f"[SlowChainMid] Shipping: {shipping_result['carrier']} - {shipping_result['shipping_method']}")
    print(f"[SlowChainMid] Next: SlowChainEnd (invoice generation)")

    return result


if __name__ == '__main__':
    # Local test
    test_event = {
        "order_id": "ORD-TEST-001",
        "customer_id": "CUST-TEST",
        "items": [{"sku": "ITEM-001", "quantity": 2}],
        "payment_result": {
            "payment_authorized": True,
            "transaction_id": "TXN-12345"
        }
    }

    result = lambda_handler(test_event, None)
    print(f"\nResult: {json.dumps(result, indent=2)}")
