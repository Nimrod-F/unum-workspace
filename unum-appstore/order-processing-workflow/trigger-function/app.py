"""
TriggerFunction - Entry point for order processing workflow

Receives order request and fans out to two branches:
1. FastProcessor - Quick inventory check (fast branch)
2. SlowChainStart - Payment/shipping/invoice processing (slow sequential chain)
"""
import json
import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Initiate order processing workflow.

    Input:
    {
        "order_id": "ORD-12345",
        "customer_id": "CUST-001",
        "items": [...]
    }

    Output: Order data dict passed to both Scalar fan-out branches
    """
    start_time = time.time()

    # Extract order details
    order_id = event.get("order_id", f"ORD-{int(time.time())}")
    customer_id = event.get("customer_id", "CUST-UNKNOWN")
    items = event.get("items", [])

    logger.info(json.dumps({
        "event": "order_received",
        "order_id": order_id,
        "customer_id": customer_id,
        "item_count": len(items)
    }))

    print(f"[TriggerFunction] Processing order: {order_id}")
    print(f"[TriggerFunction] Customer: {customer_id}, Items: {len(items)}")

    # Prepare order data - both Scalar fan-out branches receive this same dict
    order_data = {
        "order_id": order_id,
        "customer_id": customer_id,
        "items": items,
        "trigger_timestamp": time.time()
    }

    execution_time = (time.time() - start_time) * 1000

    logger.info(json.dumps({
        "event": "fanout_initiated",
        "order_id": order_id,
        "branches": ["fast_processor", "slow_chain"],
        "execution_time_ms": int(execution_time)
    }))

    print(f"[TriggerFunction] Fan-out initiated in {execution_time:.2f}ms")
    print(f"[TriggerFunction] Branch 1: FastProcessor (inventory check)")
    print(f"[TriggerFunction] Branch 2: SlowChainStart (payment validation)")

    # Return single dict - unum Scalar fan-out sends this to both branches
    return order_data


if __name__ == '__main__':
    # Local test
    test_event = {
        "order_id": "ORD-TEST-001",
        "customer_id": "CUST-TEST",
        "items": [
            {"sku": "ITEM-001", "quantity": 2},
            {"sku": "ITEM-002", "quantity": 1}
        ]
    }

    result = lambda_handler(test_event, None)
    print(f"\nResult: {json.dumps(result, indent=2)}")
