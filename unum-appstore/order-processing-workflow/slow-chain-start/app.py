"""
SlowChainStart - Sequential chain step 1 (500ms)

Payment validation - first step in slow sequential chain.
Total chain time: 500ms + 1000ms + 1500ms = 3000ms
"""
import json
import time
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Slow operation: 500ms (payment validation)
PROCESSING_DELAY_MS = 500


def lambda_handler(event, context):
    """
    Validate payment method and process authorization.

    This is step 1 of the sequential slow chain.
    """
    start_time = time.time()

    # Extract order details
    order_id = event.get("order_id", "unknown")
    customer_id = event.get("customer_id", "unknown")
    items = event.get("items", [])

    logger.info(json.dumps({
        "event": "slow_chain_start_begin",
        "order_id": order_id,
        "chain_step": 1,
        "operation": "payment_validation"
    }))

    print(f"[SlowChainStart] Starting - Chain step 1/3 ({PROCESSING_DELAY_MS}ms)")
    print(f"[SlowChainStart] Order: {order_id}")
    print(f"[SlowChainStart] Operation: Payment validation")

    # Simulate payment validation (external API call)
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    # Payment validation result
    payment_result = {
        "payment_authorized": True,
        "transaction_id": f"TXN-{int(time.time())}",
        "payment_method": "CREDIT_CARD",
        "amount_charged": 129.99
    }

    total_time = (time.time() - start_time) * 1000

    result = {
        "function": "SlowChainStart",
        "branch": "slow_chain",
        "chain_step": 1,
        "order_id": order_id,
        "customer_id": customer_id,
        "items": items,
        "operation": "payment_validation",
        "status": "complete",
        "payment_result": payment_result,
        "processing_time_ms": int(total_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "slow_chain_start_complete",
        "order_id": order_id,
        "chain_step": 1,
        "processing_time_ms": int(total_time),
        "payment_authorized": payment_result["payment_authorized"]
    }))

    print(f"[SlowChainStart] COMPLETE in {total_time:.2f}ms")
    print(f"[SlowChainStart] Payment authorized: {payment_result['transaction_id']}")
    print(f"[SlowChainStart] Next: SlowChainMid (shipping calculation)")

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
