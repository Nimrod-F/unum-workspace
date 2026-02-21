"""
SlowChainEnd - Sequential chain step 3 (1500ms)

Invoice generation - final step in slow sequential chain.
Total chain time: 500ms + 1000ms + 1500ms = 3000ms

In CLASSIC mode, this function completes last and triggers the Aggregator.
In FUTURE mode, FastProcessor already triggered Aggregator ~2900ms ago.
"""
import json
import time
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Slowest operation: 1500ms (invoice generation)
PROCESSING_DELAY_MS = 1500


def lambda_handler(event, context):
    """
    Generate invoice PDF and prepare order confirmation.

    This is step 3 (final) of the sequential slow chain.
    In CLASSIC mode, this is the LAST function to complete and triggers Aggregator.
    """
    start_time = time.time()

    # Extract order details from previous steps
    order_id = event.get("order_id", "unknown")
    customer_id = event.get("customer_id", "unknown")
    items = event.get("items", [])
    payment_result = event.get("payment_result", {})
    shipping_result = event.get("shipping_result", {})

    logger.info(json.dumps({
        "event": "slow_chain_end_begin",
        "order_id": order_id,
        "chain_step": 3,
        "operation": "invoice_generation"
    }))

    print(f"[SlowChainEnd] Starting - Chain step 3/3 ({PROCESSING_DELAY_MS}ms)")
    print(f"[SlowChainEnd] Order: {order_id}")
    print(f"[SlowChainEnd] Operation: Invoice generation (PDF)")

    # Simulate invoice PDF generation (heavy computation)
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    # Invoice generation result
    invoice_result = {
        "invoice_id": f"INV-{int(time.time())}",
        "invoice_url": f"https://invoices.example.com/INV-{int(time.time())}.pdf",
        "total_amount": payment_result.get("amount_charged", 0) + shipping_result.get("shipping_cost", 0),
        "invoice_date": time.strftime("%Y-%m-%d"),
        "pdf_size_kb": 245
    }

    total_time = (time.time() - start_time) * 1000

    result = {
        "function": "SlowChainEnd",
        "branch": "slow_chain",
        "chain_step": 3,
        "order_id": order_id,
        "customer_id": customer_id,
        "items": items,
        "operation": "invoice_generation",
        "status": "complete",
        "payment_result": payment_result,
        "shipping_result": shipping_result,
        "invoice_result": invoice_result,
        "processing_time_ms": int(total_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "slow_chain_end_complete",
        "order_id": order_id,
        "chain_step": 3,
        "processing_time_ms": int(total_time),
        "invoice_id": invoice_result["invoice_id"]
    }))

    print(f"[SlowChainEnd] COMPLETE in {total_time:.2f}ms")
    print(f"[SlowChainEnd] Invoice: {invoice_result['invoice_id']}")
    print(f"[SlowChainEnd] Total chain time: ~3000ms (500+1000+1500)")
    print(f"[SlowChainEnd] In CLASSIC mode: This triggers Aggregator")
    print(f"[SlowChainEnd] In FUTURE mode: Aggregator already started ~2900ms ago")

    return result


if __name__ == '__main__':
    # Local test
    test_event = {
        "order_id": "ORD-TEST-001",
        "customer_id": "CUST-TEST",
        "items": [{"sku": "ITEM-001", "quantity": 2}],
        "payment_result": {
            "payment_authorized": True,
            "transaction_id": "TXN-12345",
            "amount_charged": 129.99
        },
        "shipping_result": {
            "shipping_method": "EXPRESS",
            "carrier": "DHL",
            "shipping_cost": 12.99
        }
    }

    result = lambda_handler(test_event, None)
    print(f"\nResult: {json.dumps(result, indent=2)}")
