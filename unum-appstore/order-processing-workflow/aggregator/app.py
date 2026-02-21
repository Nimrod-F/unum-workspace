"""
Aggregator - Fan-in terminal function

Aggregates results from:
1. FastProcessor (fast branch - 100ms)
2. SlowChainEnd (end of slow chain - 3000ms total)

CLASSIC mode: Triggered by SlowChainEnd (slowest, ~3000ms after start)
FUTURE mode: Triggered by FastProcessor (fastest, ~100ms after start)

The key benefit of FUTURE mode is that the Aggregator's cold start (~200ms)
overlaps with the slow chain execution, effectively hiding it from the critical path.
"""
import json
import time
import logging
import asyncio
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Aggregate results from FastProcessor and SlowChainEnd.

    In CLASSIC mode: Triggered by LAST function to complete (SlowChainEnd)
    In FUTURE mode: Triggered by FIRST function to complete (FastProcessor)

    Input: Array-like object with results from both branches
    Output: Aggregated order confirmation
    """
    start_time = time.time()

    logger.info(json.dumps({
        "event": "aggregator_start",
        "mode": "FUTURE_BASED" if os.environ.get("UNUM_FUTURE_BASED") == "true" else "CLASSIC"
    }))

    print(f"[Aggregator] Starting - Fan-in aggregator")
    print(f"[Aggregator] Mode: {'FUTURE_BASED' if os.environ.get('UNUM_FUTURE_BASED') == 'true' else 'CLASSIC'}")

    # Check if inputs support async (Future-Based mode)
    is_async = hasattr(event, 'get_async') or hasattr(event, 'get_all_async')

    if is_async:
        logger.info("[Aggregator] Using FUTURE_BASED mode with async inputs")
        print("[Aggregator] Using FUTURE_BASED mode - async/await")
        result = asyncio.run(process_inputs_async(event, start_time))
    else:
        logger.info("[Aggregator] Using CLASSIC/EAGER mode with sync inputs")
        print("[Aggregator] Using CLASSIC mode - synchronous")
        result = process_inputs_sync(event, start_time)

    return result


async def process_inputs_async(inputs, start_time):
    """
    Async processing for Future-Based mode.

    Awaits both inputs non-blockingly. FastProcessor should already be resolved,
    SlowChainEnd will resolve as futures complete.

    SHORT-CIRCUIT OPTIMIZATION: If FastProcessor returns inventory_available=false,
    we can immediately reject the order WITHOUT waiting for the slow chain!
    This is a key benefit of FUTURE_BASED mode - early termination saves ~3000ms.
    """
    print("[Aggregator] Async mode: Starting to fetch inputs")

    all_data = []
    pre_resolved = 0
    branch_timings = []

    # FastProcessor should be at index 0 - fetch it first
    fetch_start = time.time()
    fast_data = await inputs.get_async(0)
    fetch_duration_ms = (time.time() - fetch_start) * 1000

    if fetch_duration_ms < 10:
        pre_resolved += 1
        print(f"[Aggregator] Input 0 ({fast_data.get('function', 'unknown')}): PRE-RESOLVED")
    else:
        print(f"[Aggregator] Input 0 ({fast_data.get('function', 'unknown')}): Awaited {fetch_duration_ms:.2f}ms")

    branch_timings.append({
        "index": 0,
        "function": fast_data.get('function', 'unknown'),
        "fetch_duration_ms": fetch_duration_ms,
        "pre_resolved": fetch_duration_ms < 10
    })
    all_data.append(fast_data)

    # CHECK FOR SHORT-CIRCUIT: If inventory unavailable, reject immediately!
    inventory_status = fast_data.get('inventory_status', {})
    if not inventory_status.get('available', True):
        print(f"[Aggregator] SHORT-CIRCUIT: Inventory unavailable!")
        print(f"[Aggregator] Rejecting order immediately - NOT waiting for slow chain!")

        # Return early without waiting for SlowChainEnd
        return merge_data_short_circuit(
            fast_data, start_time, pre_resolved, branch_timings
        )

    # Inventory available - need to wait for slow chain to complete the order
    print("[Aggregator] Inventory available - awaiting slow chain for payment/shipping")

    fetch_start = time.time()
    slow_data = await inputs.get_async(1)
    fetch_duration_ms = (time.time() - fetch_start) * 1000

    if fetch_duration_ms < 10:
        pre_resolved += 1
        print(f"[Aggregator] Input 1 ({slow_data.get('function', 'unknown')}): PRE-RESOLVED")
    else:
        print(f"[Aggregator] Input 1 ({slow_data.get('function', 'unknown')}): Awaited {fetch_duration_ms:.2f}ms")

    branch_timings.append({
        "index": 1,
        "function": slow_data.get('function', 'unknown'),
        "fetch_duration_ms": fetch_duration_ms,
        "pre_resolved": fetch_duration_ms < 10
    })
    all_data.append(slow_data)

    return merge_data(all_data, start_time, pre_resolved, branch_timings)


def process_inputs_sync(inputs, start_time):
    """
    Sync processing for CLASSIC/EAGER modes.

    In CLASSIC: All inputs already resolved (blocking wait already happened)
    In EAGER: Blocks synchronously on each access
    """
    print("[Aggregator] Sync mode: Accessing inputs")

    all_data = []
    pre_resolved = 0
    branch_timings = []

    # Access both inputs
    for i, data in enumerate(inputs):
        access_start = time.time()

        # In CLASSIC: instant (all pre-resolved)
        # In EAGER: blocks synchronously
        all_data.append(data)

        access_time_ms = (time.time() - access_start) * 1000

        if access_time_ms < 10:
            pre_resolved += 1

        branch_timings.append({
            "index": i,
            "function": data.get('function', 'unknown'),
            "access_time_ms": access_time_ms,
            "pre_resolved": access_time_ms < 10
        })

        print(f"[Aggregator] Input {i} ({data.get('function', 'unknown')}): {access_time_ms:.2f}ms")

    return merge_data(all_data, start_time, pre_resolved, branch_timings)


def merge_data_short_circuit(fast_result, start_time, pre_resolved, branch_timings):
    """
    Short-circuit aggregation when inventory is unavailable.

    In FUTURE_BASED mode, we can immediately reject the order without
    waiting for the slow chain (payment/shipping/invoice processing).
    This saves ~3000ms compared to CLASSIC mode!
    """
    aggregation_time = (time.time() - start_time) * 1000

    order_id = fast_result.get('order_id', 'unknown')
    inventory_status = fast_result.get('inventory_status', {})

    # Build rejection response
    order_rejection = {
        "status": "rejected",
        "reason": "inventory_unavailable",
        "order_id": order_id,
        "inventory_status": inventory_status,
        "payment_result": None,  # Not processed - short-circuited!
        "shipping_result": None,
        "invoice_result": None,
        "rejection_timestamp": time.time()
    }

    result = {
        "status": "rejected",
        "short_circuited": True,
        "order_rejection": order_rejection,
        "total_branches": 1,  # Only FastProcessor was used
        "pre_resolved_count": pre_resolved,
        "aggregation_time_ms": int(aggregation_time),
        "branch_timings": branch_timings,
        "branch_processing_times": {
            "fast_processor_ms": fast_result.get('processing_time_ms', 0),
            "slow_chain_end_ms": 0  # Not waited for!
        },
        "timing_variance_ms": 0,
        "mode": "FUTURE_BASED",
        "time_saved_ms": 3000,  # Approximate time saved by not waiting
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "aggregator_short_circuit",
        "order_id": order_id,
        "aggregation_time_ms": int(aggregation_time),
        "reason": "inventory_unavailable"
    }))

    print(f"[Aggregator] SHORT-CIRCUIT COMPLETE in {aggregation_time:.2f}ms")
    print(f"[Aggregator] Order REJECTED - saved ~3000ms by not waiting for slow chain!")
    print(f"[Aggregator] FUTURE mode benefit: Early termination on failure condition")

    return result


def merge_data(all_data, start_time, pre_resolved, branch_timings):
    """
    Core aggregation logic - merges results from both branches.
    """
    aggregation_time = (time.time() - start_time) * 1000

    # Extract results from each branch
    fast_result = None
    slow_result = None

    for data in all_data:
        if data.get('function') == 'FastProcessor':
            fast_result = data
        elif data.get('function') == 'SlowChainEnd':
            slow_result = data

    # Build order confirmation
    order_id = fast_result.get('order_id') if fast_result else 'unknown'
    inventory_status = fast_result.get('inventory_status', {}) if fast_result else {}

    # Check if inventory is available (for CLASSIC mode - can't short-circuit but still reject)
    inventory_available = inventory_status.get('available', True)

    if not inventory_available:
        # CLASSIC mode: Already waited for slow chain, but still rejecting due to inventory
        order_rejection = {
            "status": "rejected",
            "reason": "inventory_unavailable",
            "order_id": order_id,
            "inventory_status": inventory_status,
            "payment_result": slow_result.get('payment_result') if slow_result else None,
            "shipping_result": slow_result.get('shipping_result') if slow_result else None,
            "invoice_result": slow_result.get('invoice_result') if slow_result else None,
            "rejection_timestamp": time.time(),
            "note": "CLASSIC mode - waited for slow chain despite inventory being unavailable"
        }
        print(f"[Aggregator] CLASSIC mode: Inventory unavailable but already waited for slow chain")
    else:
        order_rejection = None

    order_confirmation = {
        "status": "confirmed" if inventory_available else "rejected",
        "order_id": order_id,
        "inventory_status": fast_result.get('inventory_status') if fast_result else None,
        "payment_result": slow_result.get('payment_result') if slow_result else None,
        "shipping_result": slow_result.get('shipping_result') if slow_result else None,
        "invoice_result": slow_result.get('invoice_result') if slow_result else None,
        "confirmation_timestamp": time.time()
    }

    # Calculate timing statistics
    branch_times = [data.get('processing_time_ms', 0) for data in all_data]
    fastest_time = min(branch_times) if branch_times else 0
    slowest_time = max(branch_times) if branch_times else 0

    result = {
        "status": "success" if inventory_available else "rejected",
        "short_circuited": False,  # CLASSIC mode can't short-circuit
        "order_confirmation": order_confirmation if inventory_available else None,
        "order_rejection": order_rejection,
        "total_branches": len(all_data),
        "pre_resolved_count": pre_resolved,
        "aggregation_time_ms": int(aggregation_time),
        "branch_timings": branch_timings,
        "branch_processing_times": {
            "fast_processor_ms": fast_result.get('processing_time_ms', 0) if fast_result else 0,
            "slow_chain_end_ms": slow_result.get('processing_time_ms', 0) if slow_result else 0
        },
        "timing_variance_ms": slowest_time - fastest_time,
        "mode": "FUTURE_BASED" if os.environ.get("UNUM_FUTURE_BASED") == "true" else "CLASSIC",
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "aggregator_complete",
        "order_id": order_id,
        "aggregation_time_ms": int(aggregation_time),
        "pre_resolved_count": pre_resolved,
        "timing_variance_ms": result["timing_variance_ms"],
        "inventory_available": inventory_available
    }))

    print(f"[Aggregator] COMPLETE in {aggregation_time:.2f}ms")
    print(f"[Aggregator] Pre-resolved: {pre_resolved}/2")
    print(f"[Aggregator] Timing variance: {result['timing_variance_ms']:.0f}ms")
    if not inventory_available:
        print(f"[Aggregator] WASTED WORK: Waited for slow chain but rejected anyway!")
    else:
        print(f"[Aggregator] FUTURE mode benefit: Cold start hidden behind slow chain")

    return result


if __name__ == '__main__':
    # Simulate aggregation
    test_results = [
        {
            'function': 'FastProcessor',
            'order_id': 'ORD-TEST-001',
            'processing_time_ms': 102,
            'inventory_status': {'available': True}
        },
        {
            'function': 'SlowChainEnd',
            'order_id': 'ORD-TEST-001',
            'processing_time_ms': 3050,
            'payment_result': {'payment_authorized': True},
            'shipping_result': {'carrier': 'DHL'},
            'invoice_result': {'invoice_id': 'INV-12345'}
        }
    ]

    result = lambda_handler(test_results, None)
    print(f"\nResult: {json.dumps(result, indent=2)}")
