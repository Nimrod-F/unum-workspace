"""
Merge Dashboard Data - Fan-in aggregation function
This function receives data from all 6 parallel fetch functions and combines them.

CRITICAL: Supports BOTH sync and async modes for benchmarking:
- CLASSIC mode: All inputs are ready as a regular list
- EAGER/FUTURE mode: Inputs are LazyInput or AsyncFutureInputList objects
"""
import time
import json
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Main handler - supports both sync and async modes.

    In CLASSIC mode: event is a list of 6 completed results
    In EAGER/FUTURE mode: event is a LazyInputList or AsyncFutureInputList
    """
    start_time = time.time()
    function_name = "MergeDashboardData"

    # Log function start
    logger.info(json.dumps({
        "event": "aggregator_start",
        "function": function_name,
        "input_type": str(type(event).__name__),
        "timestamp": time.time()
    }))

    # Check if inputs support async (Future-Based mode)
    is_async_capable = hasattr(event, 'get_all_async')

    if is_async_capable:
        # Use async mode for optimal performance
        logger.info(json.dumps({
            "event": "using_async_mode",
            "function": function_name
        }))
        result = asyncio.run(process_inputs_async(event, start_time))
    else:
        # Fall back to sync mode (works for both CLASSIC and EAGER LazyInput)
        logger.info(json.dumps({
            "event": "using_sync_mode",
            "function": function_name
        }))
        result = process_inputs_sync(event, start_time)

    # Log completion
    end_time = time.time()
    aggregation_time_ms = (end_time - start_time) * 1000

    logger.info(json.dumps({
        "event": "aggregation_complete",
        "function": function_name,
        "aggregation_time_ms": aggregation_time_ms,
        "total_sources": result.get("total_sources", 0),
        "timestamp": end_time
    }))

    return result


async def process_inputs_async(inputs, start_time):
    """
    Async processing for Future-Based mode.

    This allows non-blocking waiting for inputs and can process
    inputs as they become available.
    """
    function_name = "MergeDashboardData"

    # Track which inputs are initially ready
    initially_ready = 0
    pre_resolved = 0

    if hasattr(inputs, '__len__'):
        # Count initially ready inputs
        for i in range(len(inputs)):
            if hasattr(inputs, 'is_ready') and inputs.is_ready(i):
                initially_ready += 1

    logger.info(json.dumps({
        "event": "async_fan_in_start",
        "function": function_name,
        "initially_ready": initially_ready,
        "total_inputs": len(inputs) if hasattr(inputs, '__len__') else 0
    }))

    # Get all inputs asynchronously (waits for slowest)
    all_data = await inputs.get_all_async()

    # Count pre-resolved (inputs that were ready before we accessed them)
    for i in range(len(inputs)):
        if hasattr(inputs, 'is_ready') and inputs.is_ready(i):
            pre_resolved += 1

    logger.info(json.dumps({
        "event": "async_inputs_retrieved",
        "function": function_name,
        "pre_resolved": pre_resolved,
        "total_inputs": len(all_data)
    }))

    # Process the aggregated data
    return merge_data(all_data, start_time, pre_resolved)


def process_inputs_sync(inputs, start_time):
    """
    Sync processing for CLASSIC and EAGER LazyInput modes.

    In CLASSIC: All inputs are already available
    In EAGER LazyInput: Blocks synchronously when accessing unavailable inputs
    """
    function_name = "MergeDashboardData"

    # Convert inputs to list (handles both list and LazyInput)
    all_data = []
    pre_resolved = 0

    # Track access pattern for metrics
    for i, data in enumerate(inputs):
        access_start = time.time()

        # This may block if using LazyInput and data not ready
        all_data.append(data)

        access_time_ms = (time.time() - access_start) * 1000

        logger.info(json.dumps({
            "event": "input_accessed",
            "function": function_name,
            "input_index": i,
            "access_time_ms": access_time_ms,
            "blocked": access_time_ms > 10  # Consider blocked if took >10ms
        }))

        # In Future mode, check if it was pre-resolved
        if access_time_ms < 10:
            pre_resolved += 1

    logger.info(json.dumps({
        "event": "sync_inputs_retrieved",
        "function": function_name,
        "pre_resolved": pre_resolved,
        "total_inputs": len(all_data)
    }))

    return merge_data(all_data, start_time, pre_resolved)


def merge_data(all_data, start_time, pre_resolved):
    """
    Core merging logic - combines data from all sources.

    This function does the actual aggregation work, simulating
    some computational overhead.
    """
    function_name = "MergeDashboardData"
    merge_start = time.time()

    # Initialize merged result
    dashboard = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "total_sources": len(all_data),
        "pre_resolved_count": pre_resolved,
        "sources_status": {},
        "summary": {},
        "insights": [],
        "metadata": {
            "aggregation_mode": "async" if pre_resolved > 0 else "classic",
            "total_latency_ms": 0
        }
    }

    # Simulate some computational work during merge
    # This represents realistic processing like calculations, transformations, etc.
    time.sleep(0.05)  # 50ms of processing overhead

    # Process each source's data
    total_revenue = 0
    total_items = 0

    for data in all_data:
        if not isinstance(data, dict):
            continue

        source = data.get("source", "unknown")
        status = data.get("status", "unknown")
        latency_ms = data.get("latency_ms", 0)

        dashboard["sources_status"][source] = {
            "status": status,
            "latency_ms": latency_ms
        }

        # Extract key metrics from each source
        source_data = data.get("data", {})

        if source == "sales":
            total_revenue = source_data.get("total_revenue", 0)
            dashboard["summary"]["total_revenue"] = total_revenue
            dashboard["summary"]["transactions"] = source_data.get("transactions", 0)
            dashboard["insights"].append(f"Revenue: ${total_revenue:,.2f}")

        elif source == "inventory":
            total_items = source_data.get("total_items", 0)
            low_stock = source_data.get("low_stock_items", 0)
            dashboard["summary"]["inventory_items"] = total_items
            dashboard["summary"]["low_stock_alerts"] = low_stock
            if low_stock > 100:
                dashboard["insights"].append(f"⚠️ {low_stock} items low on stock")

        elif source == "marketing":
            conversions = source_data.get("conversions", 0)
            conversion_rate = source_data.get("conversion_rate", 0)
            dashboard["summary"]["marketing_conversions"] = conversions
            dashboard["summary"]["conversion_rate"] = conversion_rate
            dashboard["insights"].append(f"Conversion rate: {conversion_rate}%")

        elif source == "external_market":
            market_data = source_data.get("indices", {})
            sentiment = source_data.get("market_sentiment", "neutral")
            dashboard["summary"]["market_sentiment"] = sentiment
            dashboard["insights"].append(f"Market sentiment: {sentiment}")

        elif source == "weather":
            alerts = source_data.get("severe_weather_alerts", [])
            if alerts:
                dashboard["summary"]["weather_alerts"] = len(alerts)
                dashboard["insights"].append(f"⚠️ {len(alerts)} weather alerts")

        elif source == "competitor_pricing":
            position = source_data.get("market_position", {}).get("position", "unknown")
            dashboard["summary"]["market_position"] = position

    # Calculate efficiency metrics
    if total_revenue > 0 and total_items > 0:
        revenue_per_item = total_revenue / total_items
        dashboard["summary"]["revenue_per_item"] = round(revenue_per_item, 2)

    # Add timing metadata
    merge_time_ms = (time.time() - merge_start) * 1000
    total_time_ms = (time.time() - start_time) * 1000

    dashboard["metadata"]["merge_time_ms"] = merge_time_ms
    dashboard["metadata"]["total_latency_ms"] = total_time_ms
    dashboard["metadata"]["processing_overhead_ms"] = 50  # Our simulated overhead

    logger.info(json.dumps({
        "event": "merge_complete",
        "function": function_name,
        "merge_time_ms": merge_time_ms,
        "total_time_ms": total_time_ms,
        "sources_processed": len(all_data),
        "pre_resolved": pre_resolved
    }))

    return dashboard
