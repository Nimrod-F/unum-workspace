"""
Trigger Dashboard - Entry point that initiates the fan-out to all data sources

This function serves as the entry point to the workflow. It triggers the
parallel execution of all 6 data fetch functions.
"""
import json
import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Entry point for the Multi-Source Dashboard workflow.

    This function initiates the fan-out pattern by returning a list
    of payloads, one for each data source. The UNUM runtime will
    automatically invoke each fetch function in parallel.

    Returns: List of 6 payloads for parallel fetch functions
    """
    start_time = time.time()
    function_name = "TriggerDashboard"

    # Extract request parameters (if any)
    request_id = event.get("request_id", f"dash-{int(time.time() * 1000)}")
    dashboard_type = event.get("dashboard_type", "executive")
    time_range = event.get("time_range", "24h")

    logger.info(json.dumps({
        "event": "dashboard_request_start",
        "function": function_name,
        "request_id": request_id,
        "dashboard_type": dashboard_type,
        "time_range": time_range,
        "timestamp": start_time
    }))

    # Create payloads for each data source
    # The UNUM runtime will invoke the corresponding function for each payload
    # in the ParallelFetch state from the step functions definition

    payloads = [
        # Fast internal source (100-400ms)
        {
            "source": "sales",
            "request_id": request_id,
            "time_range": time_range
        },
        # Internal system (150-500ms)
        {
            "source": "inventory",
            "request_id": request_id,
            "time_range": time_range
        },
        # Marketing API (200-800ms)
        {
            "source": "marketing",
            "request_id": request_id,
            "time_range": time_range
        },
        # SLOWEST - External market data (500-3000ms)
        {
            "source": "external_market",
            "request_id": request_id,
            "time_range": time_range
        },
        # Weather API (300-1500ms)
        {
            "source": "weather",
            "request_id": request_id,
            "time_range": time_range
        },
        # Competitor scraping (400-2500ms)
        {
            "source": "competitor_pricing",
            "request_id": request_id,
            "time_range": time_range
        }
    ]

    end_time = time.time()
    trigger_time_ms = (end_time - start_time) * 1000

    logger.info(json.dumps({
        "event": "fan_out_triggered",
        "function": function_name,
        "request_id": request_id,
        "parallel_branches": len(payloads),
        "trigger_time_ms": trigger_time_ms,
        "timestamp": end_time
    }))

    # Return the payloads - UNUM will handle the parallel invocation
    return payloads


if __name__ == '__main__':
    # Test locally
    test_event = {
        "request_id": "test-123",
        "dashboard_type": "executive",
        "time_range": "24h"
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
    print(f"\nTotal branches: {len(result)}")
