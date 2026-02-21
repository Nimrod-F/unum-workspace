"""
Fetch External Market Data - Simulates fetching data from slow external provider
Fixed Latency: 7.0s (Staircase Benchmark: Step 4/6)
"""
import time
import random
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Fetch external market data with FIXED latency for benchmarking.
    """
    start_time = time.time()
    function_name = "FetchExternalMarketData"

    # --- BENCHMARK CONFIGURATION ---
    # This is Step 4 in the Staircase.
    # It finishes at T+7s.
    # In Future Mode, the Aggregator (started at T+1s) will have processed 
    # Sales (1s) and Inventory (3s), and is currently processing Marketing (5s).
    # It will pick this up immediately after Marketing is done.
    delay = 7.0
    time.sleep(delay)
    # -------------------------------

    # Generate realistic mock data
    indices = {}
    for index in ["S&P500", "NASDAQ", "DOW", "FTSE", "DAX", "NIKKEI"]:
        indices[index] = {
            "value": round(random.uniform(10000, 40000), 2),
            "change": round(random.uniform(-2.5, 3.0), 2),
            "change_pct": round(random.uniform(-1.5, 2.0), 2)
        }

    result = {
        "source": "external_market",
        "status": "success",
        "data": {
            "indices": indices,
            "commodities": {
                "gold": {"price": round(random.uniform(1800, 2100), 2), "unit": "USD/oz"},
                "silver": {"price": round(random.uniform(20, 30), 2), "unit": "USD/oz"},
                "oil_wti": {"price": round(random.uniform(70, 90), 2), "unit": "USD/barrel"},
                "oil_brent": {"price": round(random.uniform(75, 95), 2), "unit": "USD/barrel"}
            },
            "forex": {
                "EUR_USD": round(random.uniform(1.05, 1.15), 4),
                "GBP_USD": round(random.uniform(1.20, 1.35), 4),
                "USD_JPY": round(random.uniform(130, 150), 2),
                "USD_CHF": round(random.uniform(0.85, 0.95), 4)
            },
            "market_sentiment": random.choice(["bullish", "neutral", "bearish"]),
            "volatility_index": round(random.uniform(10, 30), 2)
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
        "status": "success",
        "note": "BENCHMARK_STEP_4"
    }))

    return result