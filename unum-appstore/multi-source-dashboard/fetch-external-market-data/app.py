"""
Fetch External Market Data - Simulates fetching data from slow external market data provider
Latency: 500-3000ms (SLOWEST - external API with high latency)
This is the critical path function that demonstrates the benefit of futures
"""
import time
import random
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Fetch external market data with simulated latency (500-3000ms).

    This is intentionally the SLOWEST function to demonstrate the benefit
    of Future-Based execution where other faster branches complete first.

    Returns market indices, commodity prices, and forex rates.
    """
    start_time = time.time()
    function_name = "FetchExternalMarketData"

    # Simulate realistic latency (500-3000ms for slow external market data API)
    # This is the critical slow path that benefits most from future-based execution
    delay = random.uniform(0.5, 3.0)
    time.sleep(delay)

    # Generate realistic mock data with some computational overhead
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

    # Log for metrics collection - this is the slowest branch
    logger.info(json.dumps({
        "event": "function_complete",
        "function": function_name,
        "latency_ms": result["latency_ms"],
        "simulated_delay_ms": delay * 1000,
        "status": "success",
        "note": "SLOWEST_BRANCH - demonstrates future-based benefit"
    }))

    return result
