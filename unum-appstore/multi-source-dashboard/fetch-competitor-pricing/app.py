"""
Fetch Competitor Pricing - Simulates the SLOWEST task (The Straggler)
Fixed Latency: 12.0s (Staircase Benchmark: Step 6/6)
"""
import time
import random
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Fetch competitor pricing with FIXED latency for benchmarking.
    """
    start_time = time.time()
    function_name = "FetchCompetitorPricing"

    # --- BENCHMARK CONFIGURATION ---
    # This is the "Straggler". It takes 12 seconds.
    # In Future Mode, the aggregator will process the other 5 inputs 
    # (which finish at 1s, 3s, 5s, 7s, 9s) while waiting for this one.
    delay = 12.0
    time.sleep(delay)
    # -------------------------------

    # Generate realistic mock data
    competitors = ["Competitor A", "Competitor B", "Competitor C", "Competitor D"]
    product_categories = ["Electronics", "Home & Garden", "Clothing", "Sports", "Books"]

    competitor_data = {}
    for competitor in competitors:
        category_prices = {}
        for category in product_categories:
            category_prices[category] = {
                "avg_price": round(random.uniform(20, 200), 2),
                "min_price": round(random.uniform(10, 50), 2),
                "max_price": round(random.uniform(150, 300), 2),
                "product_count": random.randint(50, 500)
            }
        competitor_data[competitor] = category_prices

    result = {
        "source": "competitor_pricing",
        "status": "success",
        "data": {
            "competitor_prices": competitor_data,
            "market_position": {
                "our_avg_price": round(random.uniform(50, 150), 2),
                "market_avg_price": round(random.uniform(45, 155), 2),
                "price_competitiveness": round(random.uniform(0.8, 1.2), 2),
                "position": random.choice(["premium", "mid-market", "value"])
            },
            "price_trends": {
                "trend_7d": round(random.uniform(-5, 5), 2),
                "trend_30d": round(random.uniform(-8, 8), 2),
                "volatility": round(random.uniform(0.5, 3.0), 2)
            },
            "opportunities": [
                {
                    "product": f"Product {i}",
                    "our_price": round(random.uniform(50, 100), 2),
                    "competitor_avg": round(random.uniform(60, 110), 2),
                    "recommended_action": random.choice(["increase", "decrease", "maintain"])
                }
                for i in range(random.randint(3, 7))
            ]
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
        "status": "success"
    }))

    return result