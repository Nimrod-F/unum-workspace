"""
Fetch Weather Data - Simulates fetching weather data for regional operations planning
Latency: 300-1500ms (external weather API with moderate latency)
"""
import time
import random
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Fetch weather data with simulated latency (300-1500ms).

    Returns weather conditions, forecasts, and alerts for operational regions.
    """
    start_time = time.time()
    function_name = "FetchWeatherData"

    # Simulate realistic latency (300-1500ms for weather API)
    delay = random.uniform(0.3, 1.5)
    time.sleep(delay)

    # Generate realistic mock data
    regions = ["north", "south", "east", "west", "central"]
    regional_weather = {}

    for region in regions:
        regional_weather[region] = {
            "temperature": round(random.uniform(-10, 35), 1),
            "humidity": random.randint(30, 90),
            "wind_speed": round(random.uniform(5, 40), 1),
            "conditions": random.choice(["sunny", "cloudy", "rainy", "stormy", "snowy"]),
            "precipitation_chance": random.randint(0, 100)
        }

    result = {
        "source": "weather",
        "status": "success",
        "data": {
            "regional_weather": regional_weather,
            "severe_weather_alerts": [
                {
                    "region": random.choice(regions),
                    "alert_type": random.choice(["storm", "flood", "heat", "cold"]),
                    "severity": random.choice(["low", "moderate", "high"])
                }
                for _ in range(random.randint(0, 3))
            ],
            "forecast_7day": [
                {
                    "day": f"Day {i+1}",
                    "high": round(random.uniform(15, 30), 1),
                    "low": round(random.uniform(5, 15), 1),
                    "conditions": random.choice(["sunny", "cloudy", "rainy"])
                }
                for i in range(7)
            ],
            "impact_on_operations": {
                "shipping_delays": random.choice([True, False]),
                "outdoor_work_safe": random.choice([True, False]),
                "hvac_load": random.choice(["low", "normal", "high"])
            }
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
