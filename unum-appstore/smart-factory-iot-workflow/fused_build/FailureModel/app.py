"""
FailureModel - Branch C Step 3: Heavy Path Terminal (1000ms)

Simulates ML failure prediction model inference based on FFT analysis.
Returns failure probability, risk level, and remaining useful life (RUL).

This is the SLOWEST function and the critical path bottleneck.
Terminal function of Branch C - fans into ActionDispatcher.

Branch C total: Windowing (400ms) + ComputeFFT (600ms) + FailureModel (1000ms) = 2000ms
"""
import json
import time
import logging
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Heaviest operation: 1000ms (ML model inference)
PROCESSING_DELAY_MS = 1000


def lambda_handler(event, context):
    """
    Run ML failure prediction model on frequency spectrum data.

    This is step 3 (terminal) of the heavy processing path (Branch C).
    Receives ComputeFFT output as input.
    """
    start_time = time.time()

    # Extract data from ComputeFFT output
    machine_id = event.get("machine_id", "unknown")
    sensor_id = event.get("sensor_id", "unknown")
    frequency_spectrum = event.get("frequency_spectrum", {})

    logger.info(json.dumps({
        "event": "failure_model_start",
        "machine_id": machine_id,
        "branch": "heavy_path",
        "chain_step": 3
    }))

    print(f"[FailureModel] Starting - Chain step 3/3 ({PROCESSING_DELAY_MS}ms)")
    print(f"[FailureModel] Machine: {machine_id}")
    print(f"[FailureModel] This is the SLOWEST function (critical path bottleneck)")

    # Simulate heavy ML model inference
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    # ML failure prediction results
    failure_probability = round(random.uniform(0.01, 0.35), 4)

    if failure_probability > 0.25:
        risk_level = "HIGH"
    elif failure_probability > 0.10:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    prediction = {
        "failure_probability": failure_probability,
        "risk_level": risk_level,
        "predicted_rul_hours": random.randint(200, 5000),
        "model_version": "v2.3.1",
        "confidence": round(random.uniform(0.85, 0.99), 3),
        "contributing_factors": {
            "vibration_anomaly": round(random.uniform(0.0, 1.0), 3),
            "thermal_stress": round(random.uniform(0.0, 1.0), 3),
            "bearing_wear": round(random.uniform(0.0, 1.0), 3)
        },
        "recommended_action": (
            "schedule_immediate_maintenance" if risk_level == "HIGH"
            else "increase_monitoring" if risk_level == "MEDIUM"
            else "nominal_operation"
        )
    }

    total_time = (time.time() - start_time) * 1000

    result = {
        "function": "FailureModel",
        "branch": "heavy_path",
        "chain_step": 3,
        "machine_id": machine_id,
        "sensor_id": sensor_id,
        "status": "complete",
        "frequency_spectrum": frequency_spectrum,
        "prediction": prediction,
        "processing_time_ms": int(total_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "failure_model_complete",
        "machine_id": machine_id,
        "chain_step": 3,
        "processing_time_ms": int(total_time),
        "failure_probability": failure_probability,
        "risk_level": risk_level
    }))

    print(f"[FailureModel] COMPLETE in {total_time:.2f}ms")
    print(f"[FailureModel] Risk: {risk_level} (p={failure_probability:.4f})")
    print(f"[FailureModel] RUL: {prediction['predicted_rul_hours']}h")
    print(f"[FailureModel] Branch C total: ~2000ms (400+600+1000)")
    print(f"[FailureModel] In CLASSIC mode: This triggers ActionDispatcher")
    print(f"[FailureModel] In FUTURE mode: ActionDispatcher already started ~1900ms ago")

    return result


if __name__ == '__main__':
    test_event = {
        "machine_id": "MACH-TEST-001",
        "sensor_id": "SENS-TEST-01",
        "frequency_spectrum": {
            "dominant_frequency_hz": 245.3,
            "spectral_energy": 2.45,
            "noise_floor_db": -32.1
        }
    }

    result = lambda_handler(test_event, None)
    print(f"\nResult: {json.dumps(result, indent=2)}")
