"""
Windowing - Branch C Step 1: Heavy Path (400ms)

Simulates preparing time-windowed signal data for FFT analysis.
Applies a Hanning window function to the raw vibration readings.

Branch C total: Windowing (400ms) + ComputeFFT (600ms) + FailureModel (1000ms) = 2000ms
"""
import json
import time
import logging
import random
import math

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Heavy operation step 1: 400ms (signal windowing)
PROCESSING_DELAY_MS = 400


def lambda_handler(event, context):
    """
    Apply time-windowing to vibration signal for frequency analysis.

    This is step 1 of the heavy processing path (Branch C).
    """
    start_time = time.time()

    # Extract sensor data from SensorIngest
    machine_id = event.get("machine_id", "unknown")
    sensor_id = event.get("sensor_id", "unknown")
    vibration_readings = event.get("vibration_readings", [])

    logger.info(json.dumps({
        "event": "windowing_start",
        "machine_id": machine_id,
        "branch": "heavy_path",
        "chain_step": 1,
        "input_samples": len(vibration_readings)
    }))

    print(f"[Windowing] Starting - Chain step 1/3 ({PROCESSING_DELAY_MS}ms)")
    print(f"[Windowing] Machine: {machine_id}, Input samples: {len(vibration_readings)}")

    # Simulate windowing computation
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    # Apply simulated Hanning window to vibration data
    window_size = min(len(vibration_readings), 32)
    raw_segment = vibration_readings[:window_size]

    # Simulated Hanning window coefficients
    windowed_signal = []
    for i, val in enumerate(raw_segment):
        # Hanning window: w(n) = 0.5 * (1 - cos(2*pi*n/(N-1)))
        if window_size > 1:
            coefficient = 0.5 * (1 - math.cos(2 * math.pi * i / (window_size - 1)))
        else:
            coefficient = 1.0
        windowed_signal.append(round(val * coefficient, 4))

    total_time = (time.time() - start_time) * 1000

    result = {
        "function": "Windowing",
        "branch": "heavy_path",
        "chain_step": 1,
        "machine_id": machine_id,
        "sensor_id": sensor_id,
        "status": "complete",
        "windowed_signal": windowed_signal,
        "window_size": window_size,
        "window_type": "hanning",
        "sample_rate_hz": 1000,
        "processing_time_ms": int(total_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "windowing_complete",
        "machine_id": machine_id,
        "chain_step": 1,
        "processing_time_ms": int(total_time),
        "window_size": window_size
    }))

    print(f"[Windowing] COMPLETE in {total_time:.2f}ms")
    print(f"[Windowing] Window: {window_size} samples, type=hanning")
    print(f"[Windowing] Next: ComputeFFT (frequency analysis)")

    return result


if __name__ == '__main__':
    test_event = {
        "machine_id": "MACH-TEST-001",
        "sensor_id": "SENS-TEST-01",
        "vibration_readings": [1.2, 3.4, 5.6, 2.1, 8.9, 4.3, 6.7, 1.5]
    }

    result = lambda_handler(test_event, None)
    print(f"\nResult: {json.dumps(result, indent=2)}")
