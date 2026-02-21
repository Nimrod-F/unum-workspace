"""
SafetyCheck - Branch A: Fast Path (100ms)

Quick safety threshold validation. This is the FASTEST branch and
triggers the ActionDispatcher first in FUTURE_BASED mode.

If force_critical is true in the payload, returns CRITICAL_STOP status,
enabling short-circuit optimization in FUTURE_BASED mode where the
ActionDispatcher can immediately dispatch an emergency stop WITHOUT
waiting for the slower branches.
"""
import json
import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Fast operation: 100ms
PROCESSING_DELAY_MS = 100


def lambda_handler(event, context):
    """
    Validate safety thresholds from sensor data.

    Returns CRITICAL_STOP if force_critical flag is set, enabling
    ActionDispatcher short-circuit in FUTURE_BASED mode.
    """
    start_time = time.time()

    # Extract sensor data
    machine_id = event.get("machine_id", "unknown")
    sensor_id = event.get("sensor_id", "unknown")
    force_critical = event.get("force_critical", False)
    vibration_readings = event.get("vibration_readings", [])

    logger.info(json.dumps({
        "event": "safety_check_start",
        "machine_id": machine_id,
        "branch": "fast_path",
        "force_critical": force_critical
    }))

    print(f"[SafetyCheck] Starting - FASTEST branch ({PROCESSING_DELAY_MS}ms)")
    print(f"[SafetyCheck] Machine: {machine_id}, force_critical={force_critical}")

    # Simulate safety threshold check
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    # Determine safety status
    if force_critical:
        status = "CRITICAL_STOP"
        safety_details = {
            "threshold_exceeded": True,
            "reason": "force_critical_flag",
            "action": "EMERGENCY_SHUTDOWN",
            "alert_level": "RED"
        }
        print(f"[SafetyCheck] CRITICAL_STOP - force_critical flag detected!")
    else:
        max_vibration = max(vibration_readings) if vibration_readings else 0
        threshold = 14.0
        exceeded = max_vibration > threshold

        if exceeded:
            status = "CRITICAL_STOP"
            safety_details = {
                "threshold_exceeded": True,
                "reason": "vibration_threshold_exceeded",
                "max_vibration": max_vibration,
                "threshold": threshold,
                "action": "EMERGENCY_SHUTDOWN",
                "alert_level": "RED"
            }
            print(f"[SafetyCheck] CRITICAL_STOP - vibration {max_vibration:.3f} > {threshold}")
        else:
            status = "safety_ok"
            safety_details = {
                "threshold_exceeded": False,
                "max_vibration": max_vibration,
                "threshold": threshold,
                "action": "CONTINUE",
                "alert_level": "GREEN"
            }
            print(f"[SafetyCheck] OK - vibration {max_vibration:.3f} <= {threshold}")

    total_time = (time.time() - start_time) * 1000

    result = {
        "function": "SafetyCheck",
        "branch": "fast_path",
        "machine_id": machine_id,
        "sensor_id": sensor_id,
        "status": status,
        "safety_details": safety_details,
        "processing_time_ms": int(total_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "safety_check_complete",
        "machine_id": machine_id,
        "status": status,
        "processing_time_ms": int(total_time)
    }))

    print(f"[SafetyCheck] COMPLETE in {total_time:.2f}ms - Status: {status}")
    print(f"[SafetyCheck] This should trigger ActionDispatcher FIRST in FUTURE mode")

    return result


if __name__ == '__main__':
    # Test normal path
    test_event = {
        "machine_id": "MACH-TEST-001",
        "sensor_id": "SENS-TEST-01",
        "force_critical": False,
        "vibration_readings": [1.2, 3.4, 5.6, 2.1, 8.9]
    }
    result = lambda_handler(test_event, None)
    print(f"\nNormal: {json.dumps(result, indent=2)}")

    # Test critical path
    test_event["force_critical"] = True
    result = lambda_handler(test_event, None)
    print(f"\nCritical: {json.dumps(result, indent=2)}")
