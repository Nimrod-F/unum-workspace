"""
MachineState - Branch B Step 1: Context Path (200ms)

Simulates looking up the current machine state from a database.
Passes machine state data to ShiftCheck for operational context.

Branch B total: MachineState (200ms) + ShiftCheck (300ms) = 500ms
"""
import json
import time
import logging
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Moderate operation: 200ms (database lookup)
PROCESSING_DELAY_MS = 200


def lambda_handler(event, context):
    """
    Look up current machine operational state.

    This is step 1 of the context path (Branch B).
    """
    start_time = time.time()

    # Extract sensor data from SensorIngest
    machine_id = event.get("machine_id", "unknown")
    sensor_id = event.get("sensor_id", "unknown")
    temperature_c = event.get("temperature_c", 85.0)
    rpm = event.get("rpm", 3000)

    logger.info(json.dumps({
        "event": "machine_state_start",
        "machine_id": machine_id,
        "branch": "context_path",
        "chain_step": 1
    }))

    print(f"[MachineState] Starting - Chain step 1/2 ({PROCESSING_DELAY_MS}ms)")
    print(f"[MachineState] Machine: {machine_id}")

    # Simulate machine state database lookup
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    # Machine state results
    machine_state = {
        "temperature_c": temperature_c,
        "rpm": rpm,
        "uptime_hours": random.randint(100, 10000),
        "last_maintenance": "2026-01-15",
        "firmware_version": "3.2.1",
        "operational_mode": random.choice(["production", "calibration", "standby"]),
        "error_count_24h": random.randint(0, 5)
    }

    total_time = (time.time() - start_time) * 1000

    result = {
        "function": "MachineState",
        "branch": "context_path",
        "chain_step": 1,
        "machine_id": machine_id,
        "sensor_id": sensor_id,
        "status": "complete",
        "machine_state": machine_state,
        "processing_time_ms": int(total_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "machine_state_complete",
        "machine_id": machine_id,
        "chain_step": 1,
        "processing_time_ms": int(total_time),
        "operational_mode": machine_state["operational_mode"]
    }))

    print(f"[MachineState] COMPLETE in {total_time:.2f}ms")
    print(f"[MachineState] Mode: {machine_state['operational_mode']}, Uptime: {machine_state['uptime_hours']}h")
    print(f"[MachineState] Next: ShiftCheck (operational context)")

    return result


if __name__ == '__main__':
    test_event = {
        "machine_id": "MACH-TEST-001",
        "sensor_id": "SENS-TEST-01",
        "temperature_c": 85.5,
        "rpm": 3200
    }

    result = lambda_handler(test_event, None)
    print(f"\nResult: {json.dumps(result, indent=2)}")
