"""
ShiftCheck - Branch B Step 2: Context Path Terminal (300ms)

Simulates checking the current shift context and operational parameters.
Terminal function of Branch B - fans into ActionDispatcher.

Branch B total: MachineState (200ms) + ShiftCheck (300ms) = 500ms
"""
import json
import time
import logging
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Moderate operation: 300ms (shift context lookup)
PROCESSING_DELAY_MS = 300


def lambda_handler(event, context):
    """
    Check current shift context and operational parameters.

    This is step 2 (terminal) of the context path (Branch B).
    Receives MachineState output as input.
    """
    start_time = time.time()

    # Extract data from MachineState output
    machine_id = event.get("machine_id", "unknown")
    sensor_id = event.get("sensor_id", "unknown")
    machine_state = event.get("machine_state", {})

    logger.info(json.dumps({
        "event": "shift_check_start",
        "machine_id": machine_id,
        "branch": "context_path",
        "chain_step": 2
    }))

    print(f"[ShiftCheck] Starting - Chain step 2/2 ({PROCESSING_DELAY_MS}ms)")
    print(f"[ShiftCheck] Machine: {machine_id}")

    # Simulate shift context lookup
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    # Shift context results
    shift_context = {
        "current_shift": random.choice(["day", "evening", "night"]),
        "operator_id": f"OP-{random.randint(1000, 9999)}",
        "maintenance_window": random.choice([True, False]),
        "production_line": f"PL-{random.randint(1, 12):02d}",
        "quality_target": round(random.uniform(98.0, 99.9), 1),
        "shift_start_time": "06:00",
        "shift_hours_remaining": round(random.uniform(1.0, 8.0), 1)
    }

    total_time = (time.time() - start_time) * 1000

    result = {
        "function": "ShiftCheck",
        "branch": "context_path",
        "chain_step": 2,
        "machine_id": machine_id,
        "sensor_id": sensor_id,
        "status": "complete",
        "machine_state": machine_state,
        "shift_context": shift_context,
        "processing_time_ms": int(total_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "shift_check_complete",
        "machine_id": machine_id,
        "chain_step": 2,
        "processing_time_ms": int(total_time),
        "shift": shift_context["current_shift"]
    }))

    print(f"[ShiftCheck] COMPLETE in {total_time:.2f}ms")
    print(f"[ShiftCheck] Shift: {shift_context['current_shift']}, Operator: {shift_context['operator_id']}")
    print(f"[ShiftCheck] Branch B total: ~500ms (200+300)")

    return result


if __name__ == '__main__':
    test_event = {
        "machine_id": "MACH-TEST-001",
        "sensor_id": "SENS-TEST-01",
        "machine_state": {
            "temperature_c": 85.5,
            "rpm": 3200,
            "uptime_hours": 5400,
            "operational_mode": "production"
        }
    }

    result = lambda_handler(test_event, None)
    print(f"\nResult: {json.dumps(result, indent=2)}")
