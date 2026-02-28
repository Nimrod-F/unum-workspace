"""
SensorIngest - Entry point for Smart Factory IoT Alert Workflow

Receives vibration telemetry and fans out to three parallel branches:
1. SafetyCheck - Quick safety threshold validation (Branch A, ~100ms)
2. MachineState - Machine context lookup chain (Branch B, ~500ms total)
3. Windowing - Heavy signal processing chain (Branch C, ~2000ms total)

Generates a 4KB dummy payload to simulate "Data Gravity" - the cost of
transporting heavy sensor telemetry through the workflow.
"""
import json
import time
import logging
import random
import string

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Data Gravity: 4KB payload simulates realistic sensor telemetry transport cost
PAYLOAD_SIZE_BYTES = 4096


def lambda_handler(event, context):
    """
    Ingest sensor telemetry and initiate 3-way fan-out.

    Input:
    {
        "machine_id": "MACH-1234",
        "sensor_id": "SENS-001",
        "force_critical": false
    }

    Output: Single dict sent to all 3 branches via Scalar fan-out
    """
    start_time = time.time()

    # Extract sensor parameters
    machine_id = event.get("machine_id", f"MACH-{random.randint(1000, 9999)}")
    sensor_id = event.get("sensor_id", f"SENS-{random.randint(100, 999)}")
    force_critical = event.get("force_critical", False)

    logger.info(json.dumps({
        "event": "sensor_ingest_start",
        "machine_id": machine_id,
        "sensor_id": sensor_id,
        "force_critical": force_critical
    }))

    print(f"[SensorIngest] Machine: {machine_id}, Sensor: {sensor_id}")
    print(f"[SensorIngest] force_critical={force_critical}")

    # Generate 4KB random payload (simulating raw sensor telemetry)
    raw_telemetry = ''.join(
        random.choices(string.ascii_letters + string.digits, k=PAYLOAD_SIZE_BYTES)
    )

    # Generate vibration readings (50 data points)
    if force_critical:
        # Generate dangerous vibration (14.0 to 20.0)
        vibration_readings = [round(random.uniform(14.1, 20.0), 3) for _ in range(50)]
        print(f"[SensorIngest] Generating CRITICAL vibration levels (Force Mode)")
    else:
        # Generate safe vibration (0.1 to 12.0) - giving a 2.0 safety margin below 14.0
        vibration_readings = [round(random.uniform(0.1, 12.0), 3) for _ in range(50)]
        print(f"[SensorIngest] Generating SAFE vibration levels (Normal Mode)")
        
    # Assemble sensor data payload
    sensor_data = {
        "machine_id": machine_id,
        "sensor_id": sensor_id,
        "force_critical": force_critical,
        "vibration_readings": vibration_readings,
        "temperature_c": round(random.uniform(60.0, 120.0), 1),
        "rpm": random.randint(1000, 5000),
        "raw_telemetry": raw_telemetry,
        "ingest_timestamp": time.time()
    }

    execution_time = (time.time() - start_time) * 1000

    logger.info(json.dumps({
        "event": "sensor_ingested",
        "machine_id": machine_id,
        "sensor_id": sensor_id,
        "force_critical": force_critical,
        "payload_size_bytes": len(raw_telemetry),
        "vibration_count": len(vibration_readings),
        "execution_time_ms": int(execution_time)
    }))

    print(f"[SensorIngest] Payload: {len(raw_telemetry)} bytes")
    print(f"[SensorIngest] Vibration readings: {len(vibration_readings)} samples")
    print(f"[SensorIngest] Fan-out to 3 branches in {execution_time:.2f}ms")
    print(f"[SensorIngest] Branch A: SafetyCheck (fast path)")
    print(f"[SensorIngest] Branch B: MachineState -> ShiftCheck (context path)")
    print(f"[SensorIngest] Branch C: Windowing -> ComputeFFT -> FailureModel (heavy path)")

    # Return single dict - unum Scalar fan-out sends this to all 3 branches
    return sensor_data


if __name__ == '__main__':
    # Local test
    test_event = {
        "machine_id": "MACH-TEST-001",
        "sensor_id": "SENS-TEST-01",
        "force_critical": False
    }

    result = lambda_handler(test_event, None)
    print(f"\nResult keys: {list(result.keys())}")
    print(f"Payload size: {len(result['raw_telemetry'])} bytes")
    print(f"Vibration samples: {len(result['vibration_readings'])}")
