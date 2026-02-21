"""
ActionDispatcher - Fan-in Aggregator (200ms processing delay)

Aggregates results from 3 asymmetric parallel branches:
  Index 0: SafetyCheck   (Branch A, ~100ms)  - Fast Path
  Index 1: ShiftCheck    (Branch B, ~500ms)  - Context Path
  Index 2: FailureModel  (Branch C, ~2000ms) - Heavy Path

SHORT-CIRCUIT OPTIMIZATION (FUTURE_BASED mode only):
If SafetyCheck (index 0) returns CRITICAL_STOP, the ActionDispatcher
immediately dispatches an emergency stop WITHOUT waiting for the slower
branches. This saves ~1900ms compared to CLASSIC mode which must wait
for FailureModel to complete before it can act.

CLASSIC mode: Triggered by FailureModel (slowest, ~2000ms after start)
FUTURE mode: Triggered by SafetyCheck (fastest, ~100ms after start)
"""
import json
import time
import logging
import asyncio
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Decision processing delay
PROCESSING_DELAY_MS = 200


def lambda_handler(event, context):
    """
    Aggregate results from SafetyCheck, ShiftCheck, and FailureModel.

    In CLASSIC mode: Triggered by LAST function to complete (FailureModel)
    In FUTURE mode: Triggered by FIRST function to complete (SafetyCheck)

    Input: Array-like object with results from all 3 branches
    Output: Aggregated action dispatch decision
    """
    start_time = time.time()

    mode = "FUTURE_BASED" if os.environ.get("UNUM_FUTURE_BASED") == "true" else "CLASSIC"

    logger.info(json.dumps({
        "event": "action_dispatcher_start",
        "mode": mode
    }))

    print(f"[ActionDispatcher] Starting - Fan-in aggregator")
    print(f"[ActionDispatcher] Mode: {mode}")

    # Check if inputs support async (Future-Based mode)
    is_async = hasattr(event, 'get_async') or hasattr(event, 'get_all_async')

    if is_async:
        logger.info("[ActionDispatcher] Using FUTURE_BASED mode with async inputs")
        print("[ActionDispatcher] Using FUTURE_BASED mode - async/await")
        result = asyncio.run(process_inputs_async(event, start_time))
    else:
        logger.info("[ActionDispatcher] Using CLASSIC/EAGER mode with sync inputs")
        print("[ActionDispatcher] Using CLASSIC mode - synchronous")
        result = process_inputs_sync(event, start_time)

    return result


async def process_inputs_async(inputs, start_time):
    """
    Async processing for Future-Based mode.

    Fetches SafetyCheck (index 0) first. If CRITICAL_STOP, returns
    immediately without awaiting ShiftCheck and FailureModel.
    """
    print("[ActionDispatcher] Async mode: Fetching SafetyCheck first (index 0)")

    all_data = []
    pre_resolved = 0
    branch_timings = []

    # Fetch SafetyCheck FIRST (index 0) - should be pre-resolved (fastest branch)
    fetch_start = time.time()
    safety_data = await inputs.get_async(0)
    fetch_duration_ms = (time.time() - fetch_start) * 1000

    if fetch_duration_ms < 10:
        pre_resolved += 1
        print(f"[ActionDispatcher] Input 0 (SafetyCheck): PRE-RESOLVED ({fetch_duration_ms:.2f}ms)")
    else:
        print(f"[ActionDispatcher] Input 0 (SafetyCheck): Awaited {fetch_duration_ms:.2f}ms")

    # Parse if string
    if isinstance(safety_data, str):
        safety_data = json.loads(safety_data)

    branch_timings.append({
        "index": 0,
        "function": safety_data.get("function", "unknown"),
        "fetch_duration_ms": fetch_duration_ms,
        "pre_resolved": fetch_duration_ms < 10
    })
    all_data.append(safety_data)

    # SHORT-CIRCUIT: If CRITICAL_STOP, return emergency response immediately!
    if safety_data.get("status") == "CRITICAL_STOP":
        print(f"[ActionDispatcher] SHORT-CIRCUIT: CRITICAL_STOP detected!")
        print(f"[ActionDispatcher] Emergency stop - NOT waiting for Branch B or C!")

        # Apply processing delay (decision computation)
        time.sleep(PROCESSING_DELAY_MS / 1000.0)

        return build_emergency_response(
            safety_data, start_time, pre_resolved, branch_timings
        )

    # Normal path: fetch remaining branches
    print("[ActionDispatcher] Safety OK - awaiting context and prediction branches")

    for idx in [1, 2]:
        fetch_start = time.time()
        data = await inputs.get_async(idx)
        fetch_duration_ms = (time.time() - fetch_start) * 1000

        if fetch_duration_ms < 10:
            pre_resolved += 1

        if isinstance(data, str):
            data = json.loads(data)

        func_name = data.get("function", f"index-{idx}")
        pre_tag = "PRE-RESOLVED" if fetch_duration_ms < 10 else f"Awaited {fetch_duration_ms:.2f}ms"
        print(f"[ActionDispatcher] Input {idx} ({func_name}): {pre_tag}")

        branch_timings.append({
            "index": idx,
            "function": func_name,
            "fetch_duration_ms": fetch_duration_ms,
            "pre_resolved": fetch_duration_ms < 10
        })
        all_data.append(data)

    # Apply processing delay (decision computation)
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    return build_normal_response(all_data, start_time, pre_resolved, branch_timings)


def process_inputs_sync(inputs, start_time):
    """
    Sync processing for CLASSIC/EAGER modes.

    In CLASSIC: All inputs already resolved
    In EAGER: May block synchronously on each access
    """
    print("[ActionDispatcher] Sync mode: Accessing all inputs")

    all_data = []
    pre_resolved = 0
    branch_timings = []

    for i, data in enumerate(inputs):
        access_start = time.time()

        if isinstance(data, str):
            data = json.loads(data)

        all_data.append(data)
        access_time_ms = (time.time() - access_start) * 1000

        if access_time_ms < 10:
            pre_resolved += 1

        func_name = data.get("function", f"index-{i}")
        branch_timings.append({
            "index": i,
            "function": func_name,
            "access_time_ms": access_time_ms,
            "pre_resolved": access_time_ms < 10
        })

        print(f"[ActionDispatcher] Input {i} ({func_name}): {access_time_ms:.2f}ms")

    # Apply processing delay
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    # Check for CRITICAL_STOP even in CLASSIC mode
    safety_data = all_data[0] if all_data else {}
    if safety_data.get("status") == "CRITICAL_STOP":
        print(f"[ActionDispatcher] CLASSIC mode: CRITICAL_STOP but already waited for all branches")
        return build_emergency_response(
            safety_data, start_time, pre_resolved, branch_timings, wasted=True
        )

    return build_normal_response(all_data, start_time, pre_resolved, branch_timings)


def build_emergency_response(safety_data, start_time, pre_resolved, branch_timings, wasted=False):
    """
    Emergency stop response.

    In FUTURE_BASED: Short-circuited, saving ~1900ms by not waiting for heavy path.
    In CLASSIC: Wasted work - waited for all branches before deciding.
    """
    aggregation_time = (time.time() - start_time) * 1000

    machine_id = safety_data.get("machine_id", "unknown")
    safety_details = safety_data.get("safety_details", {})

    result = {
        "function": "ActionDispatcher",
        "status": "EMERGENCY_STOP",
        "short_circuited": not wasted,
        "machine_id": machine_id,
        "action": "EMERGENCY_SHUTDOWN",
        "safety_details": safety_details,
        "shift_context": None if not wasted else "available_but_irrelevant",
        "failure_prediction": None if not wasted else "available_but_irrelevant",
        "branches_used": 1 if not wasted else 3,
        "pre_resolved_count": pre_resolved,
        "branch_timings": branch_timings,
        "aggregation_time_ms": int(aggregation_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "time_saved_ms": 1900 if not wasted else 0,
        "mode": "FUTURE_BASED" if not wasted else "CLASSIC",
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "action_dispatcher_emergency",
        "machine_id": machine_id,
        "short_circuited": not wasted,
        "aggregation_time_ms": int(aggregation_time)
    }))

    if wasted:
        print(f"[ActionDispatcher] EMERGENCY_STOP in {aggregation_time:.2f}ms (CLASSIC - wasted work)")
        print(f"[ActionDispatcher] WASTED: Waited for all branches but rejected anyway!")
    else:
        print(f"[ActionDispatcher] EMERGENCY_STOP in {aggregation_time:.2f}ms (short-circuited!)")
        print(f"[ActionDispatcher] SAVED ~1900ms by not waiting for Branch B and C!")
        print(f"[ActionDispatcher] FUTURE mode benefit: Immediate emergency response")

    return result


def build_normal_response(all_data, start_time, pre_resolved, branch_timings):
    """
    Normal aggregation of all 3 branches.

    Combines safety status, operational context, and failure prediction
    into a unified action dispatch decision.
    """
    aggregation_time = (time.time() - start_time) * 1000

    # Extract results from each branch
    safety = all_data[0] if len(all_data) > 0 else {}
    context = all_data[1] if len(all_data) > 1 else {}
    prediction = all_data[2] if len(all_data) > 2 else {}

    machine_id = safety.get("machine_id", "unknown")

    # Determine action based on combined analysis
    failure_prob = prediction.get("prediction", {}).get("failure_probability", 0)
    risk_level = prediction.get("prediction", {}).get("risk_level", "LOW")
    maintenance_window = context.get("shift_context", {}).get("maintenance_window", False)

    if risk_level == "HIGH":
        if maintenance_window:
            action = "IMMEDIATE_MAINTENANCE"
        else:
            action = "SCHEDULE_URGENT_MAINTENANCE"
    elif risk_level == "MEDIUM":
        action = "INCREASE_MONITORING"
    else:
        action = "NOMINAL_OPERATION"

    # Calculate branch processing times
    branch_times = {
        "safety_check_ms": safety.get("processing_time_ms", 0),
        "context_chain_ms": context.get("processing_time_ms", 0),
        "heavy_chain_ms": prediction.get("processing_time_ms", 0)
    }

    result = {
        "function": "ActionDispatcher",
        "status": "dispatched",
        "short_circuited": False,
        "machine_id": machine_id,
        "action": action,
        "safety_status": safety.get("status"),
        "safety_details": safety.get("safety_details", {}),
        "shift_context": context.get("shift_context", {}),
        "machine_state": context.get("machine_state", {}),
        "failure_prediction": prediction.get("prediction", {}),
        "branches_used": 3,
        "pre_resolved_count": pre_resolved,
        "branch_timings": branch_timings,
        "branch_processing_times": branch_times,
        "timing_variance_ms": max(branch_times.values()) - min(branch_times.values()) if branch_times else 0,
        "aggregation_time_ms": int(aggregation_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "mode": "FUTURE_BASED" if os.environ.get("UNUM_FUTURE_BASED") == "true" else "CLASSIC",
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "action_dispatcher_complete",
        "machine_id": machine_id,
        "action": action,
        "risk_level": risk_level,
        "aggregation_time_ms": int(aggregation_time),
        "pre_resolved_count": pre_resolved
    }))

    print(f"[ActionDispatcher] COMPLETE in {aggregation_time:.2f}ms")
    print(f"[ActionDispatcher] Action: {action} (risk={risk_level}, p={failure_prob:.4f})")
    print(f"[ActionDispatcher] Pre-resolved: {pre_resolved}/3")
    print(f"[ActionDispatcher] Timing variance: {result['timing_variance_ms']:.0f}ms")

    return result


if __name__ == '__main__':
    # Simulate sync aggregation with test data
    test_results = [
        {
            "function": "SafetyCheck",
            "machine_id": "MACH-TEST-001",
            "status": "safety_ok",
            "safety_details": {"threshold_exceeded": False, "action": "CONTINUE"},
            "processing_time_ms": 102
        },
        {
            "function": "ShiftCheck",
            "machine_id": "MACH-TEST-001",
            "status": "complete",
            "machine_state": {"temperature_c": 85.5, "rpm": 3200},
            "shift_context": {"current_shift": "day", "maintenance_window": False},
            "processing_time_ms": 505
        },
        {
            "function": "FailureModel",
            "machine_id": "MACH-TEST-001",
            "status": "complete",
            "prediction": {
                "failure_probability": 0.15,
                "risk_level": "MEDIUM",
                "predicted_rul_hours": 2500
            },
            "processing_time_ms": 2010
        }
    ]

    result = lambda_handler(test_results, None)
    print(f"\nResult: {json.dumps(result, indent=2)}")
