#!/usr/bin/env python3
"""
Benchmark Runner for Order Processing Workflow

Compares execution modes:
1. CLASSIC - Aggregator invoked by SlowChainEnd (after ~3000ms)
2. FUTURE_BASED - Aggregator invoked by FastProcessor (after ~100ms)

Workflow Structure:
  TriggerFunction -> [FastProcessor (100ms), SlowChainStart->SlowChainMid->SlowChainEnd (3000ms)] -> Aggregator

Expected Benefit:
  FUTURE mode hides Aggregator cold start (~200ms) behind slow chain execution

Usage:
    python run_benchmark.py --mode CLASSIC --iterations 5
    python run_benchmark.py --mode FUTURE_BASED --iterations 5
    python run_benchmark.py --all --iterations 5
"""

import argparse
import datetime
import json
import os
import re
import time
from pathlib import Path

import boto3
import yaml

# Configuration
REGION = os.environ.get("AWS_REGION", "eu-central-1")
STACK_NAME = os.environ.get("STACK_NAME", "order-processing-workflow")
PROFILE = os.environ.get("AWS_PROFILE", "default")
LOG_BUFFER_SEC = int(os.environ.get("LOG_BUFFER_SEC", "60"))
LOG_QUERY_LIMIT = int(os.environ.get("LOG_QUERY_LIMIT", "200"))

# Mode configurations
MODE_CONFIGS = {
    "CLASSIC": {
        "EAGER": "true",
        "UNUM_FUTURE_BASED": "false",
    },
    "FUTURE_BASED": {
        "EAGER": "true",
        "UNUM_FUTURE_BASED": "true",
    },
}

FUNCTION_LOGICAL_NAMES = [
    "TriggerFunction",
    "FastProcessor",
    "SlowChainStart",
    "SlowChainMid",
    "SlowChainEnd",
    "Aggregator",
]

FUNCTION_ALIASES = {
    "Trigger": "TriggerFunction",
    "TriggerFunctionFunction": "TriggerFunction",
}

# Initialize AWS clients
if PROFILE:
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
else:
    session = boto3.Session(region_name=REGION)

lambda_client = session.client("lambda")
logs_client = session.client("logs")
cloudformation_client = session.client("cloudformation")

_FUNCTION_ARNS = None


def normalize_function_key(name):
    """Normalize function logical names from stack outputs/resources."""
    if name.endswith("FunctionFunction"):
        return name[:-8]
    return name


def resolve_function_key(functions, name):
    """Resolve a function key using aliases and heuristics."""
    name = FUNCTION_ALIASES.get(name, name)

    if name in functions:
        return name
    if f"{name}Function" in functions:
        return f"{name}Function"
    if name.endswith("Function") and name[:-8] in functions:
        return name[:-8]

    lower = {k.lower(): k for k in functions}
    if name.lower() in lower:
        return lower[name.lower()]

    matches = [k for k in functions if name.lower() in k.lower()]
    if len(matches) == 1:
        return matches[0]

    return None


def load_function_arns():
    """Load function identifiers (ARN or name) from function-arn.yaml or CloudFormation."""
    global _FUNCTION_ARNS
    if _FUNCTION_ARNS is not None:
        return _FUNCTION_ARNS

    yaml_path = Path(__file__).parent.parent / "function-arn.yaml"
    functions = {}

    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            functions = yaml.safe_load(f) or {}
        if functions:
            print(f"✓ Loaded {len(functions)} functions from function-arn.yaml")
            _FUNCTION_ARNS = functions
            return functions

    # Fallback to CloudFormation discovery
    try:
        stack = cloudformation_client.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
        outputs = {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}

        for key, value in outputs.items():
            if ":function:" in value:
                functions[normalize_function_key(key)] = value

        if functions:
            print(f"✓ Discovered {len(functions)} functions from CloudFormation outputs")
            _FUNCTION_ARNS = functions
            return functions

        paginator = cloudformation_client.get_paginator("list_stack_resources")
        for page in paginator.paginate(StackName=STACK_NAME):
            for resource in page.get("StackResourceSummaries", []):
                if resource.get("ResourceType") == "AWS::Lambda::Function":
                    logical = normalize_function_key(resource.get("LogicalResourceId", ""))
                    physical = resource.get("PhysicalResourceId")
                    if logical and physical:
                        functions[logical] = physical

        if functions:
            print(f"✓ Discovered {len(functions)} functions from CloudFormation resources")
            _FUNCTION_ARNS = functions
            return functions

    except Exception as exc:
        print(f"✗ CloudFormation discovery failed: {exc}")

    print("✗ Error: Could not discover function identifiers.")
    print("  Please deploy the workflow first using: unum-cli.py deploy -t unum-template.yaml")

    _FUNCTION_ARNS = {}
    return {}


def get_function_id(func_name):
    """Get Lambda function identifier (ARN or name) from discovered mapping."""
    functions = load_function_arns()
    key = resolve_function_key(functions, func_name)
    if key:
        return functions[key]

    print(f"✗ Warning: function identifier not found for {func_name}")
    return None


def function_name_for_logs(function_id):
    """Extract Lambda function name for CloudWatch log group."""
    if not function_id:
        return None

    if function_id.startswith("arn:"):
        name = function_id.split(":function:")[-1]
        return name.split(":")[0]

    return function_id


def configure_mode(mode):
    """Configure Lambda environment variables for the specified mode."""
    print()
    print("=" * 60)
    print(f"Configuring {mode} mode")
    print("=" * 60)

    config = MODE_CONFIGS[mode]
    aggregator_id = get_function_id("Aggregator")

    if not aggregator_id:
        print("✗ Cannot configure mode: function identifiers not loaded")
        raise ValueError("Function identifiers not available. Deploy workflow first.")

    try:
        response = lambda_client.get_function_configuration(FunctionName=aggregator_id)
        current_env = response.get("Environment", {}).get("Variables", {})

        new_env = {**current_env, **config}

        lambda_client.update_function_configuration(
            FunctionName=aggregator_id,
            Environment={"Variables": new_env}
        )

        print("✓ Updated Aggregator")
        print(f"  - EAGER: {config['EAGER']}")
        print(f"  - UNUM_FUTURE_BASED: {config['UNUM_FUTURE_BASED']}")

        # Wait for update to complete
        time.sleep(5)

    except Exception as exc:
        print(f"✗ Failed to update Aggregator: {exc}")
        raise


def force_cold_start(function_name):
    """Force cold start by updating Lambda environment variable."""
    function_id = get_function_id(function_name)

    if not function_id:
        print(f"  ✗ Cannot force cold start: function ID not found for {function_name}")
        return

    try:
        response = lambda_client.get_function_configuration(FunctionName=function_id)
        current_env = response.get("Environment", {}).get("Variables", {})

        current_env["FORCE_COLD_START"] = str(int(time.time()))

        lambda_client.update_function_configuration(
            FunctionName=function_id,
            Environment={"Variables": current_env}
        )

        waiter = lambda_client.get_waiter("function_updated")
        waiter.wait(FunctionName=function_id)

        print(f"  ✓ Forced cold start: {function_name}")

    except Exception as exc:
        print(f"  ✗ Failed to force cold start for {function_name}: {exc}")


def invoke_workflow(order_id, cold_start=False):
    """Invoke the workflow via TriggerFunction."""
    trigger_id = get_function_id("TriggerFunction")

    if not trigger_id:
        return {
            "success": False,
            "order_id": order_id,
            "error": "TriggerFunction identifier not found. Deploy workflow first."
        }

    if cold_start:
        print()
        print("  Forcing cold starts...")
        for func in FUNCTION_LOGICAL_NAMES:
            force_cold_start(func)
        time.sleep(10)

    payload = {
        "order_id": order_id,
        "customer_id": "CUST-BENCH-001",
        "items": [
            {"sku": "ITEM-001", "quantity": 2, "price": 49.99},
            {"sku": "ITEM-002", "quantity": 1, "price": 29.99}
        ]
    }

    start_time = time.time()

    try:
        response = lambda_client.invoke(
            FunctionName=trigger_id,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )

        end_time = time.time()
        trigger_latency_ms = (end_time - start_time) * 1000

        response_payload = json.loads(response["Payload"].read())

        return {
            "success": True,
            "order_id": order_id,
            "trigger_latency_ms": trigger_latency_ms,
            "response": response_payload,
            "start_time": start_time,
            "end_time": end_time,
        }

    except Exception as exc:
        return {
            "success": False,
            "order_id": order_id,
            "error": str(exc)
        }


def get_cloudwatch_logs(function_id, start_time, end_time, filter_pattern=None, limit=None):
    """Retrieve CloudWatch logs for a function within a time range."""
    function_name = function_name_for_logs(function_id)
    if not function_name:
        return []

    log_group = f"/aws/lambda/{function_name}"

    try:
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + (LOG_BUFFER_SEC * 1000)

        response = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=start_ms,
            endTime=end_ms,
            filterPattern=filter_pattern or "",
            limit=limit or LOG_QUERY_LIMIT,
        )

        return response.get("events", [])

    except Exception as exc:
        print(f"    Warning: Could not fetch logs for {function_name}: {exc}")
        return []


REPORT_DURATION_RE = re.compile(r"Duration: ([0-9.]+) ms")
REPORT_BILLED_RE = re.compile(r"Billed Duration: ([0-9.]+) ms")
REPORT_MEMORY_RE = re.compile(r"Max Memory Used: ([0-9]+) MB")
REPORT_INIT_RE = re.compile(r"Init Duration: ([0-9.]+) ms")


def extract_metrics(function_name, start_time, end_time):
    """Extract metrics from CloudWatch REPORT logs for a function."""
    function_id = get_function_id(function_name)
    if not function_id:
        return {
            "duration_ms": 0,
            "billed_duration_ms": 0,
            "memory_used_mb": 0,
            "init_duration_ms": 0,
            "cold_start": False,
            "report_timestamp_ms": 0,
            "start_timestamp_ms": 0,
        }

    logs = get_cloudwatch_logs(
        function_id,
        start_time,
        end_time,
        filter_pattern="REPORT RequestId:",
        limit=LOG_QUERY_LIMIT,
    )

    if not logs:
        return {
            "duration_ms": 0,
            "billed_duration_ms": 0,
            "memory_used_mb": 0,
            "init_duration_ms": 0,
            "cold_start": False,
            "report_timestamp_ms": 0,
            "start_timestamp_ms": 0,
        }

    latest = max(logs, key=lambda e: e.get("timestamp", 0))
    message = latest.get("message", "")

    duration_match = REPORT_DURATION_RE.search(message)
    billed_match = REPORT_BILLED_RE.search(message)
    memory_match = REPORT_MEMORY_RE.search(message)
    init_match = REPORT_INIT_RE.search(message)

    duration_ms = float(duration_match.group(1)) if duration_match else 0
    billed_ms = float(billed_match.group(1)) if billed_match else 0
    memory_mb = int(memory_match.group(1)) if memory_match else 0
    init_ms = float(init_match.group(1)) if init_match else 0

    report_timestamp_ms = latest.get("timestamp", 0)
    start_timestamp_ms = report_timestamp_ms - duration_ms

    return {
        "duration_ms": duration_ms,
        "billed_duration_ms": billed_ms,
        "memory_used_mb": memory_mb,
        "init_duration_ms": init_ms,
        "cold_start": init_ms > 0,
        "report_timestamp_ms": report_timestamp_ms,
        "start_timestamp_ms": start_timestamp_ms,
    }


def collect_function_metrics(start_time, end_time, attempts=3, delay_sec=3):
    """Collect metrics with simple retry to allow for log propagation."""
    function_metrics = {name: {} for name in FUNCTION_LOGICAL_NAMES}

    for attempt in range(1, attempts + 1):
        for func in FUNCTION_LOGICAL_NAMES:
            function_metrics[func] = extract_metrics(func, start_time, end_time)

        if function_metrics["Aggregator"].get("duration_ms", 0) > 0:
            return function_metrics

        if attempt < attempts:
            print(f"  Waiting {delay_sec}s for logs to propagate (attempt {attempt}/{attempts})...")
            time.sleep(delay_sec)

    return function_metrics


def run_benchmark(mode, iterations, cold_start_frequency="first"):
    """Run benchmark for specified mode."""
    print()
    print("=" * 60)
    print(f"Running {mode} Benchmark")
    print(f"Iterations: {iterations}")
    print(f"Cold starts: {cold_start_frequency}")
    print("=" * 60)
    print()

    configure_mode(mode)

    results = []

    for i in range(iterations):
        print()
        print(f"[{i+1}/{iterations}] Running iteration {i+1}")

        cold_start = (cold_start_frequency == "all") or (
            cold_start_frequency == "first" and i == 0
        )

        order_id = f"BENCH-{mode}-{i+1:03d}-{int(time.time())}"

        result = invoke_workflow(order_id, cold_start=cold_start)

        if result["success"]:
            print(f"  ✓ Trigger latency: {result['trigger_latency_ms']:.2f}ms")

            print("  Collecting metrics...")
            time.sleep(5)

            function_metrics = collect_function_metrics(result["start_time"], result["end_time"])
            result["function_metrics"] = function_metrics

            agg_report_ts = function_metrics["Aggregator"].get("report_timestamp_ms", 0)
            if agg_report_ts:
                result["workflow_latency_ms"] = agg_report_ts - int(result["start_time"] * 1000)

            for func in FUNCTION_LOGICAL_NAMES:
                metrics = function_metrics[func]
                if metrics.get("duration_ms", 0) > 0:
                    cold_tag = " [COLD]" if metrics.get("cold_start") else ""
                    print(f"    {func}: {metrics['duration_ms']:.0f}ms{cold_tag}")

            results.append(result)

        else:
            print(f"  ✗ Failed: {result['error']}")
            results.append(result)

        if i < iterations - 1:
            print("  Waiting 5s before next iteration...")
            time.sleep(5)

    save_results(mode, results)
    print_summary(mode, results)

    return results


def save_results(mode, results):
    """Save benchmark results to JSON."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"benchmark_{mode}_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print(f"✓ Results saved to: {filename}")


def print_summary(mode, results):
    """Print benchmark summary statistics."""
    successful = [r for r in results if r.get("success")]

    if not successful:
        print()
        print("✗ No successful runs to summarize")
        return

    trigger_latencies = [r["trigger_latency_ms"] for r in successful]
    avg_trigger = sum(trigger_latencies) / len(trigger_latencies)

    workflow_latencies = [r.get("workflow_latency_ms") for r in successful if r.get("workflow_latency_ms")]
    avg_workflow = sum(workflow_latencies) / len(workflow_latencies) if workflow_latencies else 0

    agg_durations = [
        r["function_metrics"]["Aggregator"]["duration_ms"]
        for r in successful
        if r["function_metrics"]["Aggregator"]["duration_ms"] > 0
    ]
    avg_agg = sum(agg_durations) / len(agg_durations) if agg_durations else 0

    cold_starts = sum(
        1 for r in successful
        if r["function_metrics"]["Aggregator"].get("cold_start")
    )

    print()
    print("=" * 60)
    print(f"{mode} Benchmark Summary")
    print("=" * 60)
    print(f"Total runs: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    print()
    print("Trigger Latency:")
    print(f"  Average: {avg_trigger:.2f}ms")
    print()
    print("Workflow Latency (Trigger start -> Aggregator REPORT):")
    if avg_workflow:
        print(f"  Average: {avg_workflow:.2f}ms")
        print(f"  Min: {min(workflow_latencies):.2f}ms")
        print(f"  Max: {max(workflow_latencies):.2f}ms")
    else:
        print("  Not available (Aggregator REPORT not found)")
    print()
    print("Aggregator:")
    print(f"  Average duration: {avg_agg:.2f}ms")
    print(f"  Cold starts: {cold_starts}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Order Processing Workflow")
    parser.add_argument("--mode", choices=["CLASSIC", "FUTURE_BASED"], help="Execution mode")
    parser.add_argument("--all", action="store_true", help="Run all modes")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--cold", action="store_true", help="Force cold starts for all iterations")

    args = parser.parse_args()

    cold_start_freq = "all" if args.cold else "first"

    if args.all:
        modes = ["CLASSIC", "FUTURE_BASED"]
    elif args.mode:
        modes = [args.mode]
    else:
        parser.print_help()
        return

    for mode in modes:
        run_benchmark(mode, args.iterations, cold_start_freq)

        if len(modes) > 1 and mode != modes[-1]:
            print()
            print("Waiting 30s before next mode...")
            time.sleep(30)

    print()
    print("=" * 60)
    print("All benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
