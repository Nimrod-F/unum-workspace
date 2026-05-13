#!/usr/bin/env python3
"""
Smart Factory IoT - Fusion Benchmark
Runs 3 cold + 7 warm experiments on the fused (6-function) deployment.
Collects E2E latency and cost from CloudWatch REPORT logs.
"""

import json
import os
import re
import statistics
import time
from pathlib import Path

import boto3
import yaml

REGION = os.environ.get('AWS_REGION', 'eu-central-1')
PROFILE = os.environ.get('AWS_PROFILE', 'research-profile')

# 6 fused functions
FUNCTIONS = [
    'SensorIngest',
    'SafetyCheck',
    'MachineStateShiftCheck',
    'WindowingComputeFFT',
    'FailureModel',
    'ActionDispatcher',
]

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
lambda_client = session.client('lambda')
logs_client = session.client('logs')

# Pricing (eu-central-1)
LAMBDA_GB_SECOND = 0.0000166667
LAMBDA_REQUEST = 0.0000002
DYNAMODB_WCU = 0.00000125
DYNAMODB_RCU = 0.00000025


def load_arns():
    yaml_path = Path(__file__).parent.parent / 'function-arn.yaml'
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def physical_name(arn):
    return arn.split(':')[-1]


def force_cold_starts(arns):
    ts = str(int(time.time()))
    for name, arn in arns.items():
        cfg = lambda_client.get_function_configuration(FunctionName=arn)
        env = cfg.get('Environment', {}).get('Variables', {})
        env['COLD_START_TRIGGER'] = ts
        lambda_client.update_function_configuration(
            FunctionName=arn,
            Environment={'Variables': env}
        )
    print("  Cold starts forced, waiting 15 s...")
    time.sleep(15)


def invoke_workflow(arns):
    payload = {
        "Data": {
            "Source": "http",
            "Value": {
                "machine_id": f"BENCH-{int(time.time())}",
                "sensor_id": "SENS-BENCH-001",
                "force_critical": False,
            }
        }
    }
    start = time.time()
    resp = lambda_client.invoke(
        FunctionName=arns['SensorIngest'],
        InvocationType='RequestResponse',
        Payload=json.dumps(payload),
    )
    trigger_ms = (time.time() - start) * 1000
    body = json.loads(resp['Payload'].read())
    if 'FunctionError' in resp:
        raise RuntimeError(f"Lambda error: {body}")
    return start, trigger_ms


def collect_report(func_physical, start_epoch, end_epoch):
    log_group = f"/aws/lambda/{func_physical}"
    start_ms = int(start_epoch * 1000) - 5000
    end_ms = int(end_epoch * 1000) + 60000
    try:
        resp = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=start_ms,
            endTime=end_ms,
            limit=200,
        )
    except Exception:
        return None

    reports = [e for e in resp.get('events', [])
               if 'REPORT RequestId:' in e.get('message', '')]
    if not reports:
        return None
    latest = max(reports, key=lambda e: e.get('timestamp', 0))
    msg = latest['message']
    ts = latest['timestamp']

    def extract(pattern, default=0.0):
        m = re.search(pattern, msg)
        return float(m.group(1)) if m else default

    duration = extract(r'Duration:\s+([\d.]+)\s+ms')
    billed = extract(r'Billed Duration:\s+([\d.]+)\s+ms')
    mem_size = int(extract(r'Memory Size:\s+(\d+)\s+MB'))
    mem_used = int(extract(r'Max Memory Used:\s+(\d+)\s+MB'))
    init = extract(r'Init Duration:\s+([\d.]+)\s+ms')

    invocation_ms = int(ts - duration) if duration > 0 else 0
    return {
        'duration_ms': duration,
        'billed_ms': billed,
        'mem_size_mb': mem_size,
        'mem_used_mb': mem_used,
        'init_ms': init,
        'cold_start': init > 0,
        'invocation_ms': invocation_ms,
        'report_ts': ts,
    }


def run_one(arns, run_id, cold=False):
    print(f"\n  Run {run_id} ({'cold' if cold else 'warm'})...")
    if cold:
        force_cold_starts(arns)

    start_epoch, trigger_ms = invoke_workflow(arns)
    print(f"    Trigger: {trigger_ms:.0f} ms")

    # Wait for full pipeline + CloudWatch propagation
    wait = 20 if cold else 15
    print(f"    Waiting {wait} s for completion + log propagation...")
    time.sleep(wait)
    end_epoch = time.time()

    metrics = {}
    for attempt in range(3):
        for name in FUNCTIONS:
            pname = physical_name(arns[name])
            r = collect_report(pname, start_epoch, end_epoch)
            if r and r['duration_ms'] > 0:
                metrics[name] = r
        if 'ActionDispatcher' in metrics and metrics['ActionDispatcher']['duration_ms'] > 0:
            break
        print(f"    Retry metrics collection ({attempt+1}/3)...")
        time.sleep(5)
        end_epoch = time.time()

    # E2E: from invocation start to ActionDispatcher completion
    e2e = None
    if 'ActionDispatcher' in metrics:
        ad = metrics['ActionDispatcher']
        end_ms = ad['invocation_ms'] + ad['duration_ms']
        e2e = end_ms - int(start_epoch * 1000)

    total_billed = sum(m['billed_ms'] for m in metrics.values())
    cold_count = sum(1 for m in metrics.values() if m['cold_start'])

    # Cost: Lambda compute + requests
    gb_seconds = sum(
        (m['billed_ms'] / 1000) * (m['mem_size_mb'] / 1024)
        for m in metrics.values()
    )
    cost = gb_seconds * LAMBDA_GB_SECOND + len(metrics) * LAMBDA_REQUEST

    print(f"    Functions collected: {len(metrics)}/{len(FUNCTIONS)}")
    for name in FUNCTIONS:
        if name in metrics:
            m = metrics[name]
            tag = " [COLD]" if m['cold_start'] else ""
            print(f"      {name:25s}: {m['duration_ms']:7.1f} ms{tag}")
    if e2e:
        print(f"    E2E: {e2e:.0f} ms | Billed: {total_billed:.0f} ms | Cold starts: {cold_count} | Cost: ${cost:.8f}")
    else:
        print(f"    WARNING: Could not compute E2E (ActionDispatcher metrics missing)")

    return {
        'run_id': run_id,
        'type': 'cold' if cold else 'warm',
        'e2e_ms': e2e,
        'trigger_ms': trigger_ms,
        'total_billed_ms': total_billed,
        'cold_starts': cold_count,
        'cost_usd': cost,
        'per_function': metrics,
    }


def main():
    arns = load_arns()
    print(f"Loaded {len(arns)} function ARNs")
    for n, a in arns.items():
        print(f"  {n}: ...{a.split(':')[-1]}")

    # 2 warmup runs (discarded)
    print("\n=== WARMUP (2 runs, discarded) ===")
    for i in range(2):
        invoke_workflow(arns)
        print(f"  Warmup {i+1} done")
        time.sleep(8)

    results = []

    # 3 cold-start runs
    print("\n=== COLD-START RUNS (3) ===")
    for i in range(3):
        r = run_one(arns, i + 1, cold=True)
        results.append(r)

    # 7 warm runs
    print("\n=== WARM RUNS (7) ===")
    # Let containers warm up after the last cold-start cycle
    print("  Letting containers warm up (10 s)...")
    time.sleep(10)
    for i in range(7):
        r = run_one(arns, i + 1, cold=False)
        results.append(r)
        if i < 6:
            time.sleep(5)

    # Summary
    cold_runs = [r for r in results if r['type'] == 'cold' and r['e2e_ms']]
    warm_runs = [r for r in results if r['type'] == 'warm' and r['e2e_ms']]

    print("\n" + "=" * 60)
    print("FUSION BENCHMARK RESULTS")
    print("=" * 60)

    if cold_runs:
        cold_e2e = [r['e2e_ms'] for r in cold_runs]
        cold_cost = [r['cost_usd'] for r in cold_runs]
        print(f"\nCold-start ({len(cold_runs)} runs):")
        print(f"  E2E  mean: {statistics.mean(cold_e2e):.0f} ms")
        print(f"  E2E  each: {[f'{v:.0f}' for v in cold_e2e]}")
        print(f"  Cost mean: ${statistics.mean(cold_cost):.8f}")

    if warm_runs:
        warm_e2e = [r['e2e_ms'] for r in warm_runs]
        warm_cost = [r['cost_usd'] for r in warm_runs]
        print(f"\nWarm ({len(warm_runs)} runs):")
        print(f"  E2E  mean: {statistics.mean(warm_e2e):.0f} ms")
        print(f"  E2E  each: {[f'{v:.0f}' for v in warm_e2e]}")
        print(f"  Cost mean: ${statistics.mean(warm_cost):.8f}")

    # Compute delta% vs baseline (Base: Cold 11810, Warm 3295, Cost 2.24e-5)
    BASE_COLD = 11810
    BASE_WARM = 3295
    BASE_COST = 2.24e-5

    if cold_runs:
        cold_mean = statistics.mean(cold_e2e)
        pct = (cold_mean - BASE_COLD) / BASE_COLD * 100
        print(f"\n  Cold Δ% vs Base: {pct:+.1f}%")
    if warm_runs:
        warm_mean = statistics.mean(warm_e2e)
        pct = (warm_mean - BASE_WARM) / BASE_WARM * 100
        print(f"  Warm Δ% vs Base: {pct:+.1f}%")
    if warm_runs:
        cost_mean = statistics.mean(warm_cost)
        cost_1e5 = cost_mean * 1e5
        pct = (cost_mean - BASE_COST) / BASE_COST * 100
        print(f"  Cost: {cost_1e5:.2f} ×10⁻⁵  Δ% vs Base: {pct:+.1f}%")

    # Save raw results
    out_path = Path(__file__).parent / 'fusion_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {out_path}")


if __name__ == '__main__':
    main()
