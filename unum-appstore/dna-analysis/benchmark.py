#!/usr/bin/env python3
"""
DNA Analysis Pipeline Benchmark — Normal vs Streaming

Runs 3 warm + 3 cold iterations for both Normal and Streaming modes.
Collects per-function metrics from CloudWatch REPORT lines.
Saves results to JSON and prints a summary.

Usage:
    python benchmark.py
"""

import boto3
import json
import time
import statistics
import subprocess
import os
import sys
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────

REGION = "eu-central-1"
STACK_NAME = "unum-dna-analysis"
WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
UNUM_CLI = os.path.join(WORKFLOW_DIR, "..", "..", "unum", "unum-cli", "unum-cli.py")

WARM_RUNS = 3
COLD_RUNS = 3

# Function logical names → physical names (populated from CloudFormation)
FUNCTION_ORDER = ["Reader", "DnaAnalyzer", "Comparator", "Reporter"]
FUNCTIONS = {}  # filled by discover_functions()

LAMBDA = boto3.client("lambda", region_name=REGION)
LOGS = boto3.client("logs", region_name=REGION)
CF = boto3.client("cloudformation", region_name=REGION)

MEMORY_COST_PER_GB_S = 0.0000166667  # USD per GB-second

TEST_PAYLOAD = {
    "Data": {
        "Source": "http",
        "Value": {
            "sequence_length": 50000,
            "gc_content": 0.5,
            "seq_id": "benchmark_run"
        }
    }
}


# ── Helpers ──────────────────────────────────────────────────────────────

def discover_functions():
    """Read function ARNs from the CloudFormation stack outputs."""
    global FUNCTIONS
    resp = CF.describe_stacks(StackName=STACK_NAME)
    outputs = resp["Stacks"][0]["Outputs"]
    for out in outputs:
        key = out["OutputKey"]             # e.g. "ReaderFunction"
        arn = out["OutputValue"]
        short = key.replace("Function", "")  # "Reader"
        # Physical name is the last part of the ARN
        physical = arn.split(":")[-1]
        FUNCTIONS[short] = physical
    print(f"  Discovered {len(FUNCTIONS)} functions")
    for name in FUNCTION_ORDER:
        print(f"    {name}: {FUNCTIONS.get(name, '???')}")


def force_cold_start():
    """Update env var on every function to force new containers."""
    ts = str(time.time())
    for name in FUNCTION_ORDER:
        physical = FUNCTIONS[name]
        try:
            # Get current env first, then merge
            cfg = LAMBDA.get_function_configuration(FunctionName=physical)
            env = cfg.get("Environment", {}).get("Variables", {})
            env["COLD_START_TRIGGER"] = ts
            LAMBDA.update_function_configuration(
                FunctionName=physical,
                Environment={"Variables": env}
            )
        except Exception as e:
            print(f"    Warning ({name}): {e}")
    # Wait for propagation – must wait until LastUpdateStatus is Successful
    print("    Waiting for config updates…", end="", flush=True)
    for _ in range(30):
        time.sleep(2)
        try:
            cfg = LAMBDA.get_function_configuration(FunctionName=FUNCTIONS["Reader"])
            if cfg.get("LastUpdateStatus") == "Successful":
                break
        except Exception:
            pass
        print(".", end="", flush=True)
    print(" done")


def invoke_pipeline():
    """Invoke the Reader (entry) function and return wall-clock time + response."""
    t0 = time.time()
    resp = LAMBDA.invoke(
        FunctionName=FUNCTIONS["Reader"],
        InvocationType="RequestResponse",
        Payload=json.dumps(TEST_PAYLOAD),
    )
    wall = time.time() - t0
    payload = json.loads(resp["Payload"].read())
    error = resp.get("FunctionError")
    return wall, payload, error, t0


def wait_for_reporter(start_ts, timeout=180):
    """Poll CloudWatch until Reporter's REPORT line appears, return its timestamp."""
    log_group = f"/aws/lambda/{FUNCTIONS['Reporter']}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            events = LOGS.filter_log_events(
                logGroupName=log_group,
                startTime=int(start_ts * 1000),
                filterPattern="REPORT",
            ).get("events", [])
            for ev in events:
                if "REPORT" in ev.get("message", ""):
                    return ev["timestamp"] / 1000.0
        except Exception:
            pass
        time.sleep(2)
    return None


def collect_function_metrics(start_ts):
    """Collect REPORT-line metrics for every function from CloudWatch."""
    # Give CloudWatch a moment to ingest
    time.sleep(5)
    metrics = {}
    for name in FUNCTION_ORDER:
        log_group = f"/aws/lambda/{FUNCTIONS[name]}"
        try:
            events = LOGS.filter_log_events(
                logGroupName=log_group,
                startTime=int(start_ts * 1000),
                filterPattern="REPORT",
            ).get("events", [])
            report = parse_report(events)
            if report:
                metrics[name] = report
        except Exception as e:
            print(f"    Warning: could not fetch logs for {name}: {e}")

        # Also grab per-stage timing from application logs
        try:
            events = LOGS.filter_log_events(
                logGroupName=log_group,
                startTime=int(start_ts * 1000),
                filterPattern="COMPLETE",
            ).get("events", [])
            for ev in events:
                msg = ev.get("message", "")
                if "COMPLETE in" in msg:
                    ms = int(msg.split("COMPLETE in")[1].strip().rstrip("ms"))
                    if name in metrics:
                        metrics[name]["app_duration_ms"] = ms
        except Exception:
            pass
    return metrics


def parse_report(events):
    """Parse a CloudWatch REPORT line into a dict."""
    for ev in events:
        msg = ev.get("message", "")
        if "REPORT" not in msg or "Billed Duration" not in msg:
            continue
        result = {"timestamp": ev.get("timestamp", 0)}
        for part in msg.replace("\n", "\t").split("\t"):
            part = part.strip()
            if part.startswith("Duration:"):
                result["duration_ms"] = float(part.split(":")[1].strip().split()[0])
            elif part.startswith("Billed Duration:"):
                result["billed_duration_ms"] = int(part.split(":")[1].strip().split()[0])
            elif part.startswith("Memory Size:"):
                result["memory_size_mb"] = int(part.split(":")[1].strip().split()[0])
            elif part.startswith("Max Memory Used:"):
                result["max_memory_mb"] = int(part.split(":")[1].strip().split()[0])
            elif part.startswith("Init Duration:"):
                result["init_duration_ms"] = float(part.split(":")[1].strip().split()[0])
        return result
    return None


def compute_cost(func_metrics):
    """Compute estimated cost from a single function's REPORT data."""
    mem_gb = func_metrics.get("memory_size_mb", 128) / 1024
    dur_s = func_metrics.get("billed_duration_ms", 0) / 1000
    return mem_gb * dur_s * MEMORY_COST_PER_GB_S


# ── Single iteration ────────────────────────────────────────────────────

def run_iteration(run_id, cold=False):
    """Run one end-to-end pipeline invocation and collect all metrics."""
    tag = "[COLD]" if cold else "[WARM]"
    print(f"  Run {run_id} {tag} ", end="", flush=True)

    if cold:
        force_cold_start()

    wall, payload, error, t0 = invoke_pipeline()
    if error:
        print(f"  ERROR: {error} — {str(payload)[:200]}")
        return None

    # Wait for Reporter to finish (the pipeline is async after Reader)
    reporter_ts = wait_for_reporter(t0)
    if reporter_ts:
        e2e = reporter_ts - t0
    else:
        print("(reporter timeout, using wall) ", end="")
        e2e = wall

    func_metrics = collect_function_metrics(t0)

    total_billed = sum(m.get("billed_duration_ms", 0) for m in func_metrics.values())
    total_cost = sum(compute_cost(m) for m in func_metrics.values())

    print(f"E2E={e2e:.2f}s  billed={total_billed}ms  cost=${total_cost:.6f}")

    return {
        "run_id": run_id,
        "cold": cold,
        "e2e_latency_s": round(e2e, 3),
        "wall_clock_s": round(wall, 3),
        "total_billed_ms": total_billed,
        "total_cost_usd": total_cost,
        "functions": func_metrics,
    }


# ── Full benchmark for one mode ─────────────────────────────────────────

def benchmark_mode(mode_label):
    """Run COLD_RUNS cold + WARM_RUNS warm iterations, return structured results."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARKING: {mode_label.upper()}")
    print(f"{'='*60}")

    cold_results = []
    warm_results = []

    # ── Cold runs ──
    print(f"\n  [Cold Start Runs: {COLD_RUNS}]")
    for i in range(1, COLD_RUNS + 1):
        r = run_iteration(i, cold=True)
        if r:
            cold_results.append(r)

    # ── Warm-up invocation (discard) ──
    print("\n  [Warm-up invocation, discarded]")
    invoke_pipeline()
    time.sleep(5)

    # ── Warm runs ──
    print(f"\n  [Warm Runs: {WARM_RUNS}]")
    for i in range(1, WARM_RUNS + 1):
        r = run_iteration(i, cold=False)
        if r:
            warm_results.append(r)

    # ── Summaries ──
    def summarise(runs):
        lats = [r["e2e_latency_s"] for r in runs]
        billed = [r["total_billed_ms"] for r in runs]
        costs = [r["total_cost_usd"] for r in runs]
        return {
            "count": len(runs),
            "avg_latency_s": round(statistics.mean(lats), 3) if lats else 0,
            "min_latency_s": round(min(lats), 3) if lats else 0,
            "max_latency_s": round(max(lats), 3) if lats else 0,
            "std_dev_s": round(statistics.stdev(lats), 3) if len(lats) > 1 else 0,
            "avg_billed_ms": round(statistics.mean(billed), 1) if billed else 0,
            "avg_cost_usd": statistics.mean(costs) if costs else 0,
        }

    # Per-function averages (warm only)
    per_func = {}
    for fname in FUNCTION_ORDER:
        durations = [r["functions"][fname]["duration_ms"]
                     for r in warm_results if fname in r["functions"]
                     and "duration_ms" in r["functions"][fname]]
        mems = [r["functions"][fname]["max_memory_mb"]
                for r in warm_results if fname in r["functions"]
                and "max_memory_mb" in r["functions"][fname]]
        app_durs = [r["functions"][fname]["app_duration_ms"]
                    for r in warm_results if fname in r["functions"]
                    and "app_duration_ms" in r["functions"][fname]]
        per_func[fname] = {
            "avg_duration_ms": round(statistics.mean(durations), 1) if durations else 0,
            "avg_memory_mb": round(statistics.mean(mems), 1) if mems else 0,
            "avg_app_duration_ms": round(statistics.mean(app_durs), 1) if app_durs else 0,
        }

    return {
        "mode": mode_label,
        "timestamp": datetime.now().isoformat(),
        "cold_summary": summarise(cold_results),
        "warm_summary": summarise(warm_results),
        "per_function": per_func,
        "cold_runs": cold_results,
        "warm_runs": warm_results,
    }


# ── Deploy helpers ───────────────────────────────────────────────────────

def deploy(streaming=False):
    flag = "--streaming" if streaming else ""
    label = "STREAMING" if streaming else "NORMAL"
    print(f"\n  Deploying {label} mode…")

    python = sys.executable  # use same interpreter as this script
    cmds = [
        f'"{python}" "{UNUM_CLI}" build -g -p aws {flag}',
        f'"{python}" "{UNUM_CLI}" deploy',
    ]
    for cmd in cmds:
        print(f"    $ {cmd}")
        result = subprocess.run(cmd, shell=True, cwd=WORKFLOW_DIR,
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    STDOUT: {result.stdout[-500:]}")
            print(f"    STDERR: {result.stderr[:500]}")
            raise RuntimeError(f"Deploy command failed: {cmd}")

    print("    Waiting 15s for deployment to stabilise…")
    time.sleep(15)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DNA ANALYSIS PIPELINE — FULL BENCHMARK")
    print("Normal vs Partial-Parameter-Streaming")
    print(f"Pipeline: Reader → DnaAnalyzer → Comparator → Reporter")
    print(f"Sequence length: 50 000 bp, 5 fields per stage")
    print(f"Warm runs: {WARM_RUNS}, Cold runs: {COLD_RUNS}")
    print("=" * 60)

    discover_functions()

    # ── 1. Normal mode ──────────────────────────────────────────────
    deploy(streaming=False)
    discover_functions()  # ARNs unchanged but refresh just in case
    normal = benchmark_mode("normal")

    # ── 2. Streaming mode ───────────────────────────────────────────
    deploy(streaming=True)
    discover_functions()
    streaming = benchmark_mode("streaming")

    # ── 3. Combine & save ───────────────────────────────────────────
    results = {
        "pipeline": "dna-analysis",
        "region": REGION,
        "sequence_length": 50000,
        "stages": 4,
        "fields_per_stage": 5,
        "warm_runs": WARM_RUNS,
        "cold_runs": COLD_RUNS,
        "generated": datetime.now().isoformat(),
        "normal": normal,
        "streaming": streaming,
    }

    out_file = os.path.join(WORKFLOW_DIR, "benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_file}")

    # ── 4. Print comparison table ───────────────────────────────────
    print_comparison(results)

    return results


def print_comparison(results):
    n = results["normal"]
    s = results["streaming"]
    nw = n["warm_summary"]
    sw = s["warm_summary"]
    nc = n["cold_summary"]
    sc = s["cold_summary"]

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Normal':>12} {'Streaming':>12} {'Δ':>10}")
    print("-" * 64)

    def row(label, nv, sv, unit="", fmt=".3f"):
        if nv and sv:
            delta = ((sv - nv) / nv) * 100
            sign = "+" if delta > 0 else ""
            print(f"{label:<30} {nv:>10{fmt}}{unit:>2} {sv:>10{fmt}}{unit:>2} {sign}{delta:>7.1f}%")
        else:
            print(f"{label:<30} {'N/A':>12} {'N/A':>12}")

    row("Warm E2E latency", nw["avg_latency_s"], sw["avg_latency_s"], "s")
    row("Cold E2E latency", nc["avg_latency_s"], sc["avg_latency_s"], "s")
    row("Warm billed (total)", nw["avg_billed_ms"], sw["avg_billed_ms"], "ms", ".0f")
    row("Warm cost", nw["avg_cost_usd"] * 1e6, sw["avg_cost_usd"] * 1e6, "µ$", ".2f")

    print(f"\n  Per-function duration (warm avg, ms):")
    print(f"  {'Function':<20} {'Normal':>10} {'Streaming':>10}")
    print(f"  {'-'*42}")
    for fname in FUNCTION_ORDER:
        nd = n["per_function"].get(fname, {}).get("avg_duration_ms", 0)
        sd = s["per_function"].get(fname, {}).get("avg_duration_ms", 0)
        print(f"  {fname:<20} {nd:>10.1f} {sd:>10.1f}")

    print(f"\n  Per-function memory (warm avg, MB):")
    print(f"  {'Function':<20} {'Normal':>10} {'Streaming':>10}")
    print(f"  {'-'*42}")
    for fname in FUNCTION_ORDER:
        nm = n["per_function"].get(fname, {}).get("avg_memory_mb", 0)
        sm = s["per_function"].get(fname, {}).get("avg_memory_mb", 0)
        print(f"  {fname:<20} {nm:>10.1f} {sm:>10.1f}")

    # Speedup
    if nw["avg_latency_s"] and sw["avg_latency_s"]:
        speedup = nw["avg_latency_s"] / sw["avg_latency_s"]
        improvement = (1 - sw["avg_latency_s"] / nw["avg_latency_s"]) * 100
        print(f"\n  ★ Warm speedup:      {speedup:.2f}x")
        print(f"  ★ Warm improvement:  {improvement:.1f}%")

    if nc["avg_latency_s"] and sc["avg_latency_s"]:
        cold_speedup = nc["avg_latency_s"] / sc["avg_latency_s"]
        cold_imp = (1 - sc["avg_latency_s"] / nc["avg_latency_s"]) * 100
        print(f"  ★ Cold speedup:      {cold_speedup:.2f}x")
        print(f"  ★ Cold improvement:  {cold_imp:.1f}%")


if __name__ == "__main__":
    results = main()
