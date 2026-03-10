#!/usr/bin/env python3
"""
Smoke Test Script for All 5 Benchmark Workflows
================================================
Invokes each workflow once and verifies it completes successfully.

Workflows:
  1. nlp-pipeline       (Chain: Tokenizer → Analyzer → Classifier → Summarizer)
  2. text-processing    (Parallel+Fan-in: UserMention ∥ FindUrl→ShortenUrl → CreatePost → Publish)
  3. graph-analysis     (3-way Fan-out/in: GraphGenerator → PageRank∥BFS∥MST → Aggregator)
  4. montecarlo-pipeline (Diamond: DataGenerator → Transform→Estimate→Validate ∥ Simulate → Aggregator → Reporter)
  5. wordcount          (MapReduce: UnumMap0 → Mapper×N → Partition → Reducer×3 → Summary)

Usage:
    python smoke_test.py                    # Test all workflows
    python smoke_test.py nlp-pipeline       # Test specific workflow
    python smoke_test.py --timeout 120      # Custom timeout
"""

import json
import time
import uuid
import sys
import os
import threading
import argparse
from datetime import datetime, timezone
from pathlib import Path

import boto3
import yaml

REGION = "eu-central-1"
WORKSPACE = Path(__file__).resolve().parent.parent  # unum-appstore/

# ─── Workflow Definitions ───────────────────────────────────────────────────

WORKFLOWS = {
    "nlp-pipeline": {
        "dir": "nlp-pipeline",
        "start_functions": ["Tokenizer"],
        "end_function": "Summarizer",
        "function_count": 4,
        "test_payload": {
            "text": "Artificial intelligence and machine learning are transforming "
                    "cloud computing. Serverless architectures enable cost-effective "
                    "scaling. Natural language processing allows computers to understand "
                    "human language patterns efficiently.",
            "doc_id": "smoke-test-001"
        },
        "description": "Chain topology (4 functions)"
    },
    "text-processing": {
        "dir": "text-processing",
        "start_functions": ["UserMention", "FindUrl"],
        "end_function": "Publish",
        "function_count": 5,
        "test_payload": "ABC's report on cloud computing trends @CloudExpert "
                        "https://example.com/report shows serverless adoption growing 40% year over year",
        "parallel_start": True,
        "description": "Parallel + Fan-in topology (5 functions)"
    },
    "graph-analysis": {
        "dir": "graph-analysis",
        "start_functions": ["GraphGenerator"],
        "end_function": "Aggregator",
        "function_count": 5,
        "test_payload": {
            "size": 50,
            "seed": 42
        },
        "description": "3-way Fan-out/Fan-in topology (5 functions)"
    },
    "montecarlo-pipeline": {
        "dir": "montecarlo-pipeline",
        "start_functions": ["DataGenerator"],
        "end_function": "Reporter",
        "function_count": 7,
        "test_payload": {
            "matrix_size": 50,
            "n_simulations": 10000,
            "n_walks": 20,
            "seed": 42
        },
        "description": "Diamond topology (7 functions)"
    },
    "wordcount": {
        "dir": "wordcount",
        "start_functions": ["UnumMap0"],
        "end_function": "Summary",
        "function_count": 5,
        "test_payload": [
            {"text": "hello world hello serverless world", "destination": "wordcount-benchmark-683003725669"},
            {"text": "world cloud function lambda serverless", "destination": "wordcount-benchmark-683003725669"},
            {"text": "hello cloud hello lambda hello", "destination": "wordcount-benchmark-683003725669"}
        ],
        "description": "MapReduce topology (5 functions, 2 fan-in points)"
    }
}


class SmokeTestRunner:
    def __init__(self, region=REGION, timeout=90):
        self.region = region
        self.timeout = timeout
        self.lambda_client = boto3.client("lambda", region_name=region)
        self.logs_client = boto3.client("logs", region_name=region)
        self.results = {}

    def load_function_arns(self, workflow_name: str) -> dict:
        """Load function ARN mappings from function-arn.yaml"""
        workflow_dir = WORKSPACE / WORKFLOWS[workflow_name]["dir"]
        arn_file = workflow_dir / "function-arn.yaml"
        
        if not arn_file.exists():
            raise FileNotFoundError(f"function-arn.yaml not found for {workflow_name} at {arn_file}")
        
        with open(arn_file) as f:
            arns = yaml.safe_load(f)
        
        return arns

    def invoke_workflow(self, workflow_name: str) -> tuple:
        """
        Invoke a workflow and return (session_id, invocation_time_ms).
        Handles different topology patterns.
        """
        config = WORKFLOWS[workflow_name]
        arns = self.load_function_arns(workflow_name)
        session_id = str(uuid.uuid4())
        
        invocation_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        if config.get("parallel_start"):
            # text-processing: invoke both start functions with shared session
            self._invoke_parallel_starts(arns, config, session_id)
        else:
            # Standard: single start function
            start_func = config["start_functions"][0]
            start_arn = arns[start_func]
            payload = self._build_payload(config["test_payload"], session_id)
            
            print(f"    Invoking {start_func} ({start_arn.split(':')[-1][:40]}...)")
            self.lambda_client.invoke(
                FunctionName=start_arn,
                InvocationType='Event',  # Async
                Payload=json.dumps(payload)
            )
        
        return session_id, invocation_time

    def _invoke_parallel_starts(self, arns, config, session_id):
        """Invoke multiple start functions in parallel with shared session (text-processing pattern)"""
        start_funcs = config["start_functions"]
        threads = []
        
        for idx, func_name in enumerate(start_funcs):
            payload = {
                "Session": session_id,
                "Fan-out": {"Index": idx},
                "Data": {"Source": "http", "Value": config["test_payload"]}
            }
            
            func_arn = arns[func_name]
            print(f"    Invoking {func_name} (index={idx})")
            
            t = threading.Thread(
                target=self._invoke_async,
                args=(func_arn, payload)
            )
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()

    def _invoke_async(self, function_arn, payload):
        """Invoke a Lambda function asynchronously"""
        self.lambda_client.invoke(
            FunctionName=function_arn,
            InvocationType='Event',
            Payload=json.dumps(payload)
        )

    def _build_payload(self, test_data, session_id):
        """Build the invocation payload wrapping user data in unum format"""
        return {
            "Data": {"Source": "http", "Value": test_data}
        }

    def wait_for_completion(self, workflow_name: str, session_id: str, 
                           invocation_time: int) -> dict:
        """
        Wait for the end function to complete by polling CloudWatch logs.
        Returns timing info or None on timeout.
        """
        config = WORKFLOWS[workflow_name]
        arns = self.load_function_arns(workflow_name)
        end_func = config["end_function"]
        end_arn = arns[end_func]
        
        # Derive log group from ARN
        func_name = end_arn.split(":")[-1]
        log_group = f"/aws/lambda/{func_name}"
        
        print(f"    Waiting for {end_func} to complete (log: {func_name[:40]}...)")
        
        deadline = time.time() + self.timeout
        poll_interval = 2  # Start at 2 seconds
        
        while time.time() < deadline:
            try:
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=invocation_time - 1000,  # 1s buffer
                    filterPattern="REPORT RequestId"
                )
                
                events = response.get("events", [])
                if events:
                    # Found completion! Extract duration from REPORT
                    for event in events:
                        msg = event.get("message", "")
                        if "REPORT" in msg:
                            completion_time = event.get("timestamp", 0)
                            duration = self._extract_duration(msg)
                            billed = self._extract_billed_duration(msg)
                            memory = self._extract_memory(msg)
                            
                            return {
                                "completed": True,
                                "end_to_end_ms": completion_time - invocation_time,
                                "end_function_duration_ms": duration,
                                "end_function_billed_ms": billed,
                                "end_function_memory_mb": memory,
                                "completion_timestamp": completion_time
                            }
                
            except self.logs_client.exceptions.ResourceNotFoundException:
                # Log group may not exist yet
                pass
            except Exception as e:
                print(f"    Warning polling logs: {e}")
            
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.2, 5)  # Back off up to 5s
        
        return {"completed": False, "error": f"Timeout after {self.timeout}s"}

    def check_all_functions_executed(self, workflow_name: str, 
                                     invocation_time: int) -> dict:
        """
        Verify that ALL functions in the workflow executed by checking CloudWatch logs.
        Returns dict mapping function_name → bool (executed).
        """
        arns = self.load_function_arns(workflow_name)
        executed = {}
        
        for func_name, func_arn in arns.items():
            lambda_name = func_arn.split(":")[-1]
            log_group = f"/aws/lambda/{lambda_name}"
            
            try:
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=invocation_time - 1000,
                    filterPattern="REPORT RequestId"
                )
                events = response.get("events", [])
                executed[func_name] = len(events) > 0
            except self.logs_client.exceptions.ResourceNotFoundException:
                executed[func_name] = False
            except Exception:
                executed[func_name] = False
        
        return executed

    def _extract_duration(self, report_msg: str) -> float:
        """Extract Duration from REPORT line"""
        try:
            parts = report_msg.split("Duration: ")[1]
            return float(parts.split(" ms")[0])
        except (IndexError, ValueError):
            return 0.0

    def _extract_billed_duration(self, report_msg: str) -> float:
        """Extract Billed Duration from REPORT line"""
        try:
            parts = report_msg.split("Billed Duration: ")[1]
            return float(parts.split(" ms")[0])
        except (IndexError, ValueError):
            return 0.0

    def _extract_memory(self, report_msg: str) -> int:
        """Extract Max Memory Used from REPORT line"""
        try:
            parts = report_msg.split("Max Memory Used: ")[1]
            return int(parts.split(" MB")[0])
        except (IndexError, ValueError):
            return 0

    def run_smoke_test(self, workflow_name: str) -> dict:
        """Run a complete smoke test for one workflow"""
        config = WORKFLOWS[workflow_name]
        
        print(f"\n{'='*60}")
        print(f"  SMOKE TEST: {workflow_name}")
        print(f"  {config['description']}")
        print(f"{'='*60}")
        
        result = {
            "workflow": workflow_name,
            "description": config["description"],
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "UNKNOWN"
        }
        
        # Phase 1: Check function ARNs exist
        print(f"\n  [1/4] Checking function ARNs...")
        try:
            arns = self.load_function_arns(workflow_name)
            if len(arns) < config["function_count"]:
                result["status"] = "FAIL"
                result["error"] = f"Expected {config['function_count']} functions, found {len(arns)}"
                print(f"    FAIL: {result['error']}")
                return result
            print(f"    OK: {len(arns)} functions found")
            result["functions"] = list(arns.keys())
        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = f"Cannot load ARNs: {e}"
            print(f"    FAIL: {result['error']}")
            return result
        
        # Phase 2: Invoke workflow
        print(f"\n  [2/4] Invoking workflow...")
        try:
            session_id, invocation_time = self.invoke_workflow(workflow_name)
            result["session_id"] = session_id
            result["invocation_time"] = invocation_time
            print(f"    OK: Session={session_id[:12]}...")
        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = f"Invocation failed: {e}"
            print(f"    FAIL: {result['error']}")
            return result
        
        # Phase 3: Wait for completion
        print(f"\n  [3/4] Waiting for end function ({config['end_function']})...")
        completion = self.wait_for_completion(workflow_name, session_id, invocation_time)
        
        if not completion.get("completed"):
            result["status"] = "FAIL"
            result["error"] = completion.get("error", "Unknown completion error")
            print(f"    FAIL: {result['error']}")
            return result
        
        e2e = completion["end_to_end_ms"]
        print(f"    OK: Completed in {e2e}ms (end function: {completion['end_function_duration_ms']:.1f}ms)")
        result["timing"] = completion
        
        # Phase 4: Verify all functions executed
        print(f"\n  [4/4] Verifying all functions executed...")
        # Small delay for log propagation
        time.sleep(3)
        executed = self.check_all_functions_executed(workflow_name, invocation_time)
        
        all_ok = all(executed.values())
        result["functions_executed"] = executed
        
        for fname, ran in executed.items():
            status_icon = "OK" if ran else "MISSING"
            print(f"    {status_icon}: {fname}")
        
        if all_ok:
            result["status"] = "PASS"
            print(f"\n  RESULT: PASS (E2E: {e2e}ms, all {len(executed)} functions executed)")
        else:
            missing = [f for f, r in executed.items() if not r]
            result["status"] = "FAIL"
            result["error"] = f"Functions did not execute: {missing}"
            print(f"\n  RESULT: FAIL - Missing: {missing}")
        
        result["end_time"] = datetime.now(timezone.utc).isoformat()
        return result


def main():
    parser = argparse.ArgumentParser(description="Smoke test benchmark workflows")
    parser.add_argument("workflows", nargs="*", default=list(WORKFLOWS.keys()),
                        help="Workflow names to test (default: all)")
    parser.add_argument("--timeout", type=int, default=90,
                        help="Timeout per workflow in seconds (default: 90)")
    parser.add_argument("--region", default=REGION,
                        help=f"AWS region (default: {REGION})")
    parser.add_argument("--output", "-o", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()
    
    # Validate workflow names
    for wf in args.workflows:
        if wf not in WORKFLOWS:
            print(f"Error: Unknown workflow '{wf}'. Available: {list(WORKFLOWS.keys())}")
            sys.exit(1)
    
    runner = SmokeTestRunner(region=args.region, timeout=args.timeout)
    
    print(f"\n{'#'*60}")
    print(f"  UNUM BENCHMARK SMOKE TEST")
    print(f"  Region: {args.region}")
    print(f"  Timeout: {args.timeout}s per workflow")
    print(f"  Workflows: {', '.join(args.workflows)}")
    print(f"{'#'*60}")
    
    all_results = {}
    pass_count = 0
    fail_count = 0
    
    for wf_name in args.workflows:
        result = runner.run_smoke_test(wf_name)
        all_results[wf_name] = result
        
        if result["status"] == "PASS":
            pass_count += 1
        else:
            fail_count += 1
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    
    for wf_name, result in all_results.items():
        status = result["status"]
        timing = result.get("timing", {})
        e2e = timing.get("end_to_end_ms", "N/A")
        
        if status == "PASS":
            print(f"  PASS  {wf_name:<25} E2E: {e2e}ms")
        else:
            error = result.get("error", "Unknown")
            print(f"  FAIL  {wf_name:<25} {error}")
    
    print(f"\n  Total: {pass_count} passed, {fail_count} failed out of {len(all_results)}")
    print(f"{'='*60}\n")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(__file__), "smoke_test_results.json")
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")
    
    sys.exit(1 if fail_count > 0 else 0)


if __name__ == "__main__":
    main()
