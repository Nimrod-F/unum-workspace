#!/usr/bin/env python3
"""
Metrics Collector for Unum Evaluation Benchmarks
=================================================
Parses CloudWatch logs to extract per-function and per-workflow metrics.

Collects 7 primary metrics:
  1. E2E Latency (ms)
  2. Aggregator Invocation Delay (ms)
  3. DynamoDB I/O Operations (reads, writes)
  4. Total Billed Duration (ms)
  5. Peak Memory (MB)
  6. Cold Start Count
  7. Estimated Cost (USD)
"""

import re
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import boto3

logger = logging.getLogger("metrics_collector")

# ─── Pricing Constants (eu-central-1, as of 2025) ──────────────────────────

PRICING = {
    "lambda_gb_second": 0.0000166667,   # per GB-second
    "lambda_request": 0.0000002,         # per request
    "dynamodb_wcu": 0.00000125,          # per WCU (on-demand)
    "dynamodb_rcu": 0.00000025,          # per RCU (on-demand)
    "s3_put": 0.000005,                  # per PUT
    "s3_get": 0.0000004,                 # per GET
    "step_functions_transition": 0.000025, # per state transition
}


# ─── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class FunctionMetrics:
    """Metrics for a single Lambda function invocation."""
    function_name: str
    request_id: str = ""
    start_timestamp_ms: int = 0
    end_timestamp_ms: int = 0
    duration_ms: float = 0.0
    billed_duration_ms: float = 0.0
    memory_size_mb: int = 128
    max_memory_used_mb: int = 0
    init_duration_ms: float = 0.0  # >0 means cold start
    is_cold_start: bool = False
    dynamo_reads: int = 0
    dynamo_writes: int = 0
    dynamo_wcu: float = 0.0
    dynamo_rcu: float = 0.0
    s3_puts: int = 0
    s3_gets: int = 0
    error: Optional[str] = None


@dataclass 
class WorkflowRunMetrics:
    """Aggregated metrics for one complete workflow invocation."""
    workflow_name: str
    config_label: str
    session_id: str
    iteration: int
    invocation_time_ms: int = 0
    completion_time_ms: int = 0
    
    # Primary metrics
    e2e_latency_ms: float = 0.0
    aggregator_delay_ms: float = 0.0
    total_billed_duration_ms: float = 0.0
    peak_memory_mb: int = 0
    cold_start_count: int = 0
    total_dynamo_reads: int = 0
    total_dynamo_writes: int = 0
    total_s3_puts: int = 0
    total_s3_gets: int = 0
    estimated_cost_usd: float = 0.0
    
    # Per-function breakdown
    function_metrics: List[FunctionMetrics] = field(default_factory=list)
    
    # Status
    success: bool = False
    error: Optional[str] = None
    
    def compute_aggregates(self):
        """Compute aggregate metrics from per-function metrics."""
        if not self.function_metrics:
            return
        
        # E2E latency: first START to last END
        starts = [f.start_timestamp_ms for f in self.function_metrics if f.start_timestamp_ms > 0]
        ends = [f.end_timestamp_ms for f in self.function_metrics if f.end_timestamp_ms > 0]
        
        if starts and ends:
            self.e2e_latency_ms = max(ends) - self.invocation_time_ms
            self.completion_time_ms = max(ends)
        
        # Aggregated metrics
        self.total_billed_duration_ms = sum(f.billed_duration_ms for f in self.function_metrics)
        self.peak_memory_mb = max((f.max_memory_used_mb for f in self.function_metrics), default=0)
        self.cold_start_count = sum(1 for f in self.function_metrics if f.is_cold_start)
        self.total_dynamo_reads = sum(f.dynamo_reads for f in self.function_metrics)
        self.total_dynamo_writes = sum(f.dynamo_writes for f in self.function_metrics)
        self.total_s3_puts = sum(f.s3_puts for f in self.function_metrics)
        self.total_s3_gets = sum(f.s3_gets for f in self.function_metrics)
        
        # Cost calculation using actual per-function memory
        self.estimated_cost_usd = self._calculate_cost()
    
    def _calculate_cost(self) -> float:
        """Calculate estimated cost using actual per-function memory sizes."""
        lambda_compute = 0.0
        for f in self.function_metrics:
            gb_seconds = (f.billed_duration_ms / 1000.0) * (f.memory_size_mb / 1024.0)
            lambda_compute += gb_seconds * PRICING["lambda_gb_second"]
        
        lambda_requests = len(self.function_metrics) * PRICING["lambda_request"]
        
        dynamo_cost = (
            self.total_dynamo_reads * PRICING["dynamodb_rcu"] +
            self.total_dynamo_writes * PRICING["dynamodb_wcu"]
        )
        
        s3_cost = (
            self.total_s3_puts * PRICING["s3_put"] +
            self.total_s3_gets * PRICING["s3_get"]
        )
        
        return lambda_compute + lambda_requests + dynamo_cost + s3_cost
    
    def to_dict(self) -> dict:
        d = {
            "workflow_name": self.workflow_name,
            "config_label": self.config_label,
            "session_id": self.session_id,
            "iteration": self.iteration,
            "invocation_time_ms": self.invocation_time_ms,
            "completion_time_ms": self.completion_time_ms,
            "e2e_latency_ms": self.e2e_latency_ms,
            "aggregator_delay_ms": self.aggregator_delay_ms,
            "total_billed_duration_ms": self.total_billed_duration_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "cold_start_count": self.cold_start_count,
            "total_dynamo_reads": self.total_dynamo_reads,
            "total_dynamo_writes": self.total_dynamo_writes,
            "total_s3_puts": self.total_s3_puts,
            "total_s3_gets": self.total_s3_gets,
            "estimated_cost_usd": self.estimated_cost_usd,
            "success": self.success,
            "error": self.error,
            "function_metrics": [asdict(f) for f in self.function_metrics]
        }
        return d


# ─── CloudWatch Log Parser ─────────────────────────────────────────────────

# Regex patterns for parsing REPORT lines
REPORT_REGEX = re.compile(
    r'REPORT RequestId:\s*([\w-]+)\s+'
    r'Duration:\s*([\d.]+)\s*ms\s+'
    r'Billed Duration:\s*(\d+)\s*ms\s+'
    r'Memory Size:\s*(\d+)\s*MB\s+'
    r'Max Memory Used:\s*(\d+)\s*MB'
    r'(?:\s+Init Duration:\s*([\d.]+)\s*ms)?'
)

# Regex for DynamoDB I/O metrics (custom instrumented lines)
DYNAMO_METRICS_REGEX = re.compile(
    r'\[METRICS\]\s*dynamo_reads=(\d+)\s+dynamo_writes=(\d+)'
    r'(?:\s+wcu=([\d.]+)\s+rcu=([\d.]+))?'
)

# Regex for S3 operations
S3_OPS_REGEX = re.compile(
    r'\[METRICS\]\s*s3_puts=(\d+)\s+s3_gets=(\d+)'
)

# Regex for debug lines showing DynamoDB operations
DYNAMO_DEBUG_REGEX = re.compile(
    r'\[DEBUG\]\s*(?:DynamoDB|dynamodb).*(?:PutItem|GetItem|UpdateItem|Query|Scan)'
)

# Regex for fan-in ready/polling debug lines
FAN_IN_POLL_REGEX = re.compile(
    r'\[DEBUG\]\s*Ready:\s*(\d+)/(\d+)'
)


class MetricsCollector:
    """Collects metrics from CloudWatch logs for Lambda function executions."""
    
    def __init__(self, region: str = "eu-central-1"):
        self.region = region
        self.logs_client = boto3.client("logs", region_name=region)
        self.lambda_client = boto3.client("lambda", region_name=region)
    
    def collect_workflow_metrics(
        self,
        function_arns: Dict[str, str],
        invocation_time_ms: int,
        session_id: str,
        workflow_name: str,
        config_label: str,
        iteration: int,
        aggregator_function: Optional[str] = None,
        timeout_seconds: int = 90,
        end_function: Optional[str] = None,
    ) -> WorkflowRunMetrics:
        """
        Collect metrics for a complete workflow run.
        
        Args:
            function_arns: Dict mapping function names to ARNs
            invocation_time_ms: Timestamp when workflow was invoked (ms since epoch)
            session_id: Workflow session ID
            workflow_name: Name of the workflow
            config_label: Configuration label (e.g., "Unum-Base")
            iteration: Iteration number
            aggregator_function: Name of the fan-in function (for delay metric)
            timeout_seconds: Max wait time
            end_function: Name of the final function to wait for completion
        """
        run = WorkflowRunMetrics(
            workflow_name=workflow_name,
            config_label=config_label,
            session_id=session_id,
            iteration=iteration,
            invocation_time_ms=invocation_time_ms,
        )
        
        # Wait for the end function to complete
        if end_function:
            completed = self._wait_for_function(
                function_arns[end_function], 
                invocation_time_ms, 
                timeout_seconds
            )
            if not completed:
                run.error = f"Timeout waiting for {end_function} after {timeout_seconds}s"
                run.success = False
                return run
        else:
            # Wait a fixed time for all functions
            time.sleep(min(timeout_seconds, 30))
        
        # Small delay for log propagation
        time.sleep(3)
        
        # Collect metrics from each function
        for func_name, func_arn in function_arns.items():
            func_metrics = self._collect_function_metrics(
                func_name, func_arn, invocation_time_ms
            )
            if func_metrics:
                run.function_metrics.extend(func_metrics)
        
        # Compute aggregator delay
        if aggregator_function and aggregator_function in function_arns:
            agg_metrics = [
                f for f in run.function_metrics 
                if f.function_name == aggregator_function
            ]
            if agg_metrics:
                earliest_agg = min(f.start_timestamp_ms for f in agg_metrics if f.start_timestamp_ms > 0)
                run.aggregator_delay_ms = earliest_agg - invocation_time_ms
        
        # Compute aggregate metrics
        run.compute_aggregates()
        run.success = len(run.function_metrics) > 0 and run.error is None
        
        return run
    
    def _wait_for_function(
        self, func_arn: str, start_time_ms: int, timeout_seconds: int
    ) -> bool:
        """Wait for a specific Lambda function to complete."""
        func_name = func_arn.split(":")[-1]
        log_group = f"/aws/lambda/{func_name}"
        
        deadline = time.time() + timeout_seconds
        poll_interval = 2
        
        while time.time() < deadline:
            try:
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=start_time_ms - 1000,
                    filterPattern="REPORT RequestId"
                )
                if response.get("events"):
                    return True
            except self.logs_client.exceptions.ResourceNotFoundException:
                pass
            except Exception as e:
                logger.warning(f"Error polling {func_name}: {e}")
            
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.2, 5)
        
        return False
    
    def _collect_function_metrics(
        self, func_name: str, func_arn: str, start_time_ms: int
    ) -> List[FunctionMetrics]:
        """Collect metrics for a single function from CloudWatch logs."""
        lambda_name = func_arn.split(":")[-1]
        log_group = f"/aws/lambda/{lambda_name}"
        
        results = []
        
        try:
            # Get all log events for this function since invocation
            all_events = self._get_all_log_events(log_group, start_time_ms - 1000)
            
            if not all_events:
                return results
            
            # Group events by request ID
            request_groups = self._group_by_request(all_events)
            
            for request_id, events in request_groups.items():
                fm = FunctionMetrics(function_name=func_name, request_id=request_id)
                
                for event in events:
                    msg = event.get("message", "")
                    ts = event.get("timestamp", 0)
                    
                    # START event
                    if msg.startswith("START RequestId"):
                        fm.start_timestamp_ms = ts
                    
                    # END event
                    elif msg.startswith("END RequestId"):
                        fm.end_timestamp_ms = ts
                    
                    # REPORT event
                    elif "REPORT RequestId" in msg:
                        self._parse_report(fm, msg)
                    
                    # DynamoDB metrics
                    elif "[METRICS]" in msg:
                        self._parse_metrics_line(fm, msg)
                    
                    # DynamoDB debug operations (approximate count)
                    elif "[DEBUG]" in msg:
                        self._parse_debug_dynamo(fm, msg)
                    
                    # Error detection
                    elif "[ERROR]" in msg or "Traceback" in msg:
                        if not fm.error:
                            fm.error = msg.strip()[:200]
                
                results.append(fm)
        
        except self.logs_client.exceptions.ResourceNotFoundException:
            logger.debug(f"No log group for {func_name}")
        except Exception as e:
            logger.warning(f"Error collecting metrics for {func_name}: {e}")
        
        return results
    
    def _get_all_log_events(self, log_group: str, start_time_ms: int) -> List[dict]:
        """Get all log events from a log group since start_time, handling pagination."""
        events = []
        kwargs = {
            "logGroupName": log_group,
            "startTime": start_time_ms,
            "interleaved": True,
        }
        
        while True:
            response = self.logs_client.filter_log_events(**kwargs)
            events.extend(response.get("events", []))
            
            next_token = response.get("nextToken")
            if not next_token:
                break
            kwargs["nextToken"] = next_token
        
        return events
    
    def _group_by_request(self, events: List[dict]) -> Dict[str, List[dict]]:
        """Group log events by Lambda request ID."""
        groups = {}
        current_request = None
        
        for event in events:
            msg = event.get("message", "")
            
            # Extract request ID from START/END/REPORT lines
            req_match = re.search(r'RequestId:\s*([\w-]+)', msg)
            if req_match:
                current_request = req_match.group(1)
            
            if current_request:
                if current_request not in groups:
                    groups[current_request] = []
                groups[current_request].append(event)
        
        return groups
    
    def _parse_report(self, fm: FunctionMetrics, msg: str):
        """Parse a REPORT log line."""
        match = REPORT_REGEX.search(msg)
        if match:
            fm.request_id = match.group(1)
            fm.duration_ms = float(match.group(2))
            fm.billed_duration_ms = float(match.group(3))
            fm.memory_size_mb = int(match.group(4))
            fm.max_memory_used_mb = int(match.group(5))
            if match.group(6):
                fm.init_duration_ms = float(match.group(6))
                fm.is_cold_start = True
    
    def _parse_metrics_line(self, fm: FunctionMetrics, msg: str):
        """Parse a [METRICS] line for DynamoDB/S3 I/O counts."""
        dm = DYNAMO_METRICS_REGEX.search(msg)
        if dm:
            fm.dynamo_reads = int(dm.group(1))
            fm.dynamo_writes = int(dm.group(2))
            if dm.group(3):
                fm.dynamo_wcu = float(dm.group(3))
            if dm.group(4):
                fm.dynamo_rcu = float(dm.group(4))
        
        sm = S3_OPS_REGEX.search(msg)
        if sm:
            fm.s3_puts = int(sm.group(1))
            fm.s3_gets = int(sm.group(2))
    
    def _parse_debug_dynamo(self, fm: FunctionMetrics, msg: str):
        """Approximate DynamoDB operations from debug log lines."""
        # Count checkpoint writes (PutItem)
        if "PutItem" in msg or "put_item" in msg or "checkpoint" in msg.lower():
            fm.dynamo_writes += 1
        
        # Count checkpoint reads (GetItem)
        if "GetItem" in msg or "get_item" in msg or "get_checkpoint" in msg.lower():
            fm.dynamo_reads += 1
        
        # Count fan-in polls
        if FAN_IN_POLL_REGEX.search(msg):
            fm.dynamo_reads += 1  # Each poll is a DynamoDB read
    
    def force_cold_starts(self, function_arns: Dict[str, str]):
        """Force cold starts by updating Lambda env vars.
        
        After updating a Lambda configuration, AWS marks existing execution
        environments as stale. The NEXT invocation creates a fresh container
        (cold start). We must wait for the update to fully propagate before
        invoking.
        """
        import uuid
        
        for func_name, func_arn in function_arns.items():
            try:
                response = self.lambda_client.get_function_configuration(
                    FunctionName=func_arn
                )
                env = response.get("Environment", {}).get("Variables", {})
                env["COLD_START_TRIGGER"] = str(uuid.uuid4())
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={"Variables": env}
                )
                logger.debug(f"  Updated COLD_START_TRIGGER for {func_name}")
            except Exception as e:
                logger.warning(f"Could not force cold start for {func_name}: {e}")
        
        # Give AWS time to START processing the updates before polling
        time.sleep(3)
        
        # Wait for all functions to report Successful update
        for func_name, func_arn in function_arns.items():
            self._wait_function_update_complete(func_arn, func_name)
    
    def _wait_function_update_complete(self, func_arn: str, func_name: str = "", timeout: int = 60):
        """Wait until a Lambda function's configuration update is fully complete.
        
        This is critical for cold-start forcing: we must wait for
        LastUpdateStatus == 'Successful', NOT accept empty string (which
        means the status hasn't changed yet — race condition).
        """
        deadline = time.time() + timeout
        saw_in_progress = False
        
        while time.time() < deadline:
            try:
                resp = self.lambda_client.get_function(FunctionName=func_arn)
                state = resp["Configuration"].get("State", "")
                update_status = resp["Configuration"].get("LastUpdateStatus", "")
                
                if update_status == "InProgress":
                    saw_in_progress = True
                
                # Only return when we see Successful AND either:
                # - We previously saw InProgress (confirms update was processed)
                # - We've waited at least 5 seconds (for very fast updates)
                if state == "Active" and update_status == "Successful":
                    if saw_in_progress or (deadline - time.time()) < (timeout - 5):
                        return
                    
                if update_status == "Failed":
                    logger.warning(f"  Config update failed for {func_name}")
                    return
                    
            except Exception:
                pass
            time.sleep(1)
        
        logger.warning(f"  Timeout waiting for {func_name} update to complete")


def calculate_step_functions_cost(num_transitions: int, num_invocations: int = 1) -> float:
    """Calculate Step Functions cost for comparison."""
    return num_transitions * num_invocations * PRICING["step_functions_transition"]
