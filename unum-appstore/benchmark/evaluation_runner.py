#!/usr/bin/env python3
"""
Comprehensive Evaluation Benchmark Runner
==========================================
Runs all 5 workflows across 8 configurations, collecting metrics for academic
paper evaluation section.

Configurations:
  1. SF         — AWS Step Functions baseline
  2. Unum-Base  — Classic fan-in, no fusion, no streaming
  3. Unum-Fus   — Intelligent fusion only
  4. Unum-Str   — Partial parameter streaming only
  5. Unum-Fut   — Future-based execution only
  6. Unum-Fut+Fus — Future-based + fusion
  7. Unum-Fut+Str — Future-based + streaming
  8. Unum-All   — Future-based + fusion + streaming

Metrics collected (per workflow run):
  - E2E Latency (ms)
  - Aggregator Invocation Delay (ms)
  - DynamoDB I/O (reads, writes)
  - Total Billed Duration (ms)
  - Peak Memory (MB)
  - Cold Start Count
  - Estimated Cost (USD)

Usage:
    python evaluation_runner.py                      # Full benchmark (all workflows, all configs)
    python evaluation_runner.py --workflows nlp-pipeline graph-analysis
    python evaluation_runner.py --configs Unum-Base Unum-Fut
    python evaluation_runner.py --iterations 5 --warmup 2  # Quick test
    python evaluation_runner.py --skip-deploy        # Use current deployment (no config switching)
    python evaluation_runner.py --resume results/run_20250610_143000  # Resume from checkpoint
"""

import json
import time
import uuid
import sys
import os
import copy
import argparse
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import boto3
import yaml

from metrics_collector import (
    MetricsCollector, 
    WorkflowRunMetrics, 
    FunctionMetrics,
    PRICING,
    calculate_step_functions_cost,
)

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluation_runner")


# ─── Constants ──────────────────────────────────────────────────────────────

REGION = "eu-central-1"
WORKSPACE = Path(__file__).resolve().parent.parent  # unum-appstore/
UNUM_CLI = Path(__file__).resolve().parent.parent.parent / "unum" / "unum-cli" / "unum-cli.py"
UNUM_RUNTIME = Path(__file__).resolve().parent.parent.parent / "unum" / "runtime"

DEFAULT_ITERATIONS = 10   # Total iterations per config
DEFAULT_COLD_RUNS = 3     # Of those, how many force cold starts
DEFAULT_WARMUP = 2        # Warm-up runs (discarded)
INTER_RUN_DELAY = 2.0     # Seconds between invocations
COLD_START_DELAY = 15.0   # Extra wait after forcing cold start (must be long enough for container recycling)


# ─── Workflow Definitions ───────────────────────────────────────────────────

@dataclass
class WorkflowConfig:
    """Full definition of a benchmark workflow."""
    name: str
    directory: str
    start_functions: List[str]
    end_function: str
    aggregator_function: Optional[str]  # For aggregator delay metric
    all_functions: List[str]
    topology: str
    parallel_start: bool = False
    test_payload: Any = None
    sf_transition_count: int = 0  # For Step Functions cost estimation
    fusable: bool = False   # Whether fusion is applicable
    streamable: bool = True  # Whether streaming is applicable
    fusion_config: Optional[str] = None  # Path to fusion.yaml relative to workflow dir
    description: str = ""

WORKFLOWS: Dict[str, WorkflowConfig] = {
    "nlp-pipeline": WorkflowConfig(
        name="nlp-pipeline",
        directory="nlp-pipeline",
        start_functions=["Tokenizer"],
        end_function="Summarizer",
        aggregator_function=None,  # No fan-in
        all_functions=["Tokenizer", "Analyzer", "Classifier", "Summarizer"],
        topology="chain",
        test_payload={
            "text": "Artificial intelligence and machine learning are transforming "
                    "cloud computing. Serverless architectures enable cost-effective "
                    "scaling. Natural language processing allows computers to understand "
                    "human language patterns efficiently.",
            "doc_id": "bench-001"
        },
        sf_transition_count=5,  # Start + 4 Task states
        fusable=True,  # Chain: Tokenizer→Analyzer→Classifier→Summarizer
        streamable=True,
        fusion_config="fusion.yaml",
        description="Chain: 4 sequential NLP functions"
    ),
    "text-processing": WorkflowConfig(
        name="text-processing",
        directory="text-processing",
        start_functions=["UserMention", "FindUrl"],
        end_function="Publish",
        aggregator_function="CreatePost",
        all_functions=["UserMention", "FindUrl", "ShortenUrl", "CreatePost", "Publish"],
        topology="parallel-fanin",
        parallel_start=True,
        test_payload="ABC's report on cloud computing trends @CloudExpert "
                     "https://example.com/report shows serverless adoption growing 40% year over year",
        sf_transition_count=8,  # Start + Parallel(2 branches) + Task*5
        fusable=False,  # Parallel branches can't be fused
        streamable=True,
        description="Parallel+Fan-in: 5 functions, 2 branches"
    ),
    "graph-analysis": WorkflowConfig(
        name="graph-analysis",
        directory="graph-analysis",
        start_functions=["GraphGenerator"],
        end_function="Aggregator",
        aggregator_function="Aggregator",
        all_functions=["GraphGenerator", "PageRank", "BFS", "MST", "Aggregator"],
        topology="fan-out-fan-in",
        test_payload={"size": 50, "seed": 42},
        sf_transition_count=7,  # Start + Task + Parallel(3 branches) + Aggregator
        fusable=False,  # Fan-out branches
        streamable=True,
        description="Fan-out/Fan-in: 3-way parallel graph algorithms"
    ),
    "montecarlo-pipeline": WorkflowConfig(
        name="montecarlo-pipeline",
        directory="montecarlo-pipeline",
        start_functions=["DataGenerator"],
        end_function="Reporter",
        aggregator_function="Aggregator",
        all_functions=["DataGenerator", "Transform", "Estimate", "Validate", "Simulate", "Aggregator", "Reporter"],
        topology="diamond",
        test_payload={
            "matrix_size": 50,
            "n_simulations": 10000,
            "n_walks": 20,
            "seed": 42
        },
        sf_transition_count=10,  # Start + Task + Parallel(2 branches: chain-of-3 + single) + Agg + Reporter
        fusable=True,  # Transform→Estimate→Validate chain
        streamable=True,
        fusion_config="fusion.yaml",
        description="Diamond: 7 functions, chain+parallel hybrid"
    ),
    "wordcount": WorkflowConfig(
        name="wordcount",
        directory="wordcount",
        start_functions=["UnumMap0"],
        end_function="Summary",
        aggregator_function="Partition",  # First fan-in point
        all_functions=["UnumMap0", "Mapper", "Partition", "Reducer", "Summary"],
        topology="mapreduce",
        test_payload=[
            {"text": "hello world hello serverless world", "destination": "wordcount-benchmark-683003725669"},
            {"text": "world cloud function lambda serverless", "destination": "wordcount-benchmark-683003725669"},
            {"text": "hello cloud hello lambda hello", "destination": "wordcount-benchmark-683003725669"}
        ],
        sf_transition_count=12,  # Start + Map(3) + Partition + Map(3 reducers) + Summary
        fusable=False,  # MapReduce with dynamic fan-out
        streamable=True,
        description="MapReduce: dynamic fan-out/fan-in, 2 aggregation points"
    ),
}


# ─── Configuration Matrix ──────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """A single benchmark configuration."""
    label: str
    description: str
    eager: bool = False
    future_based: bool = False
    streaming: bool = False
    fusion: bool = False
    step_functions: bool = False
    
    @property
    def requires_rebuild(self) -> bool:
        """Whether this config requires a full SAM rebuild."""
        return self.fusion or self.streaming
    
    @property
    def env_changes(self) -> Dict[str, str]:
        """Environment variable overrides for this config."""
        env = {}
        if self.future_based:
            env["UNUM_FUTURE_BASED"] = "true"
        if self.streaming:
            env["UNUM_STREAMING"] = "true"
        return env


CONFIGURATIONS: Dict[str, BenchmarkConfig] = {
    "SF": BenchmarkConfig(
        label="SF",
        description="AWS Step Functions baseline",
        step_functions=True,
    ),
    "Unum-Base": BenchmarkConfig(
        label="Unum-Base",
        description="Classic Unum (no enhancements)",
        eager=False,
        future_based=False,
    ),
    "Unum-Fus": BenchmarkConfig(
        label="Unum-Fus",
        description="Intelligent Fusion only",
        eager=False,
        future_based=False,
        fusion=True,
    ),
    "Unum-Str": BenchmarkConfig(
        label="Unum-Str",
        description="Partial Parameter Streaming only",
        eager=False,
        future_based=False,
        streaming=True,
    ),
    "Unum-Fut": BenchmarkConfig(
        label="Unum-Fut",
        description="Future-based Execution only",
        eager=True,
        future_based=True,
    ),
    "Unum-Fut+Fus": BenchmarkConfig(
        label="Unum-Fut+Fus",
        description="Future-based + Fusion",
        eager=True,
        future_based=True,
        fusion=True,
    ),
    "Unum-Fut+Str": BenchmarkConfig(
        label="Unum-Fut+Str",
        description="Future-based + Streaming",
        eager=True,
        future_based=True,
        streaming=True,
    ),
    "Unum-All": BenchmarkConfig(
        label="Unum-All",
        description="All enhancements combined",
        eager=True,
        future_based=True,
        fusion=True,
        streaming=True,
    ),
}


# ─── Evaluation Runner ─────────────────────────────────────────────────────

class EvaluationRunner:
    """Main benchmark orchestrator."""
    
    def __init__(
        self,
        region: str = REGION,
        iterations: int = DEFAULT_ITERATIONS,
        cold_runs: int = DEFAULT_COLD_RUNS,
        warmup: int = DEFAULT_WARMUP,
        output_dir: Optional[str] = None,
        skip_deploy: bool = False,
        timeout: int = 120,
    ):
        self.region = region
        self.iterations = iterations
        self.cold_runs = cold_runs
        self.warmup = warmup
        self.skip_deploy = skip_deploy
        self.timeout = timeout
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(__file__).parent / "results" / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # AWS clients
        self.lambda_client = boto3.client("lambda", region_name=region)
        self.logs_client = boto3.client("logs", region_name=region)
        self.sfn_client = boto3.client("stepfunctions", region_name=region)
        
        # Metrics collector
        self.collector = MetricsCollector(region=region)
        
        # Results storage
        self.all_results: Dict[str, Dict[str, List[WorkflowRunMetrics]]] = {}
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Iterations: {iterations} ({cold_runs} cold + {iterations - cold_runs} warm)")
        logger.info(f"Warm-up runs: {warmup} (discarded)")
    
    # ─── Main Entry Point ───────────────────────────────────────────────
    
    def run_full_evaluation(
        self,
        workflow_names: Optional[List[str]] = None,
        config_labels: Optional[List[str]] = None,
    ):
        """Run the complete evaluation across all workflows and configurations."""
        workflows = workflow_names or list(WORKFLOWS.keys())
        configs = config_labels or list(CONFIGURATIONS.keys())
        
        total_runs = len(workflows) * len(configs) * (self.warmup + self.iterations)
        logger.info(f"=" * 70)
        logger.info(f"COMPREHENSIVE EVALUATION BENCHMARK")
        logger.info(f"  Workflows: {workflows}")
        logger.info(f"  Configurations: {configs}")
        logger.info(f"  Total runs: {total_runs}")
        logger.info(f"=" * 70)
        
        # Save experiment metadata
        self._save_metadata(workflows, configs)
        
        # Load already-completed checkpoints (resume support)
        completed = set()
        for f in self.output_dir.glob("*__*.json"):
            parts = f.stem.split("__")
            if len(parts) == 2:
                completed.add((parts[0], parts[1]))
        if completed:
            logger.info(f"  RESUME: Found {len(completed)} completed checkpoints, will skip them")
            for wf, cfg in sorted(completed):
                logger.info(f"    Already done: {wf} / {cfg}")
        
        for wf_name in workflows:
            workflow = WORKFLOWS[wf_name]
            logger.info(f"\n{'#' * 60}")
            logger.info(f"  WORKFLOW: {wf_name} ({workflow.description})")
            logger.info(f"{'#' * 60}")
            
            self.all_results[wf_name] = {}
            
            for cfg_label in configs:
                config = CONFIGURATIONS[cfg_label]
                
                # Skip already-completed combos (resume)
                if (wf_name, cfg_label) in completed:
                    logger.info(f"\n  [{cfg_label}] Already completed - skipping (resume)")
                    # Load existing results into memory for final aggregation
                    ckpt_file = self.output_dir / f"{wf_name}__{cfg_label}.json"
                    try:
                        with open(ckpt_file) as fh:
                            existing = json.load(fh)
                        self.all_results[wf_name][cfg_label] = existing
                    except Exception as e:
                        logger.warning(f"  Could not load checkpoint {ckpt_file}: {e}")
                    continue
                
                # Skip Step Functions for now (requires separate state machine setup)
                if config.step_functions:
                    logger.info(f"\n  [{cfg_label}] Skipping Step Functions (not yet deployed)")
                    continue
                
                # Skip fusion configs for non-fusable workflows
                if config.fusion and not workflow.fusable:
                    logger.info(f"\n  [{cfg_label}] Skipping (fusion N/A for {wf_name})")
                    continue
                
                logger.info(f"\n  {'─' * 50}")
                logger.info(f"  CONFIG: {cfg_label} — {config.description}")
                logger.info(f"  {'─' * 50}")
                
                # Phase 1: Deploy this configuration (if needed)
                if not self.skip_deploy:
                    self._deploy_configuration(workflow, config)
                else:
                    # Just update env vars if possible
                    self._apply_env_vars(workflow, config)
                
                # Phase 2: Cold start runs (BEFORE warm-up — no point
                #          warming containers that cold-start forcing will destroy)
                cold_results = []
                if self.cold_runs > 0:
                    logger.info(f"  Running {self.cold_runs} cold-start iterations...")
                    for i in range(self.cold_runs):
                        # Force cold starts
                        self._force_cold_starts(workflow)
                        time.sleep(COLD_START_DELAY)
                        
                        result = self._run_single_iteration(
                            workflow, config, iteration=i, is_cold=True
                        )
                        cold_results.append(result)
                        self._log_iteration_summary(result, f"cold-{i}")
                        time.sleep(INTER_RUN_DELAY)
                
                # Phase 3: Warm-up runs (re-warm containers after cold starts, discarded)
                if self.warmup > 0:
                    logger.info(f"  Running {self.warmup} warm-up iterations (discarded)...")
                    for i in range(self.warmup):
                        self._run_single_iteration(workflow, config, iteration=-1, is_warmup=True)
                        time.sleep(INTER_RUN_DELAY)
                
                # Phase 4: Warm runs (measured)
                warm_results = []
                warm_count = self.iterations - self.cold_runs
                logger.info(f"  Running {warm_count} warm iterations...")
                for i in range(warm_count):
                    result = self._run_single_iteration(
                        workflow, config, iteration=self.cold_runs + i, is_cold=False
                    )
                    warm_results.append(result)
                    self._log_iteration_summary(result, f"warm-{i}")
                    time.sleep(INTER_RUN_DELAY)
                
                # Store results
                all_run_results = cold_results + warm_results
                self.all_results[wf_name][cfg_label] = all_run_results
                
                # Save checkpoint after each config
                self._save_config_results(wf_name, cfg_label, all_run_results)
                self._print_config_summary(wf_name, cfg_label, all_run_results)
        
        # Save final aggregated results
        self._save_final_results()
        logger.info(f"\n{'=' * 70}")
        logger.info(f"  EVALUATION COMPLETE — Results in {self.output_dir}")
        logger.info(f"{'=' * 70}")
    
    # ─── Single Iteration ──────────────────────────────────────────────
    
    def _run_single_iteration(
        self,
        workflow: WorkflowConfig,
        config: BenchmarkConfig,
        iteration: int,
        is_warmup: bool = False,
        is_cold: bool = False,
    ) -> WorkflowRunMetrics:
        """Run a single workflow invocation and collect metrics."""
        session_id = str(uuid.uuid4())
        arns = self._load_function_arns(workflow)
        
        # Record invocation time
        invocation_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        try:
            # Invoke the workflow
            self._invoke_workflow(workflow, arns, session_id)
            
            # Collect metrics via CloudWatch
            result = self.collector.collect_workflow_metrics(
                function_arns=arns,
                invocation_time_ms=invocation_time_ms,
                session_id=session_id,
                workflow_name=workflow.name,
                config_label=config.label,
                iteration=iteration,
                aggregator_function=workflow.aggregator_function,
                timeout_seconds=self.timeout,
                end_function=workflow.end_function,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"  Error in iteration {iteration}: {e}")
            return WorkflowRunMetrics(
                workflow_name=workflow.name,
                config_label=config.label,
                session_id=session_id,
                iteration=iteration,
                invocation_time_ms=invocation_time_ms,
                success=False,
                error=str(e),
            )
    
    # ─── Workflow Invocation ────────────────────────────────────────────
    
    def _invoke_workflow(self, workflow: WorkflowConfig, arns: Dict[str, str], session_id: str):
        """Invoke a workflow handling different topology patterns."""
        if workflow.parallel_start:
            self._invoke_parallel_starts(workflow, arns, session_id)
        else:
            start_func = workflow.start_functions[0]
            start_arn = arns[start_func]
            payload = self._build_payload(workflow.test_payload, session_id)
            
            self.lambda_client.invoke(
                FunctionName=start_arn,
                InvocationType='Event',
                Payload=json.dumps(payload)
            )
    
    def _invoke_parallel_starts(self, workflow: WorkflowConfig, arns: Dict[str, str], session_id: str):
        """Invoke multiple start functions in parallel (text-processing pattern)."""
        threads = []
        for idx, func_name in enumerate(workflow.start_functions):
            payload = {
                "Session": session_id,
                "Fan-out": {"Index": idx},
                "Data": {"Source": "http", "Value": workflow.test_payload}
            }
            func_arn = arns[func_name]
            t = threading.Thread(
                target=self._invoke_async,
                args=(func_arn, payload)
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    
    def _invoke_async(self, function_arn: str, payload: dict):
        """Invoke a Lambda function asynchronously."""
        self.lambda_client.invoke(
            FunctionName=function_arn,
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
    
    def _build_payload(self, test_data: Any, session_id: str) -> dict:
        """Build unum invocation payload."""
        return {
            "Data": {"Source": "http", "Value": test_data}
        }
    
    # ─── Configuration Management ──────────────────────────────────────
    
    def _deploy_configuration(self, workflow: WorkflowConfig, config: BenchmarkConfig):
        """Deploy a specific configuration variant of a workflow."""
        wf_dir = WORKSPACE / workflow.directory
        
        if config.requires_rebuild:
            logger.info(f"  Rebuilding {workflow.name} for {config.label}...")
            self._rebuild_and_deploy(workflow, config)
        else:
            # Just update environment variables on existing functions
            self._apply_env_vars(workflow, config)
    
    def _apply_env_vars(self, workflow: WorkflowConfig, config: BenchmarkConfig):
        """Update Lambda environment variables for a configuration."""
        arns = self._load_function_arns(workflow)
        
        for func_name, func_arn in arns.items():
            try:
                # Get current configuration
                response = self.lambda_client.get_function_configuration(
                    FunctionName=func_arn
                )
                env = response.get("Environment", {}).get("Variables", {})
                
                # Apply config-specific env vars
                # Eager mode (controls whether first branch invokes aggregator)
                # This is a template-level setting, but we approximate via env vars
                if config.eager:
                    env["EAGER"] = "true"
                else:
                    env.pop("EAGER", None)
                
                # Future-based execution
                if config.future_based:
                    env["UNUM_FUTURE_BASED"] = "true"
                else:
                    env.pop("UNUM_FUTURE_BASED", None)
                
                # Update the function
                self.lambda_client.update_function_configuration(
                    FunctionName=func_arn,
                    Environment={"Variables": env}
                )
                
            except Exception as e:
                logger.warning(f"  Could not update env for {func_name}: {e}")
        
        # Wait for all functions to become active
        time.sleep(3)
        for func_name, func_arn in arns.items():
            self.collector._wait_function_update_complete(func_arn, func_name, timeout=30)
        
        logger.info(f"  Environment variables updated for {config.label}")
    
    def _rebuild_and_deploy(self, workflow: WorkflowConfig, config: BenchmarkConfig):
        """Full rebuild and deploy cycle (for fusion/streaming configs)."""
        wf_dir = WORKSPACE / workflow.directory
        
        # Step 1: If fusion, run unum-cli fuse first
        if config.fusion and workflow.fusion_config:
            logger.info(f"  Running fusion compilation...")
            cmd = [
                sys.executable, str(UNUM_CLI), "fuse",
                "-c", workflow.fusion_config,
                "-t", "unum-template.yaml",
            ]
            self._run_command(cmd, cwd=str(wf_dir))
            # After fusion, the template is unum-template-fused.yaml
            # and build directory is fused_build/
        
        # Step 2: Build with optional streaming
        build_cmd = [sys.executable, str(UNUM_CLI), "build"]
        if config.streaming:
            build_cmd.append("--streaming")
        
        logger.info(f"  Running SAM build... (streaming={config.streaming})")
        self._run_command(build_cmd, cwd=str(wf_dir))
        
        # Step 3: Deploy
        logger.info(f"  Deploying...")
        deploy_cmd = [sys.executable, str(UNUM_CLI), "deploy"]
        self._run_command(deploy_cmd, cwd=str(wf_dir))
        
        # Step 4: Apply env vars (future-based needs env update after deploy)
        time.sleep(5)
        self._apply_env_vars(workflow, config)
        
        logger.info(f"  Deploy complete for {config.label}")
    
    def _run_command(self, cmd: List[str], cwd: str) -> str:
        """Run a subprocess command and return output."""
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,  # 10 minute timeout for builds
        )
        
        if result.returncode != 0:
            logger.error(f"  Command failed: {' '.join(cmd)}")
            logger.error(f"  stderr: {result.stderr[:500]}")
            raise RuntimeError(f"Command failed: {result.stderr[:200]}")
        
        return result.stdout
    
    # ─── Cold Start Management ─────────────────────────────────────────
    
    def _force_cold_starts(self, workflow: WorkflowConfig):
        """Force cold starts on all functions by updating env vars."""
        arns = self._load_function_arns(workflow)
        self.collector.force_cold_starts(arns)
        logger.debug(f"  Forced cold starts for all {len(arns)} functions")
    
    # ─── Utility ───────────────────────────────────────────────────────
    
    def _load_function_arns(self, workflow: WorkflowConfig) -> Dict[str, str]:
        """Load function ARNs from function-arn.yaml."""
        arn_file = WORKSPACE / workflow.directory / "function-arn.yaml"
        with open(arn_file) as f:
            return yaml.safe_load(f)
    
    def _log_iteration_summary(self, result: WorkflowRunMetrics, label: str):
        """Log a brief summary of one iteration."""
        if result.success:
            logger.info(
                f"    [{label}] E2E={result.e2e_latency_ms:.0f}ms "
                f"Billed={result.total_billed_duration_ms:.0f}ms "
                f"Mem={result.peak_memory_mb}MB "
                f"Cold={result.cold_start_count} "
                f"DDB_R={result.total_dynamo_reads} DDB_W={result.total_dynamo_writes} "
                f"Cost=${result.estimated_cost_usd:.8f}"
            )
        else:
            logger.warning(f"    [{label}] FAILED: {result.error}")
    
    def _print_config_summary(self, wf_name: str, cfg_label: str, results: List[WorkflowRunMetrics]):
        """Print summary statistics after completing all runs for one config."""
        successful = [r for r in results if r.success]
        if not successful:
            logger.warning(f"  No successful runs for {wf_name}/{cfg_label}")
            return
        
        latencies = [r.e2e_latency_ms for r in successful]
        billed = [r.total_billed_duration_ms for r in successful]
        costs = [r.estimated_cost_usd for r in successful]
        
        latencies.sort()
        n = len(latencies)
        median_lat = latencies[n // 2]
        p95_lat = latencies[int(n * 0.95)] if n >= 20 else latencies[-1]
        mean_cost = sum(costs) / len(costs)
        
        logger.info(f"\n  ┌─ SUMMARY: {wf_name} / {cfg_label} ─────────────────")
        logger.info(f"  │ Successful runs: {len(successful)}/{len(results)}")
        logger.info(f"  │ E2E Latency:  median={median_lat:.0f}ms  P95={p95_lat:.0f}ms")
        logger.info(f"  │ Billed Duration: median={sorted(billed)[n//2]:.0f}ms")
        logger.info(f"  │ Mean Cost: ${mean_cost:.8f}")
        logger.info(f"  └──────────────────────────────────────────────")
    
    # ─── Results Persistence ───────────────────────────────────────────
    
    def _save_metadata(self, workflows: List[str], configs: List[str]):
        """Save experiment metadata."""
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "region": self.region,
            "iterations": self.iterations,
            "cold_runs": self.cold_runs,
            "warmup": self.warmup,
            "workflows": workflows,
            "configurations": configs,
            "workflow_details": {
                name: {
                    "topology": w.topology,
                    "functions": w.all_functions,
                    "fusable": w.fusable,
                    "streamable": w.streamable,
                    "sf_transitions": w.sf_transition_count,
                    "description": w.description,
                }
                for name, w in WORKFLOWS.items() if name in workflows
            },
            "config_details": {
                label: {
                    "description": c.description,
                    "eager": c.eager,
                    "future_based": c.future_based,
                    "streaming": c.streaming,
                    "fusion": c.fusion,
                    "step_functions": c.step_functions,
                }
                for label, c in CONFIGURATIONS.items() if label in configs
            },
            "pricing": PRICING,
        }
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _save_config_results(self, wf_name: str, cfg_label: str, results: List[WorkflowRunMetrics]):
        """Save results for one workflow/config pair (checkpoint)."""
        filename = f"{wf_name}__{cfg_label}.json"
        data = [r.to_dict() for r in results]
        
        with open(self.output_dir / filename, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"  Checkpoint saved: {filename}")
    
    def _save_final_results(self):
        """Save the final aggregated results."""
        final = {}
        for wf_name, configs in self.all_results.items():
            final[wf_name] = {}
            for cfg_label, results in configs.items():
                # Handle both WorkflowRunMetrics objects (new) and dicts (resumed)
                serialized = []
                for r in results:
                    if isinstance(r, dict):
                        serialized.append(r)
                    else:
                        serialized.append(r.to_dict())
                final[wf_name][cfg_label] = serialized
        
        with open(self.output_dir / "all_results.json", "w") as f:
            json.dump(final, f, indent=2, default=str)
        
        logger.info(f"Final results saved to {self.output_dir / 'all_results.json'}")
    
    def load_results(self, results_dir: str) -> Dict[str, Dict[str, List[dict]]]:
        """Load results from a previous run for analysis."""
        results_path = Path(results_dir) / "all_results.json"
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
        
        # Try loading individual checkpoint files
        results = {}
        for f in Path(results_dir).glob("*__*.json"):
            parts = f.stem.split("__")
            if len(parts) == 2:
                wf_name, cfg_label = parts
                if wf_name not in results:
                    results[wf_name] = {}
                with open(f) as fh:
                    results[wf_name][cfg_label] = json.load(fh)
        
        return results


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation_runner.py                         # Full benchmark
  python evaluation_runner.py --iterations 5 --warmup 1  # Quick test (5 runs)
  python evaluation_runner.py --workflows nlp-pipeline graph-analysis
  python evaluation_runner.py --configs Unum-Base Unum-Fut
  python evaluation_runner.py --skip-deploy           # Skip config changes
  python evaluation_runner.py --resume results/run_20250610_143000
        """
    )
    
    parser.add_argument(
        "--workflows", nargs="+", 
        choices=list(WORKFLOWS.keys()),
        default=None,
        help="Workflow(s) to benchmark (default: all)"
    )
    parser.add_argument(
        "--configs", nargs="+",
        choices=list(CONFIGURATIONS.keys()),
        default=None,
        help="Configuration(s) to test (default: all)"
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help=f"Total iterations per config (default: {DEFAULT_ITERATIONS})"
    )
    parser.add_argument(
        "--cold-runs", type=int, default=DEFAULT_COLD_RUNS,
        help=f"Cold-start iterations (default: {DEFAULT_COLD_RUNS})"
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Warm-up iterations to discard (default: {DEFAULT_WARMUP})"
    )
    parser.add_argument(
        "--skip-deploy", action="store_true",
        help="Skip deployment/config changes, use current state"
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout per invocation in seconds (default: 120)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from a previous result directory"
    )
    parser.add_argument(
        "--region", default=REGION,
        help=f"AWS region (default: {REGION})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = EvaluationRunner(
        region=args.region,
        iterations=args.iterations,
        cold_runs=args.cold_runs,
        warmup=args.warmup,
        output_dir=args.output or args.resume,
        skip_deploy=args.skip_deploy,
        timeout=args.timeout,
    )
    
    runner.run_full_evaluation(
        workflow_names=args.workflows,
        config_labels=args.configs,
    )


if __name__ == "__main__":
    main()
