#!/usr/bin/env python3
"""
Benchmark: Early Invocation Optimization for Chain Workflow

Tests the EARLY_INVOKE optimization on the Hello→World chain workflow.
Compares E2E latency with and without the optimization.
"""

import boto3
import json
import time
import uuid
import statistics
from datetime import datetime, timezone
from typing import List, Dict, Tuple

# Configuration
REGION = 'eu-central-1'
PROFILE = 'research-profile'
STACK_NAME = 'unum-hello-world'

class EarlyInvokeBenchmark:
    def __init__(self, profile: str = PROFILE, region: str = REGION):
        self.session = boto3.Session(profile_name=profile, region_name=region)
        self.lambda_client = self.session.client('lambda')
        self.logs_client = self.session.client('logs')
        self.cf_client = self.session.client('cloudformation')
        
        self.functions = self._get_function_names()
        print(f"Found functions: {list(self.functions.keys())}")
    
    def _get_function_names(self) -> Dict[str, str]:
        """Get Lambda function names from CloudFormation"""
        try:
            response = self.cf_client.describe_stack_resources(StackName=STACK_NAME)
            functions = {}
            for resource in response['StackResources']:
                if resource['ResourceType'] == 'AWS::Lambda::Function':
                    logical_id = resource['LogicalResourceId']
                    physical_id = resource['PhysicalResourceId']
                    if 'Hello' in logical_id:
                        functions['Hello'] = physical_id
                    elif 'World' in logical_id:
                        functions['World'] = physical_id
            return functions
        except Exception as e:
            print(f"Error getting functions: {e}")
            return {}
    
    def set_early_invoke(self, enabled: bool):
        """Enable or disable EARLY_INVOKE for all functions"""
        value = "true" if enabled else "false"
        print(f"  Setting EARLY_INVOKE={value} for all functions...")
        
        for name, func_name in self.functions.items():
            try:
                response = self.lambda_client.get_function_configuration(FunctionName=func_name)
                env_vars = response.get('Environment', {}).get('Variables', {})
                env_vars['EARLY_INVOKE'] = value
                
                self.lambda_client.update_function_configuration(
                    FunctionName=func_name,
                    Environment={'Variables': env_vars}
                )
            except Exception as e:
                print(f"    Warning: Could not update {name}: {e}")
        
        # Wait for Lambda to accept the update
        print(f"  Waiting for configuration update...")
        time.sleep(3)
        
        # Force cold starts by updating memory (this recycles all instances)
        # We temporarily change memory and change it back to force redeployment
        print(f"  Forcing cold starts to apply new config...")
        for name, func_name in self.functions.items():
            try:
                # Get current memory
                response = self.lambda_client.get_function_configuration(FunctionName=func_name)
                current_memory = response.get('MemorySize', 128)
                
                # Bump memory up and back down to force recycle
                temp_memory = current_memory + 64 if current_memory < 10176 else current_memory - 64
                self.lambda_client.update_function_configuration(
                    FunctionName=func_name,
                    MemorySize=temp_memory
                )
                time.sleep(1)
                self.lambda_client.update_function_configuration(
                    FunctionName=func_name,
                    MemorySize=current_memory
                )
            except Exception as e:
                print(f"    Warning: Could not force cold start for {name}: {e}")
        
        print(f"  Waiting for all updates to stabilize...")
        time.sleep(5)  # Wait for configuration to propagate
    
    def invoke_workflow(self) -> Tuple[str, int]:
        """Invoke the Hello function and return session ID and timestamp"""
        session_id = str(uuid.uuid4())
        
        payload = {
            "Session": session_id,
            "Data": {
                "Source": "http",
                "Value": {"message": "Hello World Test"}
            }
        }
        
        start_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        self.lambda_client.invoke(
            FunctionName=self.functions['Hello'],
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
        
        return session_id, start_time
    
    def wait_for_completion(self, start_time: int, timeout: int = 30) -> bool:
        """Wait for World function to complete"""
        log_group = f"/aws/lambda/{self.functions['World']}"
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                response = self.logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=start_time,
                    filterPattern="END RequestId"
                )
                if response.get('events', []):
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        
        return False
    
    def get_e2e_latency(self, start_time: int) -> float:
        """Get E2E latency by finding World's END timestamp"""
        log_group = f"/aws/lambda/{self.functions['World']}"
        
        try:
            response = self.logs_client.filter_log_events(
                logGroupName=log_group,
                startTime=start_time,
                filterPattern="END RequestId"
            )
            events = response.get('events', [])
            if events:
                end_time = max(e['timestamp'] for e in events)
                return end_time - start_time
        except Exception as e:
            print(f"Error getting latency: {e}")
        
        return 0
    
    def run_iterations(self, mode: str, iterations: int = 10) -> List[float]:
        """Run multiple iterations and return latencies"""
        latencies = []
        
        for i in range(iterations):
            print(f"    Iteration {i+1}/{iterations}...", end=" ", flush=True)
            
            session_id, start_time = self.invoke_workflow()
            
            if self.wait_for_completion(start_time):
                time.sleep(2)  # Wait for logs
                latency = self.get_e2e_latency(start_time)
                latencies.append(latency)
                print(f"E2E: {latency:.0f}ms")
            else:
                print("Timeout!")
            
            time.sleep(1)  # Brief pause between iterations
        
        return latencies
    
    def run_benchmark(self, iterations: int = 10):
        """Run complete benchmark comparing both modes"""
        print("="*60)
        print("EARLY INVOCATION BENCHMARK: Hello → World Chain")
        print("="*60)
        
        results = {}
        
        # Test WITHOUT Early Invoke
        print("\n[1] Testing WITHOUT Early Invocation (EARLY_INVOKE=false)")
        self.set_early_invoke(False)
        latencies_off = self.run_iterations("OFF", iterations)
        
        if latencies_off:
            results['without'] = {
                'mean': statistics.mean(latencies_off),
                'std': statistics.stdev(latencies_off) if len(latencies_off) > 1 else 0,
                'min': min(latencies_off),
                'max': max(latencies_off),
                'values': latencies_off
            }
        
        # Test WITH Early Invoke
        print("\n[2] Testing WITH Early Invocation (EARLY_INVOKE=true)")
        self.set_early_invoke(True)
        latencies_on = self.run_iterations("ON", iterations)
        
        if latencies_on:
            results['with'] = {
                'mean': statistics.mean(latencies_on),
                'std': statistics.stdev(latencies_on) if len(latencies_on) > 1 else 0,
                'min': min(latencies_on),
                'max': max(latencies_on),
                'values': latencies_on
            }
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        if 'without' in results and 'with' in results:
            improvement = results['without']['mean'] - results['with']['mean']
            improvement_pct = (improvement / results['without']['mean']) * 100 if results['without']['mean'] > 0 else 0
            
            print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    E2E Latency (ms)                         │
├─────────────────────────────────────────────────────────────┤
│ Mode                │  Mean  │  Std   │  Min   │   Max     │
├─────────────────────────────────────────────────────────────┤
│ WITHOUT Early Invoke│ {results['without']['mean']:6.0f} │ {results['without']['std']:6.0f} │ {results['without']['min']:6.0f} │ {results['without']['max']:7.0f} │
│ WITH Early Invoke   │ {results['with']['mean']:6.0f} │ {results['with']['std']:6.0f} │ {results['with']['min']:6.0f} │ {results['with']['max']:7.0f} │
├─────────────────────────────────────────────────────────────┤
│ IMPROVEMENT         │ {improvement:6.0f}ms ({improvement_pct:5.1f}%)                    │
└─────────────────────────────────────────────────────────────┘
""")
        
        return results


def main():
    benchmark = EarlyInvokeBenchmark()
    
    if not benchmark.functions:
        print("ERROR: Could not find functions. Is the stack deployed?")
        print("Deploy with: cd hello-world && sam build && sam deploy --guided")
        return
    
    results = benchmark.run_benchmark(iterations=8)
    
    # Save results
    with open('early_invoke_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to early_invoke_results.json")


if __name__ == '__main__':
    main()
