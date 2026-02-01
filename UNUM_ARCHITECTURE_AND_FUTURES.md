# Unum Architecture and Future-Based Execution

## Overview

This document provides a comprehensive explanation of the Unum system architecture, including both the original implementation and the Future-Based execution enhancement. This is intended as context for LLMs and developers creating new Unum application examples.

---

## Part 1: Original Unum Implementation

### What is Unum?

Unum is a **decentralized orchestration system** for building large FaaS (Function-as-a-Service) applications. Unlike traditional orchestrators (e.g., AWS Step Functions), Unum runs orchestration logic **in-site** with user functions, eliminating the need for a separate orchestrator service.

### Key Differentiators

1. **No Orchestrator Service Required**: Unum only needs:
   - A FaaS scheduler (e.g., AWS Lambda, Google Cloud Functions)
   - A strongly consistent data store (e.g., DynamoDB, Firestore)

2. **Portability**: Write once using Step Functions language, deploy to AWS or Google Cloud

3. **Cost Efficiency**: Up to 9x cheaper than standalone orchestrator services

4. **Flexibility**: Applications can implement custom patterns not supported by standard orchestrators

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                      │
│  (Step Functions State Machine or Hand-written IR)      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    unum-cli Compiler                     │
│  • Compiles Step Functions → Unum IR                     │
│  • Generates unum_config.json for each function         │
│  • Creates platform-specific templates (SAM, etc.)      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Deployed Lambda Functions                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Function 1   │  │ Function 2   │  │ Function 3   │ │
│  │              │  │              │  │              │ │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │ │
│  │ │ Unum     │ │  │ │ Unum     │ │  │ │ Unum     │ │ │
│  │ │ Runtime  │ │  │ │ Runtime  │ │  │ │ Runtime  │ │ │
│  │ └────┬─────┘ │  │ └────┬─────┘ │  │ └────┬─────┘ │ │
│  │      │       │  │      │       │  │      │       │ │
│  │ ┌────▼─────┐ │  │ ┌────▼─────┐ │  │ ┌────▼─────┐ │ │
│  │ │ User     │ │  │ │ User     │ │  │ │ User     │ │ │
│  │ │ Code     │ │  │ │ Code     │ │  │ │ Code     │ │ │
│  │ │ (app.py) │ │  │ │ (app.py) │ │  │ │ (app.py) │ │ │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              DynamoDB (Intermediary Data Store)          │
│  • Stores checkpoints (function outputs)                │
│  • Enables exactly-once execution                       │
│  • Coordinates fan-in operations                         │
└─────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Unum Runtime Library

The runtime wraps each user function and provides:

- **Orchestration**: Determines which functions to invoke next
- **Error Handling**: Retries, error propagation
- **Exactly-Once Execution**: Checkpoint-based deduplication
- **Fan-in/Fan-out**: Parallel execution patterns

#### 2. Execution Flow

Each Lambda function follows this pattern:

```python
def lambda_handler(event, context):
    # 1. Check for existing checkpoint (exactly-once guarantee)
    checkpoint = unum.get_checkpoint(event)
    
    if checkpoint:
        # Function already executed - use cached result
        user_output = checkpoint['User']
    else:
        # 2. Extract input (ingress)
        user_input = ingress(event)
        
        # 3. Execute user function
        user_output = user_lambda(user_input, context)
        
        # 4. Checkpoint result (egress)
        unum.checkpoint(session, instance_name, user_output)
    
    # 5. Invoke continuations (next functions in workflow)
    unum.run_continuation(event, user_output)
    
    return user_output, session, metadata
```

#### 3. Data Flow Patterns

**Chaining (Sequential):**
```
Function A → Function B → Function C
```

**Fan-out (Parallel):**
```
Function A ──┬──> Function B
             └──> Function C
```

**Fan-in (Aggregation):**
```
Function A ──┐
Function B ──┼──> Aggregation Function
Function C ──┘
```

**Traditional Fan-in Behavior:**
- Wait for ALL branches to complete
- Read all checkpoints from data store
- Invoke aggregation function with all inputs ready

### Key Concepts

#### Checkpointing

Unum uses checkpoints to ensure exactly-once execution:

1. Before executing user code, check if checkpoint exists
2. If exists → skip execution, use cached result
3. If not → execute, then write checkpoint
4. Checkpoint contains: `{"User": output, "GC": metadata}`

#### Session Management

Each workflow execution has a unique session ID:
- Generated by entry function
- Propagated through all function invocations
- Used to namespace checkpoints in data store

#### Instance Names

Each function execution gets a unique instance name:
- Format: `{FunctionName}-unumIndex-{index}`
- Used as checkpoint key in data store
- Enables parallel executions of same function

#### Continuation System

Functions declare their continuations in `unum_config.json`:

```json
{
  "Name": "MyFunction",
  "Next": [
    {
      "Name": "NextFunction",
      "InputType": "Scalar"
    }
  ]
}
```

The runtime automatically invokes continuations after checkpointing.

---

## Part 2: Future-Based Execution Enhancement

### The Problem with Traditional Fan-in

In traditional Unum fan-in:

```
Branch A (100ms) ──┐
Branch B (800ms) ──┤──> Wait 800ms ──> Cold Start (200ms) ──> Execute (50ms)
Branch C (5000ms) ─┘
Total: 5250ms
```

**Issues:**
- Aggregation function waits for slowest branch
- Cold start happens AFTER all inputs ready
- Wasted time = cold start duration

### The Solution: Early Invocation with Futures

**Key Idea:** Invoke aggregation function as soon as the FIRST branch finishes, passing:
- **Ready values** (completed branches)
- **Futures/Promises** (pending branches)

The function starts executing immediately and blocks only when accessing a value that isn't ready yet.

### Three Fan-In Modes

| Mode | Configuration | Waiting Style | When Invoked |
|------|--------------|---------------|--------------|
| **CLASSIC** | `EAGER: false` | N/A (all ready) | After all branches finish |
| **EAGER (LazyInput)** | `EAGER: true`, `UNUM_FUTURE_BASED: false` | Sync blocking (`time.sleep`) | First branch to claim wins |
| **EAGER (Future-Based)** | `EAGER: true`, `UNUM_FUTURE_BASED: true` | Async non-blocking (`asyncio`) | First branch to claim wins |

### Future-Based Execution Flow

```
Time 0ms:    Branch A, B, C all start executing

Time 100ms:  Branch A finishes ✓
             └─> Invokes aggregate_function with:
                 ├─ param_a: UnumFuture(value=result_A, is_ready=True)  🟢
                 ├─ param_b: UnumFuture(is_ready=False)                 🔴
                 └─ param_c: UnumFuture(is_ready=False)                 🔴

Time 150ms:  aggregate_function starts (cold start overlaps with waiting)
             
             Line: val_a = await param_a.await_value()
                   └─> Event is GREEN → returns immediately ✓
             
             Line: val_b = await param_b.await_value()
                   └─> Event is RED → ⏸️ BLOCKS HERE (non-blocking)

Time 800ms:  Branch B finishes ✓
             └─> Writes checkpoint → param_b.set_value(result_B)
                 └─> Event turns GREEN → aggregate_function ▶️ RESUMES

Time 5000ms: Branch C finishes ✓
             └─> Writes checkpoint → param_c.set_value(result_C)
                 └─> Event turns GREEN → aggregate_function ▶️ RESUMES

Time 5001ms: Function completes
Total: ~5000ms (cold start hidden in wait time)
```

### Core Mechanism: asyncio.Event()

The blocking/waiting mechanism uses Python's `asyncio.Event()`:

```python
class UnumFuture:
    def __init__(self, is_ready=False):
        self._event = asyncio.Event()  # Traffic light
        if is_ready:
            self._event.set()  # GREEN: value ready
    
    async def await_value(self):
        """Non-blocking wait for value"""
        await self._event.wait()  # ← Blocks if RED, returns if GREEN
        return self._value
    
    def set_value(self, value):
        """Called when value arrives"""
        self._value = value
        self._is_ready = True
        self._event.set()  # ← Turn GREEN, wake up waiters
```

**How it works:**
- `asyncio.Event()` starts as "not set" (RED light)
- `await event.wait()` suspends the coroutine if RED
- `event.set()` turns GREEN and wakes up all waiters
- CPU doesn't spin-wait - function yields to event loop

### User Code Interface

#### Option 1: Synchronous (Transparent)

No changes needed! Existing code works:

```python
def lambda_handler(inputs, context):
    # inputs is AsyncFutureInputList but works like a list
    user_mentions = inputs[0]    # Blocks if not ready
    shortened_urls = inputs[1]    # Blocks if not ready
    
    for data in inputs:          # Iteration works
        process(data)
    
    return combine(user_mentions, shortened_urls)
```

#### Option 2: Asynchronous (Optimal)

For maximum efficiency:

```python
async def process_inputs_async(inputs):
    # Get all inputs in parallel (waits for slowest)
    all_data = await inputs.get_all_async()
    
    # Or get individually with non-blocking wait
    data0 = await inputs.get_async(0)
    data1 = await inputs.get_async(1)
    
    # Or iterate asynchronously
    async for data in inputs:
        process(data)
    
    return combine(all_data)

def lambda_handler(event, context):
    inputs = ingress(event)
    result = asyncio.run(process_inputs_async(inputs))
    return egress(result, event)
```

### Configuration

Enable Future-Based execution in `template.yaml`:

```yaml
Globals:
  Function:
    Environment:
      Variables:
        EAGER: "true"                    # Enable eager fan-in
        UNUM_FUTURE_BASED: "true"        # Use asyncio-based futures
        UNUM_EAGER_POLL_INTERVAL: "0.1"  # Poll interval (seconds)
        UNUM_EAGER_TIMEOUT: "300"        # Timeout (seconds)
```

### Performance Benefits

1. **Cold Start Elimination**: Cold start overlaps with waiting for slow branches
2. **CPU Efficiency**: Uses `asyncio.sleep()` instead of `time.sleep()` - yields to event loop
3. **Early Processing**: Function can start processing ready inputs while waiting for others

### Comparison: LazyInput vs UnumFuture

| Feature | LazyInput | UnumFuture |
|--------|-----------|------------|
| Waiting mechanism | `time.sleep()` | `asyncio.sleep()` |
| Thread blocking | Yes | No (cooperative) |
| Parallel waiting | Sequential | `asyncio.gather()` |
| User code style | Sync only | Sync or Async |
| CPU while waiting | Idle (blocked) | Available for other tasks |
| Event-driven | No (polling) | Yes (`asyncio.Event`) |

---

## Part 3: How They Work Together

### Integration Points

1. **Ingress Function**: Detects eager fan-in and creates appropriate input type
   - CLASSIC: Regular list of values
   - EAGER (LazyInput): `LazyInputList` with sync blocking
   - EAGER (Future-Based): `AsyncFutureInputList` with async waiting

2. **Checkpoint System**: Unchanged - still uses DynamoDB for exactly-once

3. **Continuation System**: Unchanged - still invokes next functions after checkpointing

4. **User Code**: Can remain unchanged (transparent) or use async for optimization

### Execution Modes Summary

**CLASSIC Mode (Original):**
```
All branches finish → Read checkpoints → Invoke aggregation → Execute
```

**EAGER Mode (LazyInput):**
```
First branch finishes → Invoke aggregation → Sync wait for others → Execute
```

**EAGER Mode (Future-Based):**
```
First branch finishes → Invoke aggregation → Async wait for others → Execute
```

### When to Use Each Mode

- **CLASSIC**: Simple workflows, all branches similar duration
- **EAGER (LazyInput)**: Eager fan-in needed, but async not available/needed
- **EAGER (Future-Based)**: Eager fan-in with async support, optimal performance

---

## Part 4: Application Structure

### Directory Layout

```
myapp/
├── unum-template.yaml          # Application configuration
├── unum-step-functions.json   # Workflow definition (Step Functions)
├── template.yaml              # Generated platform template (SAM)
├── function1/
│   ├── app.py                 # User function code
│   ├── unum_config.json       # Unum configuration (generated or manual)
│   ├── requirements.txt       # Python dependencies
│   └── main.py                # Unum runtime wrapper (generated)
├── function2/
│   └── ... (same structure)
└── common/
    └── ... (shared Unum runtime)
```

### unum-template.yaml

```yaml
Globals:
  ApplicationName: my-app
  WorkflowType: step-functions
  WorkflowDefinition: unum-step-functions.json
  FaaSPlatform: aws
  UnumIntermediaryDataStoreType: dynamodb
  UnumIntermediaryDataStoreName: unum-intermediate-datastore
  Checkpoint: true
  GC: false
  Debug: false

Functions:
  Function1:
    Properties:
      CodeUri: function1/
      Runtime: python3.13
      Start: true
```

### unum_config.json

Generated per function, defines orchestration:

```json
{
  "Name": "Function1",
  "Next": [
    {
      "Name": "Function2",
      "InputType": "Scalar"
    }
  ],
  "Checkpoint": true
}
```

### Build and Deploy Process

1. **Compile**: Convert Step Functions → Unum IR
   ```bash
   unum-cli compile -p step-functions -w unum-step-functions.json -t unum-template.yaml
   ```

2. **Build**: Generate platform template and package functions
   ```bash
   unum-cli build -g -p aws
   ```

3. **Deploy**: Deploy to AWS using SAM
   ```bash
   unum-cli deploy -b
   ```

---

## Part 5: Key Implementation Details

### Ingress Function

The `ingress()` function extracts user input from the event:

```python
def ingress(event):
    if event["Data"]["Source"] == "http":
        # Direct input - pass value directly
        return event["Data"]["Value"]
    else:
        # Fan-in - read from data store
        if EAGER and UNUM_FUTURE_BASED:
            # Create AsyncFutureInputList
            return create_future_inputs(...)
        elif EAGER:
            # Create LazyInputList
            return create_lazy_inputs(...)
        else:
            # CLASSIC - read all checkpoints immediately
            return read_input(event["Data"]["Value"])
```

### Egress Function

The `egress()` function handles checkpointing and continuation:

```python
def egress(user_output, event):
    # 1. Checkpoint user output
    checkpoint_data = {"User": json.dumps(user_output)}
    unum.run_checkpoint(event, checkpoint_data)
    
    # 2. Invoke continuations
    session, metadata = unum.run_continuation(event, user_output)
    
    return session, metadata
```

### Checkpoint System

Checkpoints stored in DynamoDB with key: `{FunctionName}-unumIndex-{index}`

Structure:
```json
{
  "Name": "Function1-unumIndex-0",
  "User": "<serialized user output>",
  "GC": {"outgoing_edges": [...]}
}
```

### Continuation Invocation

When a function finishes, it invokes its continuations:

```python
def run_continuation(event, user_output):
    for continuation in config["Next"]:
        payload = {
            "Data": {
                "Source": "http",
                "Value": user_output  # For scalar
            },
            "Session": session
        }
        invoke_lambda(continuation["Name"], payload)
```

---

## Part 6: Creating New Examples

### Checklist for New Applications

1. **Define Workflow**: Create `unum-step-functions.json` or write `unum_config.json` manually
2. **Create Functions**: Each function in its own directory with `app.py`
3. **Configure**: Set up `unum-template.yaml` with application settings
4. **Choose Mode**: Decide on CLASSIC, EAGER (LazyInput), or EAGER (Future-Based)
5. **Build**: Run `unum-cli build -g -p aws`
6. **Deploy**: Run `unum-cli deploy -b`
7. **Test**: Create test scripts to invoke deployed functions

### Common Patterns

**Simple Chain:**
```json
Function1 → Function2 → Function3
```

**Parallel Fan-out:**
```json
Function1 → [Function2, Function3]
```

**Fan-in:**
```json
[Function1, Function2, Function3] → AggregateFunction
```

**Map Pattern:**
```json
MapFunction → [F1-instance-0, F1-instance-1, ...] → ReduceFunction
```

### Testing Considerations

- Create DynamoDB table: `unum-intermediate-datastore`
- Use proper payload format: `{"Data": {"Source": "http", "Value": {...}}}`
- Check CloudWatch logs for debugging
- Verify checkpoint creation in DynamoDB

---

## Summary

**Original Unum:**
- Decentralized orchestration without separate orchestrator service
- Checkpoint-based exactly-once execution
- Supports all Step Functions patterns
- Portable across cloud platforms

**Future-Based Enhancement:**
- Early invocation of fan-in functions
- Non-blocking async waiting with `asyncio.Event()`
- Hides cold start latency
- Transparent to user code (backwards compatible)

**Together:**
- Maintains all original guarantees (exactly-once, error handling)
- Adds performance optimization for fan-in patterns
- Provides flexible execution modes
- Enables both sync and async user code

This architecture enables building efficient, portable serverless applications with fine-grained control over execution patterns while maintaining the simplicity of traditional orchestrators.
