# Future-Based Execution Implementation

## Overview

This document describes the Future-Based execution implementation added to the Unum runtime. This feature enables **true async fan-in** where the aggregation function is invoked early (before all inputs are ready) and uses `asyncio.Event()` for non-blocking waiting.

## Three Fan-In Modes

Unum now supports three fan-in execution modes:

| Mode                     | Configuration                             | Waiting Style                  | When Invoked               |
| ------------------------ | ----------------------------------------- | ------------------------------ | -------------------------- |
| **CLASSIC**              | `EAGER: false`                            | N/A (all ready)                | After all branches finish  |
| **EAGER (LazyInput)**    | `EAGER: true`, `UNUM_FUTURE_BASED: false` | Sync blocking (`time.sleep`)   | First branch to claim wins |
| **EAGER (Future-Based)** | `EAGER: true`, `UNUM_FUTURE_BASED: true`  | Async non-blocking (`asyncio`) | First branch to claim wins |

## Configuration

### Enable Future-Based Execution

In your `template.yaml`, set these environment variables:

```yaml
Globals:
  Function:
    Environment:
      Variables:
        EAGER: "true" # Enable eager fan-in
        UNUM_FUTURE_BASED: "true" # Use asyncio-based futures
        UNUM_EAGER_POLL_INTERVAL: "0.1" # Poll interval in seconds (optional)
        UNUM_EAGER_TIMEOUT: "300" # Timeout in seconds (optional)
```

## Key Classes

### `UnumFuture`

A true Future/Promise implementation using `asyncio.Event()`:

```python
class UnumFuture:
    """
    Wraps either:
    - A ready value (immediately available)
    - A pending value (will arrive when the branch finishes)

    The asyncio.Event() is the key to non-blocking waiting:
    - Starts as "not set" (blocking)
    - When value arrives, we call set() to unblock
    - await wait() suspends the coroutine until the event is set
    """

    def __init__(self, datastore, session, instance_name, is_ready=False, ...):
        self._event = asyncio.Event()
        if is_ready:
            self._event.set()  # Signal: "value is ready"

    async def await_value(self):
        """Non-blocking wait for the value."""
        while not self._is_ready:
            if await self._poll_datastore():
                break
            await asyncio.sleep(self._poll_interval)  # Yields to event loop
        return self._value

    def set_value(self, value):
        """Called when value arrives - unblocks all waiters."""
        self._value = value
        self._is_ready = True
        self._event.set()  # UNBLOCKS all waiting await_value() calls
```

### `AsyncFutureInputList`

A list-like container supporting both sync and async access:

```python
class AsyncFutureInputList:
    """
    Supports both sync and async access patterns:

    Async (optimal):
        async def lambda_handler(inputs, context):
            data0 = await inputs.get_async(0)  # Non-blocking wait
            async for data in inputs:
                process(data)

    Sync (backwards compatible):
        def lambda_handler(inputs, context):
            data0 = inputs[0]  # Blocks until ready
    """
```

## Usage in User Code

### Option 1: Synchronous (Transparent)

No changes needed! Your existing code works:

```python
def lambda_handler(inputs, context):
    # inputs is AsyncFutureInputList but works like a list
    user_mentions = inputs[0]    # Blocks if not ready
    shortened_urls = inputs[1]   # Blocks if not ready

    for data in inputs:          # Iteration works
        process(data)

    return combine(user_mentions, shortened_urls)
```

### Option 2: Asynchronous (Optimal)

For maximum efficiency, use async:

```python
import asyncio

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

### Option 3: Hybrid (Advanced)

Do initialization before waiting:

```python
import asyncio

async def smart_aggregation(inputs):
    # Do initialization that doesn't need inputs
    model = load_ml_model()
    db_connection = setup_database()

    # Check what's ready without blocking
    if inputs.all_ready():
        print("All inputs ready!")
    else:
        ready, pending = inputs.get_ready_count()
        print(f"{ready} ready, {pending} pending")

    # Now get inputs (waits only for pending ones)
    results = []
    async for data in inputs:
        result = model.predict(data)
        results.append(result)

    return aggregate(results)
```

## API Reference

### UnumFuture Methods

| Method             | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `is_ready`         | Property: check if value is available (non-blocking) |
| `await_value()`    | Async: wait for value with non-blocking polling      |
| `get_value_sync()` | Sync: get value (raises if not ready)                |
| `try_resolve()`    | Try to fetch from datastore without blocking         |
| `set_value(value)` | Set value and unblock all waiters                    |

### AsyncFutureInputList Methods

| Method                         | Description                              |
| ------------------------------ | ---------------------------------------- |
| `len(inputs)`                  | Number of inputs                         |
| `inputs[i]`                    | Sync: get value (blocks if needed)       |
| `await inputs.get_async(i)`    | Async: get value with non-blocking wait  |
| `await inputs.get_all_async()` | Async: get all values in parallel        |
| `async for data in inputs`     | Async iteration                          |
| `inputs.is_ready(i)`           | Check if input i is ready (non-blocking) |
| `inputs.all_ready()`           | Check if all inputs are ready            |
| `inputs.get_ready_count()`     | Returns (ready_count, pending_count)     |
| `inputs.try_resolve_all()`     | Try to resolve all without blocking      |
| `inputs.get_futures()`         | Get underlying UnumFuture list           |

## Performance Benefits

### Cold Start Elimination

```
CLASSIC mode:
  Wait for all branches ──> Cold start ──> Execute
  Total: 5000ms + 200ms + 50ms = 5250ms

Future-Based mode:
  First branch ──> Invoke (cold start overlaps with waiting) ──> Execute
  Total: ≈5000ms (cold start hidden in wait time)

  Savings: 250ms per invocation
```

### CPU Efficiency

- **LazyInput (sync)**: Uses `time.sleep()` - thread blocks, can't do other work
- **UnumFuture (async)**: Uses `asyncio.sleep()` - yields to event loop, other tasks can run

## Architecture Diagram

```
Branch A ──┐
           │   [First to finish claims]
Branch B ──┼───────────────────────────────> Aggregation Function
           │                                      │
Branch C ──┘                                      │
    │                                             │
    │    [Writes checkpoint]                      │
    └──────────────────────────────> [Future resolves, unblocks]
```

## Files Modified

- `unum/runtime/ds.py`: Added `UnumFuture`, `AsyncFutureInputList`, `create_future_inputs()`
- `unum/runtime/main.py`: Updated `ingress()` to use Future-Based when enabled
- `unum-appstore/text-processing/common/ds.py`: Same updates as above

## Environment Variables

| Variable                   | Default | Description                                       |
| -------------------------- | ------- | ------------------------------------------------- |
| `EAGER`                    | `false` | Enable eager fan-in (first branch invokes)        |
| `UNUM_FUTURE_BASED`        | `false` | Use asyncio-based UnumFuture instead of LazyInput |
| `UNUM_EAGER_POLL_INTERVAL` | `0.1`   | Seconds between datastore polls                   |
| `UNUM_EAGER_TIMEOUT`       | `300`   | Maximum seconds to wait for inputs                |

## Comparison: LazyInput vs UnumFuture

| Feature           | LazyInput      | UnumFuture                |
| ----------------- | -------------- | ------------------------- |
| Waiting mechanism | `time.sleep()` | `asyncio.sleep()`         |
| Thread blocking   | Yes            | No (cooperative)          |
| Parallel waiting  | Sequential     | `asyncio.gather()`        |
| User code style   | Sync only      | Sync or Async             |
| CPU while waiting | Idle (blocked) | Available for other tasks |
| Event-driven      | No (polling)   | Yes (`asyncio.Event`)     |
