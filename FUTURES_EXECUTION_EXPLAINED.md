# Future-Based Execution: Complete Explanation

## Overview: The Problem

In traditional Unum fan-in patterns, you have multiple branches running in parallel:

```
Branch A ──┐
Branch B ──┤──> Aggregation Function
Branch C ──┘
```

**Traditional approach:** Wait for ALL branches to finish, THEN invoke aggregation function.

**Problem:** If Branch A finishes in 100ms but Branch C takes 5000ms, the aggregation function waits 5000ms before starting, even though it could start processing Branch A's result immediately.

---

## The Future-Based Solution

**Key Idea:** Invoke the aggregation function EARLY (as soon as the FIRST branch finishes) with a mix of:

- **Ready values** (branches that finished)
- **Futures/Promises** (branches still running)

The function starts executing immediately and **blocks only when it actually needs** a value that isn't ready yet.

---

## The Blocking/Waiting Mechanism Explained

### Step 1: The UnumFuture Object

At the heart of the system is `UnumFuture` - a wrapper that can hold either:

- A **ready value** (immediately available)
- A **pending value** (will arrive later)

```python
class UnumFuture:
    def __init__(self, value=None, is_ready=False):
        self._value = value              # The actual data
        self._is_ready = is_ready        # Is it available now?
        self._event = asyncio.Event()    # THE BLOCKING MECHANISM
        if is_ready:
            self._event.set()            # Signal: "value is ready"
```

**The `asyncio.Event()` is the KEY to blocking/waiting:**

- It's like a traffic light that starts RED (blocking)
- When the value arrives, we turn it GREEN (unblock)
- Functions waiting at the light can proceed

### Step 2: Waiting for a Value

When your function needs a parameter:

```python
async def await_value(self):
    """Wait until the value is available and return it"""
    await self._event.wait()  # ← THIS LINE BLOCKS if value not ready
    return self._value
```

**What `await self._event.wait()` does:**

1. **If value is ready** (`_event` is set): Returns immediately
2. **If value is NOT ready** (`_event` is not set):
   - Suspends the function execution
   - Yields control back to the event loop
   - Waits until someone calls `set_value()`

**This is non-blocking waiting!** The Lambda function doesn't waste CPU cycles checking "is it ready yet?" over and over. Instead, it goes to sleep and the event loop wakes it up when data arrives.

### Step 3: Providing the Value (Unblocking)

When a pending branch finishes and writes its checkpoint:

```python
def set_value(self, value):
    """Set the value when it becomes available"""
    self._value = value
    self._is_ready = True
    self._event.set()  # ← UNBLOCKS all waiting await_value() calls
```

**What `self._event.set()` does:**

- Turns the traffic light GREEN
- Wakes up any function waiting at `await self._event.wait()`
- Allows execution to continue

---

## Complete Execution Flow Example

Let's trace a real scenario with 3 branches:

### Timeline

```
Time 0ms:   Branch A, B, C all start executing
Time 100ms: Branch A finishes ✓
            └─> Invokes aggregate_function with:
                - param_a: UnumFuture(value=result_A, is_ready=True)
                - param_b: UnumFuture(is_ready=False)  ← NOT READY
                - param_c: UnumFuture(is_ready=False)  ← NOT READY

Time 150ms: aggregate_function starts executing
            User code: result = param_a + param_b + param_c

            Step 1: Access param_a
            ├─> await param_a.await_value()
            ├─> Event is already SET (ready=True)
            └─> Returns immediately with result_A ✓

            Step 2: Access param_b
            ├─> await param_b.await_value()
            ├─> Event is NOT set (ready=False)
            └─> **BLOCKS HERE** ⏸️ (function suspended)

Time 800ms: Branch B finishes ✓
            └─> Writes checkpoint to DynamoDB
            └─> Triggers update: param_b.set_value(result_B)
                ├─> Sets _event for param_b
                └─> **UNBLOCKS aggregate_function** ▶️

Time 801ms: aggregate_function resumes
            ├─> await param_b.await_value() returns result_B
            └─> Continues to next line: Access param_c

            Step 3: Access param_c
            ├─> await param_c.await_value()
            ├─> Event is NOT set (ready=False)
            └─> **BLOCKS HERE** ⏸️ (function suspended again)

Time 5000ms: Branch C finishes ✓
             └─> Writes checkpoint to DynamoDB
             └─> Triggers update: param_c.set_value(result_C)
                 ├─> Sets _event for param_c
                 └─> **UNBLOCKS aggregate_function** ▶️

Time 5001ms: aggregate_function resumes
             ├─> await param_c.await_value() returns result_C
             ├─> Completes: result = result_A + result_B + result_C
             └─> Returns final result ✓
```

### Performance Comparison

**Traditional (wait-then-invoke):**

```
Total time = max(Branch A, B, C) + Cold start + Execution
          = 5000ms + 200ms + 50ms
          = 5250ms
```

**Future-based (invoke-early):**

```
Total time = max(Branch A, B, C) + Execution (overlapped)
          = 5000ms + 50ms (but overlapped with waiting)
          ≈ 5000ms
```

**Savings: 250ms** (eliminated cold start delay)

---

## Code Transformation

### Your Original Function

```python
def aggregate_results(sensor_a, sensor_b, sensor_c):
    total = sensor_a + sensor_b + sensor_c
    average = total / 3
    return average
```

### What the AST Transformer Does

```python
async def aggregate_results(sensor_a, sensor_b, sensor_c):
    # Each parameter access is wrapped with await
    val_a = await sensor_a.await_value()  # ← Blocks if not ready
    val_b = await sensor_b.await_value()  # ← Blocks if not ready
    val_c = await sensor_c.await_value()  # ← Blocks if not ready

    # Now use the actual values
    total = val_a + val_b + val_c
    average = total / 3
    return average
```

**The transformation is automatic** - you don't write the `await` calls yourself.

---

## How Values Arrive From Slow Branches

When Branch B finishes after Branch A already invoked the aggregation function:

1. **Branch B writes checkpoint**

   ```python
   datastore.checkpoint(session, 'branch-b-instance', result_B)
   ```

2. **Datastore notifies waiting functions** (via polling or callback)

   ```python
   backend.update_pending_parameter(
       session='session-uuid',
       instance_name='aggregation-func-instance',
       param_name='branch-b-instance'
   )
   ```

3. **Update mechanism finds the UnumFuture**

   ```python
   # In the running Lambda function
   future_for_param_b.set_value(result_B)
   # This sets the asyncio.Event, unblocking await
   ```

4. **Function resumes execution**
   ```python
   # The await_value() call that was blocked now returns
   val_b = await sensor_b.await_value()  # ← Unblocks and returns result_B
   ```

---

## The Asyncio Event Loop Magic

**Why doesn't the Lambda function timeout while waiting?**

The Lambda function is running an `asyncio` event loop:

```python
# In the lambda_handler
async def async_main():
    result = await user_function_with_futures(**params)
    return result

# This runs the event loop
result = asyncio.run(async_main())
```

**While waiting:**

- The function is **suspended** (not running, not consuming CPU)
- The event loop can handle other tasks (like HTTP requests checking for updates)
- When `_event.set()` is called, the event loop **schedules the function to resume**

**This is cooperative multitasking:**

- Not true parallelism (still single-threaded Python)
- But allows efficient waiting without busy-polling
- Function yields control, gets woken up when data arrives

---

## Polling Mechanism (How Updates Are Detected)

Since Lambda functions can't receive push notifications, we use polling:

```python
class UnumFuture:
    async def await_value(self):
        """Wait with periodic polling"""
        while not self._is_ready:
            # Check datastore every 100ms
            if await self._poll_datastore():
                break
            await asyncio.sleep(0.1)  # Non-blocking sleep
        return self._value

    async def _poll_datastore(self):
        """Check if value is available in datastore"""
        if self.datastore:
            checkpoint = self.datastore.get_checkpoint(
                self.session,
                self.instance_name
            )
            if checkpoint:
                self.set_value(checkpoint['User'])
                return True
        return False
```

**Polling loop:**

1. Check DynamoDB for checkpoint
2. If not found, sleep 100ms (non-blocking)
3. Repeat until found or timeout

---

## Key Takeaways

1. **Blocking is non-blocking:** Uses `asyncio.Event()` - CPU doesn't spin-wait
2. **Early invocation:** Function starts as soon as ONE input is ready
3. **Lazy evaluation:** Only blocks when accessing a specific parameter
4. **Automatic transformation:** AST injects `await` calls for you
5. **Update mechanism:** Polling checks DynamoDB for new checkpoints
6. **Event-driven wakeup:** When value arrives, `set_value()` unblocks waiting code

The "magic" is Python's `asyncio` + `Event` primitive, which allows efficient waiting without wasting Lambda execution time or hitting timeouts.
