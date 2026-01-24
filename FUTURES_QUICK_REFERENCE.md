# Quick Reference: Future-Based Execution

## The Core Mechanism in 3 Steps

### 1. **The Traffic Light: `asyncio.Event()`**

```python
self._event = asyncio.Event()  # Starts RED (not set)
```

- **RED (not set):** "Value not ready - block anyone who asks"
- **GREEN (set):** "Value ready - let everyone through"

### 2. **Waiting at the Light: `await event.wait()`**

```python
async def await_value(self):
    await self._event.wait()  # ← Stops here if RED
    return self._value
```

**If RED:**

- Function execution **pauses** at this exact line
- CPU goes idle (no busy-waiting)
- Event loop handles other tasks
- Function "sleeps" until woken up

**If GREEN:**

- Returns immediately
- No waiting at all

### 3. **Turning Green: `event.set()`**

```python
def set_value(self, value):
    self._value = value
    self._event.set()  # ← Turn GREEN, wake up waiters
```

- Changes light from RED → GREEN
- **Wakes up all waiting coroutines**
- They resume execution and get the value

---

## Complete Example Flow

```python
# ===== Time: 0ms =====
future = UnumFuture(is_ready=False)  # Event is RED

# ===== Time: 100ms =====
# Your function tries to access the value
result = await future.await_value()
         ↓
await self._event.wait()  # ← STOPS HERE (light is RED)
         ↓
    ⏸️ PAUSED
         ↓
    (sleeping...)

# ===== Time: 500ms =====
# Another Lambda writes checkpoint and calls:
future.set_value(42)
         ↓
self._event.set()  # ← Turn GREEN
         ↓
    ▶️ RESUME (wake up the paused function)
         ↓
await self._event.wait()  # ← Returns immediately (light is GREEN)
return self._value  # Returns 42
         ↓
result = 42  # Your code continues
```

---

## Why This Is Efficient (Local) vs Expensive (Serverless)

### ❌ **Bad Approach: Busy-Waiting**

```python
while not ready:
    if check_datastore():  # Wastes CPU
        break
    # Immediately loops again - burns resources
```

### ✅ **Good Approach (Local): Event-Based Waiting**

```python
await event.wait()  # Suspends, CPU idles
# Wakes up automatically when value arrives
```

**Benefits (on regular servers):**

- No CPU waste
- Server runs 24/7 anyway (fixed cost)
- Can handle other tasks while waiting
- Automatic wakeup when data arrives

### ⚠️ **Serverless Reality: Lambda Keeps Running**

```python
# In AWS Lambda context
await event.wait()
# ↑ Lambda function is STILL ALIVE
# ↑ Lambda is STILL CHARGING YOU
# ↑ You're paying per millisecond while waiting
```

**Serverless costs:**

- Lambda charges per millisecond of execution
- Waiting 5 seconds = paying for 5 seconds
- No "free suspend" - function must stay alive
- Only economical for SHORT waits (< 1-2 seconds)

**When to use futures in Lambda:**

- ✅ Wait time < 1-2 seconds (hide cold start)
- ❌ Wait time > 5 seconds (use Step Functions instead)

---

## Visual: 3 Branches, Early Invocation

```
Time 0ms:    Branch A ──┐
             Branch B ──┤  All start
             Branch C ──┘

Time 100ms:  Branch A ✓ (finishes first)
             └─> Invokes aggregate_function with:
                 ├─ param_a: UnumFuture(ready=True, value=10)  🟢
                 ├─ param_b: UnumFuture(ready=False)           🔴
                 └─ param_c: UnumFuture(ready=False)           🔴

Time 150ms:  aggregate_function starts

             Line: val_a = await param_a.await_value()
                   └─> Event is GREEN → returns 10 instantly ✓

             Line: val_b = await param_b.await_value()
                   └─> Event is RED → ⏸️ BLOCKS HERE

Time 800ms:  Branch B ✓ (finishes)
             └─> Calls: param_b.set_value(20)
                 └─> Event turns GREEN 🟢
                 └─> aggregate_function ▶️ RESUMES

             Line: val_b = await param_b.await_value()
                   └─> Returns 20 ✓

             Line: val_c = await param_c.await_value()
                   └─> Event is RED → ⏸️ BLOCKS HERE

Time 5000ms: Branch C ✓ (finishes)
             └─> Calls: param_c.set_value(30)
                 └─> Event turns GREEN 🟢
                 └─> aggregate_function ▶️ RESUMES

             Line: val_c = await param_c.await_value()
                   └─> Returns 30 ✓

             Line: return val_a + val_b + val_c
                   └─> Returns 60 ✓

Time 5001ms: Function completes
```

---

## The Key Insight

**Traditional Unum:**

```
Wait for ALL branches → Invoke function → Execute
         ↓
    Wasted time (cold start happens AFTER all inputs ready)
```

**Future-Based Unum:**

```
First branch finishes → Invoke function IMMEDIATELY → Execute with futures
                             ↓
                    Cold start happens WHILE other branches run
                             ↓
                    Function blocks ONLY when it needs pending data
```

**Result:** Cold start penalty is **hidden** behind waiting for slow branches.

---

## Implementation Summary

### What You Write (Original):

```python
def aggregate(sensor_a, sensor_b, sensor_c):
    return sensor_a + sensor_b + sensor_c
```

### What AST Transformer Creates:

```python
async def aggregate(sensor_a, sensor_b, sensor_c):
    val_a = await sensor_a.await_value()  # Blocks if RED
    val_b = await sensor_b.await_value()  # Blocks if RED
    val_c = await sensor_c.await_value()  # Blocks if RED
    return val_a + val_b + val_c
```

### What Runtime Provides:

```python
sensor_a = UnumFuture(value=10, is_ready=True)   # GREEN
sensor_b = UnumFuture(is_ready=False)            # RED (will turn GREEN later)
sensor_c = UnumFuture(is_ready=False)            # RED (will turn GREEN later)

result = await aggregate(sensor_a, sensor_b, sensor_c)
```

---

## Test It Yourself

Run the demo:

```bash
cd examples
python future_blocking_demo.py
```

Watch the timestamps to see:

- When functions block (⏸️)
- When they resume (▶️)
- How long they wait

---

## Files to Read

1. **`FUTURES_EXECUTION_EXPLAINED.md`** - Complete detailed explanation
2. **`function_transformer.py`** - UnumFuture class with detailed comments
3. **`examples/future_blocking_demo.py`** - Interactive demonstration
4. **`docs/futures-partial-application.md`** - Full architecture documentation

---

## Bottom Line

**The "magic" is just `asyncio.Event()`:**

- A simple, efficient synchronization primitive
- Built into Python's standard library
- Powers all async/await code
- Enables non-blocking waits without CPU waste

**No complex polling, no timeouts, no busy-waiting.**  
Just efficient cooperative multitasking.
