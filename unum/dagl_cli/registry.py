"""DAGL Function Registry — Redis-backed registry for tracking deployed functions.

Tracks:
  - Which functions are deployed, their ARN, runtime, region
  - Input/output schemas (optional, declared in dagl.yaml)
  - Deployment timestamps and account ID

Redis key layout:
  dagl:functions:{name}          → Hash (arn, runtime, handler, region, account, deployed_at, ...)
  dagl:functions:{name}:input    → JSON Schema string
  dagl:functions:{name}:output   → JSON Schema string
  dagl:accounts:{account_id}     → Set of function names owned by this account

Usage:
  from registry import Registry
  reg = Registry(host="localhost", port=6379)
  reg.register("my-func", arn="arn:...", runtime="python3.13", ...)
  reg.lookup("my-func")  →  dict
  reg.check_deployed("my-func")  →  bool
  reg.validate_io(workflow_edges)  →  list of errors
"""

import json
import time


class Registry:
    """Redis-backed function registry."""

    def __init__(self, host="localhost", port=6379, db=0, password=None, url=None):
        import redis
        if url:
            self.r = redis.Redis.from_url(url, decode_responses=True)
        else:
            self.r = redis.Redis(
                host=host, port=port, db=db,
                password=password, decode_responses=True,
            )
        # Verify connection
        self.r.ping()

    # ── Register / Update ────────────────────────────────────────────────────

    def register(self, name, *, arn, runtime, handler, region, account_id,
                 platform="aws", input_schema=None, output_schema=None):
        """Register or update a function in the registry."""
        key = f"dagl:functions:{name}"
        data = {
            "arn": arn,
            "runtime": runtime,
            "handler": handler,
            "region": region,
            "account_id": account_id,
            "platform": platform,
            "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.r.hset(key, mapping=data)

        # Track function in account set
        self.r.sadd(f"dagl:accounts:{account_id}", name)

        # Store schemas if provided
        if input_schema is not None:
            self.r.set(f"{key}:input", json.dumps(input_schema))
        if output_schema is not None:
            self.r.set(f"{key}:output", json.dumps(output_schema))

    def unregister(self, name):
        """Remove a function from the registry."""
        key = f"dagl:functions:{name}"
        info = self.r.hgetall(key)
        if info and "account_id" in info:
            self.r.srem(f"dagl:accounts:{info['account_id']}", name)
        self.r.delete(key, f"{key}:input", f"{key}:output")

    # ── Query ────────────────────────────────────────────────────────────────

    def lookup(self, name):
        """Get full info for a function. Returns dict or None."""
        key = f"dagl:functions:{name}"
        info = self.r.hgetall(key)
        if not info:
            return None
        # Attach schemas if present
        inp = self.r.get(f"{key}:input")
        out = self.r.get(f"{key}:output")
        if inp:
            info["input_schema"] = json.loads(inp)
        if out:
            info["output_schema"] = json.loads(out)
        return info

    def check_deployed(self, name):
        """Check if a function is registered."""
        return self.r.exists(f"dagl:functions:{name}") > 0

    def list_functions(self, account_id=None):
        """List all registered functions, optionally filtered by account."""
        if account_id:
            names = self.r.smembers(f"dagl:accounts:{account_id}")
        else:
            # Scan for all dagl:functions:* keys (exclude :input/:output)
            names = set()
            cursor = 0
            while True:
                cursor, keys = self.r.scan(cursor, match="dagl:functions:*", count=100)
                for k in keys:
                    parts = k.split(":")
                    if len(parts) == 3:  # dagl:functions:{name} only
                        names.add(parts[2])
                if cursor == 0:
                    break
        return sorted(names)

    # ── Schema Validation ────────────────────────────────────────────────────

    def set_schema(self, name, input_schema=None, output_schema=None):
        """Set input/output schemas for a function."""
        key = f"dagl:functions:{name}"
        if input_schema is not None:
            self.r.set(f"{key}:input", json.dumps(input_schema))
        if output_schema is not None:
            self.r.set(f"{key}:output", json.dumps(output_schema))

    def get_input_schema(self, name):
        raw = self.r.get(f"dagl:functions:{name}:input")
        return json.loads(raw) if raw else None

    def get_output_schema(self, name):
        raw = self.r.get(f"dagl:functions:{name}:output")
        return json.loads(raw) if raw else None

    def validate_workflow(self, configs, functions_map):
        """Validate a workflow against the registry.

        Returns a list of (level, message) tuples:
          level = "error" | "warning"

        Checks:
          1. All functions exist in registry (or are deployed)
          2. Output schema of source matches input schema of target for each edge
        """
        issues = []

        # Check all functions are registered
        for func_name, func_ref in functions_map.items():
            if not self.check_deployed(func_ref):
                issues.append(("error", f"Function '{func_ref}' (mapped from {func_name}) is not in the registry"))

        # Check I/O compatibility along edges
        for func_name, cfg in configs.items():
            nxt = cfg.get("Next")
            if not nxt:
                continue

            next_name = nxt.get("Name") if isinstance(nxt, dict) else nxt
            if not next_name:
                continue

            src_ref = functions_map.get(func_name)
            dst_ref = functions_map.get(next_name)
            if not src_ref or not dst_ref:
                continue

            src_out = self.get_output_schema(src_ref)
            dst_in = self.get_input_schema(dst_ref)

            if src_out and dst_in:
                # Check that all required input fields are provided by output
                mismatches = _check_schema_compat(src_out, dst_in)
                for m in mismatches:
                    issues.append(("error", f"{func_name} → {next_name}: {m}"))
            elif src_out and not dst_in:
                issues.append(("warning", f"{next_name} ({dst_ref}) has no input schema — cannot validate edge from {func_name}"))
            elif not src_out and dst_in:
                issues.append(("warning", f"{func_name} ({src_ref}) has no output schema — cannot validate edge to {next_name}"))

        return issues


# ── Schema compatibility check ───────────────────────────────────────────────

# Simple type system: string, number, boolean, object, array, string[], number[]
# Schemas are dicts of {field_name: type_string}

_TYPE_ALIASES = {
    "str": "string",
    "int": "number",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}


def _normalize_type(t):
    t = t.strip().lower()
    return _TYPE_ALIASES.get(t, t)


def _check_schema_compat(output_schema, input_schema):
    """Check if output_schema provides all fields required by input_schema.

    Both are dicts: {field_name: type_string}.
    Returns list of error messages.
    """
    errors = []
    for field, expected_type in input_schema.items():
        expected_type = _normalize_type(expected_type)
        if field not in output_schema:
            errors.append(f"missing field '{field}' (expected {expected_type})")
        else:
            actual_type = _normalize_type(output_schema[field])
            if actual_type != expected_type:
                errors.append(f"field '{field}': type mismatch — outputs {actual_type}, expects {expected_type}")
    return errors
