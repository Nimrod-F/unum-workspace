"""Function signature registry for unum type checking.

This module implements the docstring-based type extraction + Redis registry approach
(matching the pattern used in incubator-openwhisk-cli). The workflow is:

1. Developer writes a docstring/javadoc in their function source using
   @input and @output annotations.
2. During `unum-cli deploy`, signatures are extracted from source code and
   registered in Redis (function_name → {input, output}).
3. During `unum-cli compile`, the compiler queries Redis for all function
   signatures referenced in the DAG and validates edge compatibility.

Docstring format (Python):
    def lambda_handler(event, context):
        '''
        @input {bucket: string, key: string}
        @output {url: string, size: integer}
        '''

Docstring format (JavaScript):
    /**
     * @input {name: string, age: integer}
     * @output {greeting: string}
     */
    function handler(event) { ... }
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


# ─── Data types ─────────────────────────────────────────────────────────────────

VALID_TYPES = {"string", "number", "integer", "boolean", "object", "array", "any"}


@dataclass
class FunctionSignature:
    """Extracted type signature for a single function."""
    name: str
    input_params: dict[str, str]   # param_name → type
    output_params: dict[str, str]  # param_name → type
    source_file: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input": self.input_params,
            "output": self.output_params,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FunctionSignature":
        return cls(
            name=data["name"],
            input_params=data.get("input", {}),
            output_params=data.get("output", {}),
        )

    def __repr__(self) -> str:
        inp = ", ".join(f"{k}: {v}" for k, v in self.input_params.items())
        out = ", ".join(f"{k}: {v}" for k, v in self.output_params.items())
        return f"{self.name}({inp}) -> {{{out}}}"


# ─── Signature extraction from source code ──────────────────────────────────────

# Pattern: @input {field: type, field: type, ...}
_ANNOTATION_PATTERN = re.compile(
    r'@(input|output)\s*\{([^}]*)\}',
    re.IGNORECASE
)

# Pattern for individual field: name: type or name:type
_FIELD_PATTERN = re.compile(
    r'(\w+)\s*:\s*(\w+)'
)


def _parse_type_annotation(annotation_body: str) -> dict[str, str]:
    """Parse '{field: type, field: type}' into a dict."""
    fields: dict[str, str] = {}
    for match in _FIELD_PATTERN.finditer(annotation_body):
        field_name = match.group(1)
        field_type = match.group(2).lower()
        # Normalize common aliases
        if field_type == "int":
            field_type = "integer"
        elif field_type == "str":
            field_type = "string"
        elif field_type == "bool":
            field_type = "boolean"
        elif field_type == "float" or field_type == "double":
            field_type = "number"
        elif field_type == "list":
            field_type = "array"
        elif field_type == "dict" or field_type == "map":
            field_type = "object"
        fields[field_name] = field_type
    return fields


def extract_signature_from_python(source: str, func_name: str) -> FunctionSignature | None:
    """Extract @input/@output annotations from a Python function's docstring.

    Looks for the docstring of `lambda_handler` (or any top-level function) and
    parses @input/@output annotations.
    """
    input_params: dict[str, str] = {}
    output_params: dict[str, str] = {}

    for match in _ANNOTATION_PATTERN.finditer(source):
        kind = match.group(1).lower()
        body = match.group(2)
        fields = _parse_type_annotation(body)
        if kind == "input":
            input_params.update(fields)
        elif kind == "output":
            output_params.update(fields)

    if not input_params and not output_params:
        return None

    return FunctionSignature(
        name=func_name,
        input_params=input_params,
        output_params=output_params,
    )


def extract_signature_from_javascript(source: str, func_name: str) -> FunctionSignature | None:
    """Extract @input/@output annotations from a JavaScript function's JSDoc."""
    # Same annotation format works for JS too
    return extract_signature_from_python(source, func_name)


def extract_signature_from_file(file_path: str, func_name: str) -> FunctionSignature | None:
    """Extract signature from a source file, auto-detecting language."""
    if not os.path.isfile(file_path):
        return None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".py":
        sig = extract_signature_from_python(source, func_name)
    elif ext in (".js", ".ts", ".mjs"):
        sig = extract_signature_from_javascript(source, func_name)
    else:
        # Try generic: just look for @input/@output anywhere
        sig = extract_signature_from_python(source, func_name)

    if sig:
        sig.source_file = file_path
    return sig


def extract_signature_from_directory(code_dir: str, func_name: str) -> FunctionSignature | None:
    """Extract signature from a function's source directory.

    Searches for app.py, handler.js, handler.ts, index.js, etc.
    """
    candidates = ["app.py", "handler.py", "main.py", "handler.js", "index.js", "handler.ts", "index.ts"]

    for filename in candidates:
        filepath = os.path.join(code_dir, filename)
        sig = extract_signature_from_file(filepath, func_name)
        if sig is not None:
            return sig

    return None


def extract_all_signatures(template: dict) -> dict[str, FunctionSignature]:
    """Extract signatures from all functions in a unum template.

    Args:
        template: Parsed unum-template.yaml

    Returns:
        dict mapping function_name → FunctionSignature (only for functions with annotations)
    """
    signatures: dict[str, FunctionSignature] = {}
    functions = template.get("Functions", {})

    for func_name, func_def in functions.items():
        code_uri = func_def.get("Properties", {}).get("CodeUri", f"{func_name}/")
        code_dir = code_uri.rstrip("/")
        sig = extract_signature_from_directory(code_dir, func_name)
        if sig is not None:
            signatures[func_name] = sig

    return signatures


# ─── Redis registry ─────────────────────────────────────────────────────────────

# Redis key prefix for function signatures
_REDIS_KEY_PREFIX = "unum:signature:"


def _make_key(profile: str, func_name: str) -> str:
    """Build a Redis key scoped to an AWS profile."""
    return f"{_REDIS_KEY_PREFIX}{profile}:{func_name}"


class SignatureRegistry:
    """Stores and retrieves function signatures from Redis.

    Keys are scoped by AWS profile/account so that teams sharing a Redis
    instance but using different AWS accounts don't collide::

        unum:signature:<profile>:<function_name>
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", db: int = 0,
                 aws_profile: str | None = None):
        """Connect to Redis.

        Args:
            redis_url: Redis connection URL.
            db: Redis database number.
            aws_profile: AWS profile name (defaults to AWS_PROFILE env var or 'default').
        """
        try:
            import redis as _redis
        except ImportError:
            raise ImportError(
                "redis package required for type registry. Install with: pip install redis"
            )
        self._client = _redis.Redis.from_url(redis_url, db=db, decode_responses=True)
        self._profile = aws_profile or os.environ.get("AWS_PROFILE", "default")

    def register(self, signature: FunctionSignature) -> None:
        """Register a function signature in Redis."""
        key = _make_key(self._profile, signature.name)
        self._client.set(key, json.dumps(signature.to_dict()))

    def register_all(self, signatures: dict[str, FunctionSignature]) -> int:
        """Register multiple signatures. Returns count registered."""
        pipe = self._client.pipeline()
        for name, sig in signatures.items():
            pipe.set(_make_key(self._profile, name), json.dumps(sig.to_dict()))
        pipe.execute()
        return len(signatures)

    def lookup(self, func_name: str) -> FunctionSignature | None:
        """Look up a function's signature from Redis."""
        data = self._client.get(_make_key(self._profile, func_name))
        if data is None:
            return None
        return FunctionSignature.from_dict(json.loads(data))

    def lookup_all(self, func_names: list[str]) -> dict[str, FunctionSignature]:
        """Look up multiple functions. Returns dict of found signatures."""
        pipe = self._client.pipeline()
        for name in func_names:
            pipe.get(_make_key(self._profile, name))
        results = pipe.execute()

        signatures: dict[str, FunctionSignature] = {}
        for name, data in zip(func_names, results):
            if data is not None:
                signatures[name] = FunctionSignature.from_dict(json.loads(data))
        return signatures

    def list_all(self) -> dict[str, FunctionSignature]:
        """List all registered signatures for the current profile."""
        keys = self._client.keys(_make_key(self._profile, "*"))
        signatures: dict[str, FunctionSignature] = {}
        for key in keys:
            data = self._client.get(key)
            if data:
                sig = FunctionSignature.from_dict(json.loads(data))
                signatures[sig.name] = sig
        return signatures

    def remove(self, func_name: str) -> bool:
        """Remove a function from the registry."""
        return self._client.delete(_make_key(self._profile, func_name)) > 0

    def clear_all(self) -> int:
        """Clear all registered signatures for the current profile."""
        keys = self._client.keys(_make_key(self._profile, "*"))
        if keys:
            return self._client.delete(*keys)
        return 0

    @property
    def profile(self) -> str:
        return self._profile


# ─── DAG type validation ────────────────────────────────────────────────────────

@dataclass
class TypeCheckError:
    source_func: str
    target_func: str
    edge_type: str
    message: str
    is_warning: bool = False

    def __str__(self) -> str:
        icon = "\u26A0" if self.is_warning else "\u2717"
        return f"  {icon} {self.source_func} \u2192 {self.target_func} ({self.edge_type}): {self.message}"


def _types_compatible(output_type: str, input_type: str) -> bool:
    """Check if two simple types are compatible."""
    if output_type == "any" or input_type == "any":
        return True
    if output_type == input_type:
        return True
    # integer is a subtype of number
    if output_type == "integer" and input_type == "number":
        return True
    if output_type == "number" and input_type == "integer":
        return True
    return False


def validate_edge(
    source_sig: FunctionSignature,
    target_sig: FunctionSignature,
    edge_type: str,
) -> list[TypeCheckError]:
    """Validate type compatibility between two connected functions.

    For a Scalar edge: source.output fields must provide target.input fields with compatible types.
    For a Map edge: source.output elements are forwarded individually to target — same field check applies.
    For a Fan-in edge: multiple source outputs are collected into an array and forwarded to target —
        each element is a source output, so source.output fields should satisfy target.input.
    """
    errors: list[TypeCheckError] = []

    # All edge types perform the same field-level check:
    # target's @input fields must be present in source's @output with compatible types.
    # For Map: source output is split element-wise, each element goes to target.
    # For Fan-in: each source output becomes an element in the array passed to target.
    for field_name, expected_type in target_sig.input_params.items():
        if field_name not in source_sig.output_params:
            errors.append(TypeCheckError(
                source_func=source_sig.name,
                target_func=target_sig.name,
                edge_type=edge_type,
                message=f"target requires input field '{field_name}' ({expected_type}) "
                        f"but source output does not provide it. "
                        f"Source output fields: {list(source_sig.output_params.keys())}",
            ))
        else:
            actual_type = source_sig.output_params[field_name]
            if not _types_compatible(actual_type, expected_type):
                errors.append(TypeCheckError(
                    source_func=source_sig.name,
                    target_func=target_sig.name,
                    edge_type=edge_type,
                    message=f"field '{field_name}': source produces {actual_type}, "
                            f"target expects {expected_type}",
                ))

    return errors


def validate_workflow(
    configs: dict[str, dict],
    signatures: dict[str, FunctionSignature],
) -> list[TypeCheckError]:
    """Validate all edges in a compiled workflow against registered signatures.

    Args:
        configs: function_name → unum_config.json content
        signatures: function_name → FunctionSignature (from registry or extraction)

    Returns:
        List of type errors. Empty = all good.
    """
    errors: list[TypeCheckError] = []

    for func_name, config in configs.items():
        next_entries = config.get("Next")
        if next_entries is None:
            continue

        if isinstance(next_entries, dict):
            next_entries = [next_entries]

        source_sig = signatures.get(func_name)

        for entry in next_entries:
            target_name = entry.get("Name")
            if not target_name:
                continue

            target_sig = signatures.get(target_name)

            # Determine edge type
            raw_input_type = entry.get("InputType", "Scalar")
            if isinstance(raw_input_type, str):
                edge_type = raw_input_type
            elif isinstance(raw_input_type, dict) and "Fan-in" in raw_input_type:
                edge_type = "Fan-in"
            else:
                edge_type = "Scalar"

            # Skip validation if either side has no signature
            if source_sig is None and target_sig is None:
                continue

            if source_sig is None:
                errors.append(TypeCheckError(
                    source_func=func_name,
                    target_func=target_name,
                    edge_type=edge_type,
                    message=f"no type annotation for '{func_name}' — cannot validate",
                    is_warning=True,
                ))
                continue

            if target_sig is None:
                errors.append(TypeCheckError(
                    source_func=func_name,
                    target_func=target_name,
                    edge_type=edge_type,
                    message=f"no type annotation for '{target_name}' — cannot validate",
                    is_warning=True,
                ))
                continue

            # Validate the edge
            edge_errors = validate_edge(source_sig, target_sig, edge_type)
            errors.extend(edge_errors)

    return errors


def format_validation_report(
    errors: list[TypeCheckError],
    signatures: dict[str, FunctionSignature],
    configs: dict[str, dict],
) -> str:
    """Format a human-readable type validation report."""
    lines: list[str] = []

    total_funcs = len(configs)
    typed_funcs = len(signatures)

    lines.append(f"\n\033[33m\033[1mType Validation Report\033[0m\n")
    lines.append(f"  Functions with type annotations: {typed_funcs}/{total_funcs}")

    if typed_funcs == 0:
        lines.append(f"\n  \033[36mNo @input/@output annotations found in function source code.\033[0m")
        lines.append(f"  \033[36mAdd docstring annotations to enable compile-time type checking.\033[0m")
        lines.append(f"  \033[36mExample:\033[0m")
        lines.append(f"  \033[36m  def lambda_handler(event, context):\033[0m")
        lines.append(f"  \033[36m      '''@input {{bucket: string, key: string}}\033[0m")
        lines.append(f"  \033[36m      @output {{url: string, size: integer}}'''\033[0m\n")
        return "\n".join(lines)

    # Show registered signatures
    lines.append(f"\n  \033[36mRegistered signatures:\033[0m")
    for name, sig in signatures.items():
        lines.append(f"    {sig}")

    actual_errors = [e for e in errors if not e.is_warning]
    warnings = [e for e in errors if e.is_warning]

    if not actual_errors and not warnings:
        lines.append(f"\n  \033[32m\u2713 All typed edges are compatible!\033[0m\n")
        return "\n".join(lines)

    if actual_errors:
        lines.append(f"\n  \033[31m\u2717 {len(actual_errors)} type error(s):\033[0m\n")
        for err in actual_errors:
            lines.append(f"    \033[31m{err}\033[0m")

    if warnings:
        lines.append(f"\n  \033[33m{len(warnings)} warning(s):\033[0m\n")
        for w in warnings:
            lines.append(f"    \033[33m{w}\033[0m")

    lines.append("")
    return "\n".join(lines)
