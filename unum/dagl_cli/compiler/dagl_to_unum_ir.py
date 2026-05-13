"""DAGL-U AST → Unum IR Code Generator

Walks a DAGL-U AST (produced by dagl_compiler.compile_dagl) and emits:
  1. One unum_config.json per function in the workflow
  2. A unum-template.yaml skeleton

Unum IR format (per function):
    {
        "Name": str,
        "Start": bool,
        "Checkpoint": bool,
        "Debug": bool,
        "Next": { "Name": str, "InputType": "Scalar" | "Map" | {"Fan-in": {"Values": [...]}} },
        "Next Payload Modifiers": ["Pop"]   (optional, for fan-in tail functions)
    }

Compilation strategy:
    The AST is a sequential block of statements. We walk top-to-bottom,
    tracking the *data flow* between function invocations to build the DAG.
    Each invocation becomes a node in the workflow graph. Edges are inferred
    from variable bindings (let x = FuncA() ... FuncB(x) means A→B).
"""

import json
import copy
from typing import Any

from dagl_cli.compiler.dagl_compiler import compile_dagl, node


# ─── Workflow Graph ─────────────────────────────────────────────────────────────

class FunctionNode:
    """Represents one function in the compiled workflow DAG."""
    def __init__(self, name: str):
        self.name = name
        self.start = False
        self.checkpoint = True
        self.debug = True
        self.next: list[dict] = []         # list of Next entries
        self.payload_modifiers: list[str] = []
        self.platform: str | None = None   # @aws, @gcp, @azure annotation
        self.is_map_entry = False          # synthetic UnumMapN entry
        self.is_fan_in_tail = False        # tail of a fan-in (needs Pop)
        self.remote = False                # imported remote function (already deployed)
        self.remote_arn: str | None = None # explicit ARN for remote functions

    def to_config(self) -> dict:
        cfg: dict[str, Any] = {
            "Name": self.name,
            "Start": self.start,
            "Checkpoint": self.checkpoint,
            "Debug": self.debug,
        }
        if len(self.next) == 1:
            cfg["Next"] = self.next[0]
        elif len(self.next) > 1:
            cfg["Next"] = self.next
        if self.payload_modifiers:
            cfg["Next Payload Modifiers"] = self.payload_modifiers
        if self.platform:
            cfg["Platform"] = self.platform
        return cfg


class WorkflowGraph:
    """Builds the Unum workflow DAG from DAGL-U AST."""

    def __init__(self):
        self.functions: dict[str, FunctionNode] = {}
        self.directives: dict[str, Any] = {}
        self._var_env: dict[str, str | list[str]] = {}  # var → func_name or [func_names]
        self._var_kind: dict[str, str] = {}              # var → "scalar" | "map" | "parallel" | "collect"
        self._map_counter = 0
        self._order: list[str] = []  # insertion order for tracking first function

    def _get_or_create(self, name: str) -> FunctionNode:
        if name not in self.functions:
            fn = FunctionNode(name)
            self.functions[name] = fn
            self._order.append(name)
        return self.functions[name]

    def _extract_func_name(self, ast_node: dict) -> str | None:
        """Extract function name from an invocation or platform-wrapped invocation."""
        if ast_node.get("data") == "invocation":
            return ast_node["children"][0]
        if ast_node.get("data") == "platform":
            inner = ast_node["children"][1]
            if inner.get("data") == "invocation":
                return inner["children"][0]
        return None

    def _extract_platform(self, ast_node: dict) -> str | None:
        if ast_node.get("data") == "platform":
            return ast_node["children"][0]
        return None

    def _extract_arg_vars(self, args_node: dict) -> list[str]:
        """Extract variable references from invocation arguments."""
        refs = []
        if args_node.get("data") == "dict":
            for pair in args_node.get("children", []):
                val = pair["children"][1] if len(pair.get("children", [])) > 1 else None
                if val and val.get("data") == "id":
                    refs.append(val["children"][0])
        elif args_node.get("data") == "list":
            for child in args_node.get("children", []):
                if child.get("data") == "id":
                    refs.append(child["children"][0])
        return refs

    def _get_invocation_args(self, ast_node: dict) -> dict | None:
        """Get the arguments node from an invocation."""
        if ast_node.get("data") == "invocation":
            return ast_node["children"][1] if len(ast_node["children"]) > 1 else None
        if ast_node.get("data") == "platform":
            inner = ast_node["children"][1]
            if inner.get("data") == "invocation":
                return inner["children"][1] if len(inner["children"]) > 1 else None
        return None

    # ── Main compilation entry ──────────────────────────────────────────────

    def compile(self, ast: dict) -> dict[str, dict]:
        """Compile a DAGL-U AST into Unum IR configs.

        Returns:
            dict mapping function_name → unum_config dict
        """
        if ast.get("data") != "block_expr":
            raise ValueError(f"Expected top-level block_expr, got {ast.get('data')}")

        stmts = ast.get("children", [])
        self._process_statements(stmts)
        self._resolve_start()
        return {name: self.functions[name].to_config() for name in self._order
                if not self.functions[name].is_map_entry or True}

    # ── Statement processing ────────────────────────────────────────────────

    def _process_statements(self, stmts: list[dict]):
        prev_func: str | None = None

        for stmt in stmts:
            data = stmt.get("data")

            if data == "directive":
                self._process_directive(stmt)

            elif data == "assign":
                prev_func = self._process_assign(stmt, prev_func)

            elif data == "return":
                self._process_return(stmt, prev_func)

            elif data == "invocation" or data == "platform":
                name = self._extract_func_name(stmt)
                if name:
                    fn = self._get_or_create(name)
                    plat = self._extract_platform(stmt)
                    if plat:
                        fn.platform = plat
                    # Link from previous if exists
                    if prev_func:
                        self._link_scalar(prev_func, name)
                    prev_func = name

            elif data == "collect":
                self._process_collect(stmt, prev_func)

            elif data == "list":
                # Parallel invocations at top level
                prev_func = self._process_parallel_list(stmt, prev_func)

    def _process_directive(self, stmt: dict):
        name = stmt["children"][0]
        value_node = stmt["children"][1]

        # Handle @import directive for remote functions
        if name == "import":
            self._process_import(value_node)
            return

        # Handle @api directive for API Gateway
        if name == "api":
            self._process_api(value_node)
            return

        # Handle @event directive for EventBridge
        if name == "event":
            self._process_event(value_node)
            return

        if value_node.get("data") == "string":
            self.directives[name] = value_node["children"][0]
        elif value_node.get("data") == "id":
            val = value_node["children"][0]
            self.directives[name] = True if val == "true" else (False if val == "false" else val)
        elif value_node.get("data") == "number":
            self.directives[name] = value_node["children"][0]
        else:
            self.directives[name] = value_node

    def _process_import(self, value_node: dict):
        """Process @import("FuncName") or @import("FuncName", "arn:aws:...").

        Marks the function as remote (already deployed elsewhere).
        """
        if value_node.get("data") == "string":
            # @import("FuncName") — no explicit ARN, resolved via registry
            func_name = value_node["children"][0]
            fn = self._get_or_create(func_name)
            fn.remote = True
        elif value_node.get("data") == "list":
            # @import("FuncName", "arn:aws:lambda:...") — explicit ARN
            children = value_node["children"]
            func_name = children[0]["children"][0] if children[0].get("data") == "string" else str(children[0])
            fn = self._get_or_create(func_name)
            fn.remote = True
            if len(children) > 1 and children[1].get("data") == "string":
                fn.remote_arn = children[1]["children"][0]
        else:
            raise ValueError(f"@import requires a string function name, got: {value_node}")

    def _process_api(self, value_node: dict):
        """Process @api("/path", "METHOD") directive for API Gateway.

        Stores api config in directives as: {"path": "/path", "method": "POST"}
        """
        if value_node.get("data") == "string":
            # @api("/path") — default to POST
            path = value_node["children"][0]
            self.directives["api"] = {"path": path, "method": "POST"}
        elif value_node.get("data") == "list":
            # @api("/path", "METHOD")
            children = value_node["children"]
            path = children[0]["children"][0] if children[0].get("data") == "string" else "/"
            method = "POST"
            if len(children) > 1 and children[1].get("data") == "string":
                method = children[1]["children"][0].upper()
            self.directives["api"] = {"path": path, "method": method}
        else:
            raise ValueError(f"@api requires a path string, got: {value_node}")

    def _process_event(self, value_node: dict):
        """Process @event("source", "detail-type") directive for EventBridge.

        Stores event config in directives as: {"source": "...", "detailType": "..."}
        """
        if value_node.get("data") == "string":
            # @event("source") — match all detail types from this source
            source = value_node["children"][0]
            self.directives["event"] = {"source": source}
        elif value_node.get("data") == "list":
            # @event("source", "detail-type")
            children = value_node["children"]
            source = children[0]["children"][0] if children[0].get("data") == "string" else "custom"
            detail_type = None
            if len(children) > 1 and children[1].get("data") == "string":
                detail_type = children[1]["children"][0]
            config = {"source": source}
            if detail_type:
                config["detailType"] = detail_type
            self.directives["event"] = config
        else:
            raise ValueError(f"@event requires a source string, got: {value_node}")

    def _process_assign(self, stmt: dict, prev_func: str | None) -> str | None:
        var_name = stmt["children"][0]["children"][0]
        expr = stmt["children"][1]

        # Case 1: let x = FuncA()  → scalar chain
        func_name = self._extract_func_name(expr)
        if func_name:
            fn = self._get_or_create(func_name)
            plat = self._extract_platform(expr)
            if plat:
                fn.platform = plat

            # Check if invocation args reference known variables
            args = self._get_invocation_args(expr)
            linked = False
            if args:
                arg_refs = self._extract_arg_vars(args)
                for ref in arg_refs:
                    if ref in self._var_env:
                        src = self._var_env[ref]
                        kind = self._var_kind.get(ref, "scalar")
                        if kind == "map" and isinstance(src, str):
                            # Map results → fan-in with wildcard
                            self._link_fan_in_wildcard(src, func_name)
                            linked = True
                        elif isinstance(src, list):
                            # Parallel branches → fan-in
                            self._link_fan_in(src, func_name)
                            linked = True
                        elif isinstance(src, str):
                            self._link_scalar(src, func_name)
                            linked = True

            # If no explicit arg refs link and we have a prev, chain sequentially
            if not linked:
                if prev_func and prev_func != func_name:
                    self._link_scalar(prev_func, func_name)

            self._var_env[var_name] = func_name
            self._var_kind[var_name] = "scalar"
            return func_name

        # Case 2: let x = map item in source { FuncB(item) }
        if expr.get("data") == "map_expr":
            return self._process_map_assign(var_name, expr, prev_func)

        # Case 3: let x = parallel { FuncA() FuncB() FuncC() }
        if expr.get("data") == "parallel":
            return self._process_parallel_assign(var_name, expr, prev_func)

        # Case 4: let x = [FuncA(), FuncB()]  → parallel
        if expr.get("data") == "list":
            invocations = [c for c in expr.get("children", []) if self._extract_func_name(c)]
            if invocations:
                return self._process_parallel_assign(var_name, node("parallel", invocations), prev_func)

        # Case 5: Non-invocation assignment (local computation) - just store
        self._var_env[var_name] = prev_func if prev_func else var_name
        self._var_kind[var_name] = "scalar"
        return prev_func

    def _process_map_assign(self, var_name: str, map_node: dict, prev_func: str | None) -> str | None:
        """Handle: let x = map item in source { return Func(item) }"""
        _iter_var = map_node["children"][0]      # "item"
        iterable = map_node["children"][1]        # source expression
        body = map_node["children"][2]            # block or expression

        # Find the function invoked in the body
        body_func = self._find_invocation_in_body(body)
        if not body_func:
            self._var_env[var_name] = prev_func if prev_func else var_name
            self._var_kind[var_name] = "scalar"
            return prev_func

        fn = self._get_or_create(body_func)

        # Determine the source of the iterable
        source_func = None
        if iterable.get("data") == "id":
            ref = iterable["children"][0]
            if ref in self._var_env:
                source_func = self._var_env[ref]
                if isinstance(source_func, list):
                    source_func = source_func[0]  # take first for now
            elif ref == "input":
                # Top-level input → this is the first map, needs UnumMap entry
                source_func = None
        
        if source_func:
            # Create synthetic UnumMap entry if it's from a scalar func
            if self._var_kind.get(iterable["children"][0], "scalar") != "map":
                # Source is a scalar function → needs Map InputType
                self._link_map(source_func, body_func)
            else:
                self._link_map(source_func, body_func)
        else:
            # No explicit source → create synthetic entry point
            entry_name = f"UnumMap{self._map_counter}"
            self._map_counter += 1
            entry = self._get_or_create(entry_name)
            entry.is_map_entry = True
            entry.start = True
            entry.next.append({"Name": body_func, "InputType": "Map"})
            if prev_func and prev_func != entry_name:
                self._link_scalar(prev_func, entry_name)

        # The variable now holds the results of the map → list of func outputs
        self._var_env[var_name] = body_func
        self._var_kind[var_name] = "map"
        return body_func

    def _process_parallel_assign(self, var_name: str, par_node: dict, prev_func: str | None) -> str | None:
        """Handle: let x = parallel { FuncA() FuncB() FuncC() } or let x = [FuncA(), FuncB()]"""
        func_names = []
        for child in par_node.get("children", []):
            name = self._extract_func_name(child)
            if name:
                fn = self._get_or_create(name)
                plat = self._extract_platform(child)
                if plat:
                    fn.platform = plat
                func_names.append(name)

                # Link from prev → each parallel branch
                if prev_func:
                    self._link_scalar(prev_func, name)

        self._var_env[var_name] = func_names
        self._var_kind[var_name] = "parallel"
        return None  # No single "prev" after parallel

    def _process_parallel_list(self, list_node: dict, prev_func: str | None) -> str | None:
        """Handle top-level parallel invocations (auto-detected)."""
        for child in list_node.get("children", []):
            name = self._extract_func_name(child)
            if name:
                fn = self._get_or_create(name)
                if prev_func:
                    self._link_scalar(prev_func, name)
        return None

    def _process_collect(self, stmt: dict, prev_func: str | None):
        """Handle: collect <source> into <target>"""
        source = stmt["children"][0]
        target = stmt["children"][1]

        target_name = self._extract_func_name(target)
        if not target_name:
            return

        self._get_or_create(target_name)

        # Source should be a variable referencing parallel branches or map results
        source_funcs = []
        if source.get("data") == "id":
            ref = source["children"][0]
            if ref in self._var_env:
                val = self._var_env[ref]
                if isinstance(val, list):
                    source_funcs = val
                elif isinstance(val, str):
                    source_funcs = [val]

        if source_funcs:
            self._link_fan_in(source_funcs, target_name)

    def _process_return(self, stmt: dict, prev_func: str | None):
        """Handle: return FuncX(...)"""
        expr = stmt["children"][0]
        func_name = self._extract_func_name(expr)
        if func_name:
            fn = self._get_or_create(func_name)
            plat = self._extract_platform(expr)
            if plat:
                fn.platform = plat
            # Check arg refs
            args = self._get_invocation_args(expr)
            if args:
                arg_refs = self._extract_arg_vars(args)
                linked = False
                for ref in arg_refs:
                    if ref in self._var_env:
                        src = self._var_env[ref]
                        kind = self._var_kind.get(ref, "scalar")
                        if kind == "map":
                            # The map results feed into this function as fan-in
                            if isinstance(src, str):
                                self._link_fan_in_wildcard(src, func_name)
                            linked = True
                        elif isinstance(src, list):
                            self._link_fan_in(src, func_name)
                            linked = True
                        elif isinstance(src, str):
                            self._link_scalar(src, func_name)
                            linked = True
                if not linked and prev_func:
                    self._link_scalar(prev_func, func_name)
            elif prev_func:
                self._link_scalar(prev_func, func_name)

    # ── Linking helpers ─────────────────────────────────────────────────────

    def _link_scalar(self, src: str, dst: str):
        fn = self._get_or_create(src)
        # Avoid duplicate links
        for n in fn.next:
            if n.get("Name") == dst and n.get("InputType") == "Scalar":
                return
        fn.next.append({"Name": dst, "InputType": "Scalar"})

    def _link_map(self, src: str, dst: str):
        fn = self._get_or_create(src)
        for n in fn.next:
            if n.get("Name") == dst and n.get("InputType") == "Map":
                return
        fn.next.append({"Name": dst, "InputType": "Map"})

    def _link_fan_in(self, sources: list[str], dst: str):
        """Link multiple source functions to a fan-in target."""
        values = [f"{s}-unumIndex-{i}" for i, s in enumerate(sources)]
        fan_in_next = {
            "Name": dst,
            "InputType": {"Fan-in": {"Values": values}},
            "Fan-in-Group": True,
        }
        for src in sources:
            fn = self._get_or_create(src)
            # Check if already linked
            already = False
            for n in fn.next:
                if n.get("Name") == dst:
                    already = True
                    break
            if not already:
                fn.next.append(copy.deepcopy(fan_in_next))
                fn.payload_modifiers = ["Pop"]
                fn.is_fan_in_tail = True

    def _link_fan_in_wildcard(self, src: str, dst: str):
        """Link a map's body function to a fan-in target using wildcard."""
        fan_in_next = {
            "Name": dst,
            "InputType": {"Fan-in": {"Values": [f"{src}-unumIndex-*"]}},
        }
        fn = self._get_or_create(src)
        for n in fn.next:
            if n.get("Name") == dst:
                return
        fn.next.append(fan_in_next)
        fn.payload_modifiers = ["Pop"]
        fn.is_fan_in_tail = True

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _find_invocation_in_body(self, body: dict) -> str | None:
        """Recursively find the first function invocation in a block body."""
        data = body.get("data")
        if data == "invocation":
            return body["children"][0]
        if data == "platform":
            inner = body["children"][1]
            if inner.get("data") == "invocation":
                name = inner["children"][0]
                fn = self._get_or_create(name)
                fn.platform = body["children"][0]
                return name
        if data == "return":
            return self._find_invocation_in_body(body["children"][0])
        if data in ("block_expr", "if_expr"):
            for child in body.get("children", []):
                result = self._find_invocation_in_body(child)
                if result:
                    return result
        return None

    def _resolve_start(self):
        """Mark the first function as Start if none is marked."""
        has_start = any(fn.start for fn in self.functions.values())
        if not has_start and self._order:
            self.functions[self._order[0]].start = True


# ─── Template generation ────────────────────────────────────────────────────────

def generate_unum_template(graph: WorkflowGraph) -> dict:
    """Generate a unum-template.yaml dict from the compiled workflow graph."""
    tmpl: dict[str, Any] = {
        "Globals": {
            "ApplicationName": graph.directives.get("workflow", "my-workflow"),
            "WorkflowType": "dagl",
            "UnumIntermediaryDataStoreType": "dynamodb",
            "UnumIntermediaryDataStoreName": "unum-intermediate-datastore",
            "FaaSPlatform": "aws",
            "Checkpoint": graph.directives.get("checkpoint", True),
            "GC": False,
            "Debug": False,
            "Eager": graph.directives.get("eager", False),
        },
        "Functions": {},
    }

    # Add API Gateway config if @api directive present
    if "api" in graph.directives:
        tmpl["Globals"]["Api"] = graph.directives["api"]

    # Add EventBridge config if @event directive present
    if "event" in graph.directives:
        tmpl["Globals"]["Event"] = graph.directives["event"]

    for name, fn in graph.functions.items():
        func_entry: dict[str, Any] = {
            "Properties": {
                "CodeUri": f"{name}/",
                "Runtime": "python3.11",
            }
        }
        if fn.start:
            func_entry["Properties"]["Start"] = True
        if fn.platform:
            func_entry["Platform"] = fn.platform
        if fn.remote:
            func_entry["Remote"] = True
            if fn.remote_arn:
                func_entry["Arn"] = fn.remote_arn
        tmpl["Functions"][name] = func_entry

    return tmpl


# ─── Public API ─────────────────────────────────────────────────────────────────

def compile_dagl_to_unum_ir(source: str) -> tuple[dict[str, dict], dict]:
    """Compile DAGL-U source code to Unum IR configs.

    Args:
        source: DAGL-U source code string

    Returns:
        Tuple of (configs, template) where:
        - configs: dict mapping function_name → unum_config.json content
        - template: unum-template.yaml content as dict
    """
    ast = compile_dagl(source)
    graph = WorkflowGraph()
    configs = graph.compile(ast)
    template = generate_unum_template(graph)
    return configs, template
