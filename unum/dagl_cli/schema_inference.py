"""DAGL Schema Inference — Automatically infer input/output schemas from function source code.

Supports:
  1. Explicit annotations: @input/@output in docstrings (Python) or JSDoc (Node.js)
  2. Code analysis: field access patterns (event.get(), event["key"], input.field, destructuring)
  3. Type inference from defaults, operations, and naming conventions

Priority: explicit annotations > code analysis
"""

import ast
import re
import json


# ── Type inference helpers ───────────────────────────────────────────────────

# Map Python default values to schema types
_PY_DEFAULT_TYPE = {
    str: "string",
    int: "number",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# Naming conventions → likely types
_NAME_HINTS = {
    "count": "number", "total": "number", "size": "number", "length": "number",
    "num": "number", "amount": "number", "price": "number", "score": "number",
    "index": "number", "offset": "number", "limit": "number", "age": "number",
    "width": "number", "height": "number", "weight": "number", "timeout": "number",
    "text": "string", "name": "string", "title": "string", "message": "string",
    "description": "string", "label": "string", "key": "string", "value": "string",
    "url": "string", "path": "string", "email": "string", "id": "string",
    "language": "string", "lang": "string", "report": "string", "summary": "string",
    "enabled": "boolean", "active": "boolean", "valid": "boolean", "flag": "boolean",
    "is_active": "boolean", "is_valid": "boolean", "debug": "boolean",
    "words": "array", "tokens": "array", "items": "array", "results": "array",
    "list": "array", "entries": "array", "records": "array", "data": "array",
    "stats": "object", "config": "object", "options": "object", "metadata": "object",
    "settings": "object", "response": "object", "result": "object",
}


def _guess_type_from_name(name):
    """Guess type from field name using conventions."""
    lower = name.lower()
    # Check exact match first
    if lower in _NAME_HINTS:
        return _NAME_HINTS[lower]
    # Check prefix patterns first (higher priority — "totalWords" → number, not array)
    for prefix, typ in [("is_", "boolean"), ("is", "boolean"),
                        ("has_", "boolean"), ("has", "boolean"),
                        ("can_", "boolean"), ("can", "boolean"),
                        ("should_", "boolean"), ("should", "boolean"),
                        ("total", "number"), ("num_", "number"),
                        ("num", "number"), ("unique", "number"),
                        ("max", "number"), ("min", "number"),
                        ("avg", "number")]:
        if lower.startswith(prefix):
            return typ
    # Check suffix patterns (case-insensitive)
    for suffix, typ in [("count", "number"), ("total", "number"), ("size", "number"),
                        ("length", "number"), ("num", "number"), ("index", "number"),
                        ("amount", "number"), ("score", "number"),
                        ("words", "array"), ("tokens", "array"),
                        ("list", "array"), ("items", "array"),
                        ("name", "string"), ("text", "string"),
                        ("id", "string"), ("url", "string"),
                        ("flag", "boolean")]:
        if lower.endswith(suffix):
            return typ
    return None


# ── Python Analysis (AST-based) ─────────────────────────────────────────────

def _infer_python(source, handler_name="handler"):
    """Infer input/output schemas from Python source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, None

    # Find the handler function
    handler_fn = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in (handler_name, "lambda_handler"):
                handler_fn = node
                break

    if not handler_fn:
        return None, None

    # 1. Check for @input/@output annotations in docstring
    docstring = ast.get_docstring(handler_fn)
    if docstring:
        explicit_in, explicit_out = _parse_annotations(docstring)
        if explicit_in or explicit_out:
            return explicit_in, explicit_out

    # 2. Analyze code patterns
    # Find the event parameter name (first param)
    event_param = None
    if handler_fn.args.args:
        event_param = handler_fn.args.args[0].arg

    if not event_param:
        return None, None

    input_schema = _analyze_python_input(handler_fn, event_param)
    output_schema = _analyze_python_output(handler_fn)

    return input_schema or None, output_schema or None


def _analyze_python_input(fn_node, event_param):
    """Analyze event.get("key", default) and event["key"] patterns."""
    schema = {}

    for node in ast.walk(fn_node):
        # Pattern: event.get("key", default)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == event_param
            and node.args):

            key_node = node.args[0]
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                field_name = key_node.value
                field_type = None

                # Infer type from default value
                if len(node.args) >= 2:
                    default = node.args[1]
                    field_type = _python_const_type(default)

                # Fall back to name heuristic
                if not field_type:
                    field_type = _guess_type_from_name(field_name)

                schema[field_name] = field_type or "any"

        # Pattern: event["key"]
        elif (isinstance(node, ast.Subscript)
              and isinstance(node.value, ast.Name)
              and node.value.id == event_param):
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                field_name = slice_node.value
                field_type = _guess_type_from_name(field_name) or "any"
                schema[field_name] = field_type

    return schema


def _python_const_type(node):
    """Determine schema type from a Python AST constant/literal."""
    if isinstance(node, ast.Constant):
        return _PY_DEFAULT_TYPE.get(type(node.value))
    if isinstance(node, ast.List):
        return "array"
    if isinstance(node, ast.Dict):
        return "object"
    if isinstance(node, ast.NameConstant):  # Python 3.7
        if isinstance(node.value, bool):
            return "boolean"
    return None


def _analyze_python_output(fn_node):
    """Analyze return statements for output schema."""
    schema = {}

    for node in ast.walk(fn_node):
        if isinstance(node, ast.Return) and node.value:
            ret_val = node.value
            if isinstance(ret_val, ast.Dict):
                for key, value in zip(ret_val.keys, ret_val.values):
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        field_name = key.value
                        field_type = _infer_python_expr_type(value)
                        if not field_type:
                            field_type = _guess_type_from_name(field_name)
                        schema[field_name] = field_type or "any"

    return schema


def _infer_python_expr_type(node):
    """Infer type from a Python expression."""
    if isinstance(node, ast.Constant):
        return _PY_DEFAULT_TYPE.get(type(node.value))
    if isinstance(node, ast.List) or isinstance(node, ast.ListComp):
        return "array"
    if isinstance(node, ast.Dict) or isinstance(node, ast.DictComp):
        return "object"
    if isinstance(node, ast.JoinedStr):  # f-string
        return "string"
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in ("str", "repr", "format"):
                return "string"
            if node.func.id in ("int", "float", "len", "sum", "min", "max", "abs", "round"):
                return "number"
            if node.func.id in ("list", "sorted", "reversed"):
                return "array"
            if node.func.id in ("dict",):
                return "object"
            if node.func.id in ("bool",):
                return "boolean"
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr == "join":
                return "string"
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            # Check if string concatenation
            if isinstance(node.left, (ast.Constant,)) and isinstance(node.left.value, str):
                return "string"
            return "number"
    return None


# ── Node.js Analysis (regex-based) ──────────────────────────────────────────

def _infer_nodejs(source):
    """Infer input/output schemas from Node.js source code."""

    # 1. Check for JSDoc @input/@output or @param/@returns
    jsdoc = _extract_jsdoc(source)
    if jsdoc:
        explicit_in, explicit_out = _parse_annotations(jsdoc)
        if explicit_in or explicit_out:
            return explicit_in, explicit_out

    # 2. Find the event/input parameter name
    param_name = _find_js_handler_param(source)
    if not param_name:
        return None, None

    input_schema = _analyze_js_input(source, param_name)
    output_schema = _analyze_js_output(source)

    return input_schema or None, output_schema or None


def _find_js_handler_param(source):
    """Find the first parameter of the handler function."""
    # exports.handler = async (input, context) =>
    # exports.handler = async function(event, context)
    # module.exports.handler = function handler(event, context)
    patterns = [
        r'exports\.handler\s*=\s*(?:async\s+)?(?:function\s*\w*)?\s*\(\s*(\w+)',
        r'handler\s*=\s*(?:async\s+)?(?:function\s*\w*)?\s*\(\s*(\w+)',
        r'module\.exports\s*=\s*(?:async\s+)?(?:function\s*\w*)?\s*\(\s*(\w+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, source)
        if m:
            return m.group(1)
    return None


def _extract_jsdoc(source):
    """Extract JSDoc comment block preceding the handler."""
    # Find /** ... */ blocks
    blocks = re.findall(r'/\*\*(.*?)\*/', source, re.DOTALL)
    # Return the one closest to 'handler' or 'exports'
    for block in blocks:
        cleaned = re.sub(r'^\s*\*\s?', '', block, flags=re.MULTILINE).strip()
        if cleaned:
            return cleaned
    return None


def _analyze_js_input(source, param_name):
    """Analyze input.field and destructuring patterns."""
    schema = {}

    # Pattern: input.field (property access)
    for m in re.finditer(rf'\b{re.escape(param_name)}\.(\w+)', source):
        field = m.group(1)
        # Skip if it's a method call like input.toString()
        pos = m.end()
        if pos < len(source) and source[pos] == '(':
            continue
        if field in schema:
            continue

        # Try to infer type from surrounding context
        field_type = _infer_js_field_type(source, param_name, field)
        if not field_type:
            field_type = _guess_type_from_name(field)
        schema[field] = field_type or "any"

    # Pattern: const { field1, field2 } = input
    destr_pattern = rf'(?:const|let|var)\s*\{{([^}}]+)\}}\s*=\s*{re.escape(param_name)}'
    for m in re.finditer(destr_pattern, source):
        fields_str = m.group(1)
        for field in re.findall(r'(\w+)', fields_str):
            if field not in schema:
                field_type = _guess_type_from_name(field)
                schema[field] = field_type or "any"

    return schema


def _infer_js_field_type(source, param_name, field):
    """Infer type of a JS field from usage context."""
    # Pattern: input.field || '' or input.field || ""  → string
    pattern = rf'{re.escape(param_name)}\.{re.escape(field)}\s*\|\|\s*[\'"]'
    if re.search(pattern, source):
        return "string"

    # Pattern: input.field || []  → array
    pattern = rf'{re.escape(param_name)}\.{re.escape(field)}\s*\|\|\s*\['
    if re.search(pattern, source):
        return "array"

    # Pattern: input.field || {{}}  → object
    pattern = rf'{re.escape(param_name)}\.{re.escape(field)}\s*\|\|\s*\{{'
    if re.search(pattern, source):
        return "object"

    # Pattern: input.field || 0 or input.field || -1  → number
    pattern = rf'{re.escape(param_name)}\.{re.escape(field)}\s*\|\|\s*-?\d'
    if re.search(pattern, source):
        return "number"

    # Pattern: input.field || false/true  → boolean
    pattern = rf'{re.escape(param_name)}\.{re.escape(field)}\s*\|\|\s*(?:false|true)'
    if re.search(pattern, source):
        return "boolean"

    # Array method usage: input.field.map/filter/forEach/slice  → array
    pattern = rf'{re.escape(param_name)}\.{re.escape(field)}(?:\s*\|\|[^)]+\))?\s*\.(?:map|filter|forEach|slice|reduce|find|some|every|flat|join)\s*\('
    if re.search(pattern, source):
        return "array"

    # String method: .split/.trim/.toLowerCase  → string
    pattern = rf'{re.escape(param_name)}\.{re.escape(field)}\s*\.(?:split|trim|toLowerCase|toUpperCase|replace|match|includes|startsWith)\s*\('
    if re.search(pattern, source):
        return "string"

    # Template literal usage: `${input.field}` — don't force string, use name hint
    # (numbers are commonly interpolated too)
    pattern = rf'\$\{{\s*{re.escape(param_name)}\.{re.escape(field)}\s*\}}'
    if re.search(pattern, source):
        return _guess_type_from_name(field)

    return None


def _analyze_js_output(source):
    """Analyze return { ... } statements."""
    schema = {}

    # Find return { key: value, ... } blocks — match balanced braces
    for m in re.finditer(r'return\s*\{', source):
        start = m.end()
        block = _extract_balanced_braces(source, start)
        if not block:
            continue
        # Extract top-level key: value pairs (skip nested objects)
        for field_match in re.finditer(r'^\s*(\w+)\s*:', block, re.MULTILINE):
            field = field_match.group(1)
            if field in schema:
                continue

            # Get the value part after the colon
            rest = block[field_match.end():]
            value_type = _infer_js_return_value_type(rest.strip(), field)
            schema[field] = value_type or "any"

    return schema


def _extract_balanced_braces(source, start):
    """Extract content between balanced { } starting after the opening brace."""
    depth = 1
    i = start
    while i < len(source) and depth > 0:
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return source[start:i-1]
    return None


def _infer_js_return_value_type(value_str, field_name):
    """Infer type of a return value from its expression."""
    value_str = value_str.split(',')[0].strip().rstrip(',')

    # Template literal: `...`
    if value_str.startswith('`'):
        return "string"
    # String literal
    if value_str.startswith(("'", '"')):
        return "string"
    # Array literal or method that returns array
    if value_str.startswith('[') or '.map(' in value_str or '.filter(' in value_str:
        return "array"
    # Object literal
    if value_str.startswith('{'):
        return "object"
    # Number literal
    if re.match(r'^-?\d+(\.\d+)?$', value_str):
        return "number"
    # Boolean literal
    if value_str in ('true', 'false'):
        return "boolean"
    # .length property → number
    if value_str.endswith('.length'):
        return "number"

    # Fall back to name heuristic
    return _guess_type_from_name(field_name)


# ── Shared: Parse @input/@output annotations ────────────────────────────────

def _parse_annotations(text):
    """Parse @input/@output annotations from docstring or JSDoc.

    Supported formats:
      @input {text: string, count: number}
      @output {words: array, count: integer}
      @param {Object} event
      @param {string} event.text - The input text
      @returns {Object} - {words: array, count: number}
    """
    input_schema = {}
    output_schema = {}

    # Format 1: @input {key: type, ...}
    m = re.search(r'@input\s*\{([^}]+)\}', text)
    if m:
        input_schema = _parse_field_list(m.group(1))

    # Format 2: @output {key: type, ...}
    m = re.search(r'@output\s*\{([^}]+)\}', text)
    if m:
        output_schema = _parse_field_list(m.group(1))

    # Format 3: JSDoc @param {type} event.field
    for m in re.finditer(r'@param\s*\{(\w+)\}\s*\w+\.(\w+)', text):
        typ = m.group(1).lower()
        field = m.group(2)
        input_schema[field] = _normalize_jsdoc_type(typ)

    # Format 4: @returns {Object} - {key: type, ...}
    m = re.search(r'@returns?\s*(?:\{[^}]*\}\s*-?\s*)?\{([^}]+)\}', text)
    if m and not output_schema:
        output_schema = _parse_field_list(m.group(1))

    return input_schema or None, output_schema or None


def _parse_field_list(text):
    """Parse 'key: type, key: type' into a dict."""
    schema = {}
    for m in re.finditer(r'(\w+)\s*:\s*(\w+(?:\[\])?)', text):
        field = m.group(1)
        typ = _normalize_jsdoc_type(m.group(2))
        schema[field] = typ
    return schema


def _normalize_jsdoc_type(t):
    """Normalize type names to our schema type system."""
    t = t.lower().strip()
    aliases = {
        "str": "string", "int": "number", "integer": "number",
        "float": "number", "double": "number", "bool": "boolean",
        "dict": "object", "list": "array", "map": "object",
        "string[]": "string[]", "number[]": "number[]",
    }
    return aliases.get(t, t)


# ── Public API ───────────────────────────────────────────────────────────────

def infer_schemas_from_source(source, runtime_family, handler_name=None):
    """Infer input/output schemas from function source code.

    Args:
        source: The source code string
        runtime_family: "python" or "nodejs"
        handler_name: Override handler function name (default: "handler")

    Returns:
        (input_schema, output_schema) — each is a dict or None
    """
    if runtime_family == "python":
        return _infer_python(source, handler_name or "handler")
    elif runtime_family == "nodejs":
        return _infer_nodejs(source)
    return None, None


def infer_schemas_from_lambda(lambda_client, function_name, region=None):
    """Download a Lambda function's code and infer schemas.

    Args:
        lambda_client: boto3 Lambda client
        function_name: Lambda function name or ARN

    Returns:
        (input_schema, output_schema, runtime_family) — schemas are dicts or None
    """
    from urllib.request import urlopen

    # Get function info
    resp = lambda_client.get_function(FunctionName=function_name)
    runtime = resp["Configuration"].get("Runtime", "")
    handler_str = resp["Configuration"]["Handler"]
    code_url = resp["Code"]["Location"]

    # Determine runtime family
    if runtime.startswith("python"):
        runtime_family = "python"
    elif runtime.startswith("nodejs"):
        runtime_family = "nodejs"
    else:
        return None, None, None

    # Parse handler to find module and function name
    if runtime_family == "python":
        # handler = "module.submodule.function_name"
        parts = handler_str.split(".")
        handler_func = parts[-1] if parts else "handler"
        module_path = "/".join(parts[:-1]) + ".py" if len(parts) > 1 else parts[0] + ".py"
    else:
        # handler = "module.function_name"
        parts = handler_str.split(".")
        handler_func = parts[-1] if parts else "handler"
        module_path = parts[0] + ".js" if parts else "index.js"

    # Skip DAGL wrapper — look at original handler
    env = resp["Configuration"].get("Environment", {}).get("Variables", {})
    if "DAGL_USER_HANDLER" in env:
        handler_str = env["DAGL_USER_HANDLER"]
        parts = handler_str.split(".")
        handler_func = parts[-1] if parts else "handler"
        if runtime_family == "python":
            module_path = "/".join(parts[:-1]) + ".py" if len(parts) > 1 else parts[0] + ".py"
        else:
            module_path = parts[0] + ".js" if parts else "index.js"

    # Download and extract the code
    import zipfile
    import io

    with urlopen(code_url) as r:
        zip_data = r.read()

    with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
        # Find the handler file
        source = None
        for name in zf.namelist():
            if name == module_path or name.endswith("/" + module_path):
                source = zf.read(name).decode("utf-8")
                break

        if not source:
            # Try common alternatives
            alternatives = []
            if runtime_family == "python":
                alternatives = ["app.py", "lambda_function.py", "index.py", "main.py"]
            else:
                alternatives = ["index.js", "app.js", "main.js"]
            for alt in alternatives:
                if alt in zf.namelist():
                    source = zf.read(alt).decode("utf-8")
                    break

    if not source:
        return None, None, runtime_family

    input_schema, output_schema = infer_schemas_from_source(
        source, runtime_family, handler_func
    )

    return input_schema, output_schema, runtime_family
