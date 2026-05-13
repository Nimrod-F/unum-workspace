"""Tests for the unum type registry (docstring extraction + validation)."""

import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from type_registry import (
    FunctionSignature,
    extract_signature_from_python,
    extract_signature_from_javascript,
    extract_signature_from_file,
    extract_signature_from_directory,
    extract_all_signatures,
    validate_edge,
    validate_workflow,
    format_validation_report,
    _parse_type_annotation,
)


# ─── Helpers ────────────────────────────────────────────────────────────────────

_pass_count = 0
_fail_count = 0


def assert_eq(label, actual, expected):
    global _pass_count, _fail_count
    if actual != expected:
        print(f"  FAIL: {label}")
        print(f"    expected: {json.dumps(expected, indent=2, default=str)}")
        print(f"    actual:   {json.dumps(actual, indent=2, default=str)}")
        _fail_count += 1
        return False
    print(f"  PASS: {label}")
    _pass_count += 1
    return True


def assert_true(label, condition):
    global _pass_count, _fail_count
    if not condition:
        print(f"  FAIL: {label}")
        _fail_count += 1
        return False
    print(f"  PASS: {label}")
    _pass_count += 1
    return True


def _make_temp_dir():
    return tempfile.mkdtemp(prefix="unum_type_test_")


# ─── Test: Annotation parsing ───────────────────────────────────────────────────

def test_parse_annotation():
    print("\n=== Test: Parse type annotation body ===")
    fields = _parse_type_annotation("bucket: string, key: string")
    assert_eq("two string fields", fields, {"bucket": "string", "key": "string"})

    fields = _parse_type_annotation("count: int, name: str, active: bool")
    assert_eq("type aliases normalized", fields, {"count": "integer", "name": "string", "active": "boolean"})

    fields = _parse_type_annotation("data: float, items: list")
    assert_eq("float/list aliases", fields, {"data": "number", "items": "array"})

    fields = _parse_type_annotation("")
    assert_eq("empty body", fields, {})


# ─── Test: Python source extraction ────────────────────────────────────────────

def test_extract_python_basic():
    print("\n=== Test: Extract from Python source ===")
    source = '''
def lambda_handler(event, context):
    """Process an image.
    
    @input {bucket: string, key: string}
    @output {url: string, size: integer}
    """
    pass
'''
    sig = extract_signature_from_python(source, "ImageLoader")
    assert_true("signature extracted", sig is not None)
    assert_eq("function name", sig.name, "ImageLoader")
    assert_eq("input params", sig.input_params, {"bucket": "string", "key": "string"})
    assert_eq("output params", sig.output_params, {"url": "string", "size": "integer"})


def test_extract_python_no_annotations():
    print("\n=== Test: Extract from Python — no annotations ===")
    source = '''
def lambda_handler(event, context):
    """Just a regular docstring."""
    return event
'''
    sig = extract_signature_from_python(source, "PassThrough")
    assert_eq("no annotation returns None", sig, None)


def test_extract_python_single_line():
    print("\n=== Test: Extract from Python — single line annotations ===")
    source = '''
def lambda_handler(event, context):
    """@input {text: string} @output {count: integer}"""
    return {"count": len(event["text"])}
'''
    sig = extract_signature_from_python(source, "Counter")
    assert_true("extracted", sig is not None)
    assert_eq("input", sig.input_params, {"text": "string"})
    assert_eq("output", sig.output_params, {"count": "integer"})


# ─── Test: JavaScript source extraction ────────────────────────────────────────

def test_extract_javascript():
    print("\n=== Test: Extract from JavaScript source ===")
    source = '''
/**
 * Greet someone.
 * @input {name: string, age: integer}
 * @output {greeting: string, processed: boolean}
 */
function main(params) {
    return { greeting: "Hello " + params.name, processed: true };
}
'''
    sig = extract_signature_from_javascript(source, "Hello")
    assert_true("js signature extracted", sig is not None)
    assert_eq("js input", sig.input_params, {"name": "string", "age": "integer"})
    assert_eq("js output", sig.output_params, {"greeting": "string", "processed": "boolean"})


# ─── Test: File-based extraction ────────────────────────────────────────────────

def test_extract_from_file():
    print("\n=== Test: Extract from file ===")
    tmpdir = _make_temp_dir()
    try:
        py_file = os.path.join(tmpdir, "app.py")
        with open(py_file, "w") as f:
            f.write('''def lambda_handler(event, context):
    """@input {bucket: string, key: string}
    @output {result: object}"""
    pass
''')
        sig = extract_signature_from_file(py_file, "MyFunc")
        assert_true("file extraction works", sig is not None)
        assert_eq("input from file", sig.input_params, {"bucket": "string", "key": "string"})
        assert_eq("output from file", sig.output_params, {"result": "object"})
        assert_eq("source_file set", sig.source_file, py_file)
    finally:
        shutil.rmtree(tmpdir)


def test_extract_from_directory():
    print("\n=== Test: Extract from directory ===")
    tmpdir = _make_temp_dir()
    try:
        with open(os.path.join(tmpdir, "app.py"), "w") as f:
            f.write('"""@input {x: number} @output {y: number}"""\ndef lambda_handler(event, context): pass\n')
        sig = extract_signature_from_directory(tmpdir, "Compute")
        assert_true("directory extraction works", sig is not None)
        assert_eq("name", sig.name, "Compute")
    finally:
        shutil.rmtree(tmpdir)


def test_extract_from_directory_no_annotations():
    print("\n=== Test: Extract from directory — no annotations ===")
    tmpdir = _make_temp_dir()
    try:
        with open(os.path.join(tmpdir, "app.py"), "w") as f:
            f.write('def lambda_handler(event, context): return event\n')
        sig = extract_signature_from_directory(tmpdir, "Plain")
        assert_eq("no annotations returns None", sig, None)
    finally:
        shutil.rmtree(tmpdir)


# ─── Test: FunctionSignature serialization ──────────────────────────────────────

def test_signature_serialization():
    print("\n=== Test: Signature to_dict/from_dict round-trip ===")
    sig = FunctionSignature(
        name="Mapper",
        input_params={"text": "string", "bucket": "string"},
        output_params={"count": "integer"},
    )
    data = sig.to_dict()
    assert_eq("to_dict name", data["name"], "Mapper")
    assert_eq("to_dict input", data["input"], {"text": "string", "bucket": "string"})

    restored = FunctionSignature.from_dict(data)
    assert_eq("round-trip name", restored.name, "Mapper")
    assert_eq("round-trip input", restored.input_params, sig.input_params)
    assert_eq("round-trip output", restored.output_params, sig.output_params)


# ─── Test: Edge validation ──────────────────────────────────────────────────────

def test_validate_edge_compatible():
    print("\n=== Test: Edge validation — compatible ===")
    source = FunctionSignature("A", {}, {"bucket": "string", "key": "string"})
    target = FunctionSignature("B", {"bucket": "string", "key": "string"}, {})
    errors = validate_edge(source, target, "Scalar")
    assert_eq("compatible edge has no errors", len(errors), 0)


def test_validate_edge_missing_field():
    print("\n=== Test: Edge validation — missing required field ===")
    source = FunctionSignature("A", {}, {"bucket": "string"})
    target = FunctionSignature("B", {"bucket": "string", "key": "string"}, {})
    errors = validate_edge(source, target, "Scalar")
    assert_eq("1 error for missing field", len(errors), 1)
    assert_true("mentions 'key'", "'key'" in errors[0].message)


def test_validate_edge_type_mismatch():
    print("\n=== Test: Edge validation — type mismatch ===")
    source = FunctionSignature("A", {}, {"count": "string"})
    target = FunctionSignature("B", {"count": "integer"}, {})
    errors = validate_edge(source, target, "Scalar")
    assert_eq("1 error for type mismatch", len(errors), 1)
    assert_true("mentions types", "string" in errors[0].message and "integer" in errors[0].message)


def test_validate_edge_integer_number_compat():
    print("\n=== Test: Edge validation — integer/number compatible ===")
    source = FunctionSignature("A", {}, {"count": "integer"})
    target = FunctionSignature("B", {"count": "number"}, {})
    errors = validate_edge(source, target, "Scalar")
    assert_eq("integer → number is compatible", len(errors), 0)


def test_validate_edge_extra_output_ok():
    print("\n=== Test: Edge validation — extra output fields are fine ===")
    source = FunctionSignature("A", {}, {"x": "string", "y": "number", "z": "boolean"})
    target = FunctionSignature("B", {"x": "string"}, {})
    errors = validate_edge(source, target, "Scalar")
    assert_eq("extra fields don't cause errors", len(errors), 0)


# ─── Test: Workflow validation ──────────────────────────────────────────────────

def test_validate_workflow_compatible():
    print("\n=== Test: Workflow validation — all compatible ===")
    configs = {
        "A": {"Name": "A", "Start": True, "Next": {"Name": "B", "InputType": "Scalar"}},
        "B": {"Name": "B", "Start": False},
    }
    signatures = {
        "A": FunctionSignature("A", {"x": "string"}, {"result": "number"}),
        "B": FunctionSignature("B", {"result": "number"}, {"done": "boolean"}),
    }
    errors = validate_workflow(configs, signatures)
    actual_errors = [e for e in errors if not e.is_warning]
    assert_eq("no errors", len(actual_errors), 0)


def test_validate_workflow_mismatch():
    print("\n=== Test: Workflow validation — mismatch detected ===")
    configs = {
        "A": {"Name": "A", "Start": True, "Next": {"Name": "B", "InputType": "Scalar"}},
        "B": {"Name": "B"},
    }
    signatures = {
        "A": FunctionSignature("A", {}, {"name": "string"}),
        "B": FunctionSignature("B", {"count": "integer"}, {}),
    }
    errors = validate_workflow(configs, signatures)
    actual_errors = [e for e in errors if not e.is_warning]
    assert_eq("1 error: missing field", len(actual_errors), 1)


def test_validate_workflow_partial_coverage():
    print("\n=== Test: Workflow validation — partial type coverage ===")
    configs = {
        "A": {"Name": "A", "Start": True, "Next": {"Name": "B", "InputType": "Scalar"}},
        "B": {"Name": "B"},
    }
    signatures = {
        "A": FunctionSignature("A", {}, {"x": "string"}),
        # B has no signature
    }
    errors = validate_workflow(configs, signatures)
    warnings = [e for e in errors if e.is_warning]
    assert_true("warning for missing B", len(warnings) >= 1)


def test_validate_workflow_no_signatures():
    print("\n=== Test: Workflow validation — no signatures ===")
    configs = {
        "A": {"Name": "A", "Start": True, "Next": {"Name": "B", "InputType": "Scalar"}},
        "B": {"Name": "B"},
    }
    errors = validate_workflow(configs, {})
    assert_eq("no errors if no signatures", len(errors), 0)


# ─── Test: Report formatting ───────────────────────────────────────────────────

def test_report_no_annotations():
    print("\n=== Test: Report — no annotations found ===")
    report = format_validation_report([], {}, {"A": {"Name": "A"}})
    assert_true("mentions no annotations", "@input" in report or "annotation" in report.lower())


def test_report_all_pass():
    print("\n=== Test: Report — all pass ===")
    sigs = {"A": FunctionSignature("A", {"x": "string"}, {"y": "number"})}
    report = format_validation_report([], sigs, {"A": {"Name": "A"}})
    assert_true("mentions compatible", "compatible" in report.lower() or "\u2713" in report)


# ─── Test: extract_all_signatures (integration) ────────────────────────────────

def test_extract_all_signatures():
    print("\n=== Test: extract_all_signatures ===")
    tmpdir = _make_temp_dir()
    try:
        # Create two function directories
        func_a = os.path.join(tmpdir, "FuncA")
        func_b = os.path.join(tmpdir, "FuncB")
        func_c = os.path.join(tmpdir, "FuncC")
        os.makedirs(func_a)
        os.makedirs(func_b)
        os.makedirs(func_c)

        with open(os.path.join(func_a, "app.py"), "w") as f:
            f.write('"""@input {x: string} @output {y: number}"""\ndef lambda_handler(event, context): pass\n')
        with open(os.path.join(func_b, "app.py"), "w") as f:
            f.write('"""@input {y: number} @output {z: boolean}"""\ndef lambda_handler(event, context): pass\n')
        with open(os.path.join(func_c, "app.py"), "w") as f:
            f.write('def lambda_handler(event, context): return event\n')  # no annotations

        template = {
            "Functions": {
                "FuncA": {"Properties": {"CodeUri": f"{func_a}/"}},
                "FuncB": {"Properties": {"CodeUri": f"{func_b}/"}},
                "FuncC": {"Properties": {"CodeUri": f"{func_c}/"}},
            }
        }

        sigs = extract_all_signatures(template)
        assert_eq("extracted 2 signatures", len(sigs), 2)
        assert_true("FuncA found", "FuncA" in sigs)
        assert_true("FuncB found", "FuncB" in sigs)
        assert_true("FuncC not found (no annotation)", "FuncC" not in sigs)
    finally:
        shutil.rmtree(tmpdir)


# ─── Runner ─────────────────────────────────────────────────────────────────────

def main():
    global _pass_count, _fail_count

    test_parse_annotation()
    test_extract_python_basic()
    test_extract_python_no_annotations()
    test_extract_python_single_line()
    test_extract_javascript()
    test_extract_from_file()
    test_extract_from_directory()
    test_extract_from_directory_no_annotations()
    test_signature_serialization()
    test_validate_edge_compatible()
    test_validate_edge_missing_field()
    test_validate_edge_type_mismatch()
    test_validate_edge_integer_number_compat()
    test_validate_edge_extra_output_ok()
    test_validate_workflow_compatible()
    test_validate_workflow_mismatch()
    test_validate_workflow_partial_coverage()
    test_validate_workflow_no_signatures()
    test_report_no_annotations()
    test_report_all_pass()
    test_extract_all_signatures()

    print(f"\n{'=' * 60}")
    print(f"Results: {_pass_count} passed, {_fail_count} failed")
    if _fail_count > 0:
        print("\033[31mSOME TESTS FAILED\033[0m")
        sys.exit(1)
    else:
        print("\033[32mALL TESTS PASSED\033[0m")


if __name__ == "__main__":
    main()
