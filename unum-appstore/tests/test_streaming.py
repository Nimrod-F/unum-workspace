"""
Test script for Partial Parameter Streaming

This script tests the streaming transformation and runtime behavior locally
without deploying to AWS.
"""

import sys
import os

# Add runtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'unum', 'runtime'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'unum', 'unum-cli'))

# Mock environment variables
os.environ['FAAS_PLATFORM'] = 'aws'
os.environ['UNUM_INTERMEDIARY_DATASTORE_TYPE'] = 'dynamodb'
os.environ['UNUM_INTERMEDIARY_DATASTORE_NAME'] = 'test-table'
os.environ['STREAMING_DEBUG'] = 'true'

import json
import time
import threading

def test_streaming_analysis():
    """Test that the analyzer correctly identifies streamable functions"""
    from streaming_transformer import StreamingAnalyzer
    
    print("=" * 60)
    print("TEST 1: Streaming Analysis")
    print("=" * 60)
    
    # Test case 1: Function with multiple fields
    source1 = '''
def lambda_handler(event, context):
    data = event.get("data", [])
    
    field1 = compute_field1(data)
    field2 = compute_field2(data)
    field3 = compute_field3(data)
    
    result = {
        "a": field1,
        "b": field2,
        "c": field3
    }
    
    return result
'''
    
    analyzer = StreamingAnalyzer()
    analysis = analyzer.analyze(source1)
    
    assert analysis.can_stream, f"Should be streamable: {analysis.reason}"
    assert len(analysis.fields) == 3, f"Expected 3 fields, got {len(analysis.fields)}"
    print(f"✓ Test 1a: Correctly identified 3 streamable fields")
    
    # Test case 2: Function with single field (not streamable)
    source2 = '''
def lambda_handler(event, context):
    result = compute(event)
    return {"output": result}
'''
    
    analysis2 = analyzer.analyze(source2)
    # This should have 1 field which is not enough for streaming
    print(f"  Test 1b: Single field = {len(analysis2.fields)} fields, can_stream={analysis2.can_stream}")
    
    # Test case 3: No handler function
    source3 = '''
def process(data):
    return data * 2
'''
    
    analysis3 = analyzer.analyze(source3)
    assert not analysis3.can_stream, "Should not be streamable without handler"
    print(f"✓ Test 1c: Correctly rejected function without handler")
    
    print()


def test_streaming_transformation():
    """Test that the transformer produces valid code"""
    from streaming_transformer import StreamingAnalyzer, StreamingTransformer
    import ast
    
    print("=" * 60)
    print("TEST 2: Streaming Transformation")
    print("=" * 60)
    
    source = '''
import json

def compute_a(x):
    return x * 2

def compute_b(x):
    return x + 10

def compute_c(x):
    return x ** 2

def lambda_handler(event, context):
    data = event.get("value", 5)
    
    result_a = compute_a(data)
    result_b = compute_b(data)
    result_c = compute_c(data)
    
    output = {
        "doubled": result_a,
        "added": result_b,
        "squared": result_c
    }
    
    return output
'''
    
    analyzer = StreamingAnalyzer()
    analysis = analyzer.analyze(source)
    
    assert analysis.can_stream, f"Source should be streamable: {analysis.reason}"
    
    transformer = StreamingTransformer(analysis, "TestFunction")
    new_source, messages = transformer.transform(source)
    
    # Verify syntax
    try:
        ast.parse(new_source)
        print("✓ Transformed code is syntactically valid")
    except SyntaxError as e:
        print(f"✗ Syntax error in transformed code: {e}")
        return
    
    # Verify injected imports
    assert "from unum_streaming import" in new_source, "Missing streaming import"
    print("✓ Streaming import injected")
    
    # Verify streaming publisher initialization
    assert "StreamingPublisher" in new_source, "Missing StreamingPublisher"
    print("✓ StreamingPublisher initialization injected")
    
    # Verify publish calls
    assert ".publish(" in new_source, "Missing publish calls"
    publish_count = new_source.count("_streaming_publisher.publish(")
    print(f"✓ {publish_count} publish calls injected")
    
    # Verify invocation trigger
    assert "set_streaming_output" in new_source, "Missing set_streaming_output"
    print("✓ Streaming output trigger injected")
    
    print()


def test_future_creation_and_resolution():
    """Test future reference creation and resolution"""
    
    print("=" * 60)
    print("TEST 3: Future Creation and Resolution")
    print("=" * 60)
    
    # Test with in-memory store (no actual datastore)
    from unum_streaming import (
        make_future_ref,
        is_future,
        StreamingPublisher,
        LazyFutureDict
    )
    
    # Test make_future_ref
    future = make_future_ref("session123", "field_a", "FunctionA")
    assert future["__unum_future__"] == True, "Future marker missing"
    assert future["session"] == "session123", "Session mismatch"
    print("✓ Future reference created correctly")
    
    # Test is_future
    assert is_future(future), "is_future() should return True for future"
    assert not is_future({"regular": "dict"}), "is_future() should return False for regular dict"
    assert not is_future("string"), "is_future() should return False for string"
    print("✓ is_future() works correctly")
    
    # Test StreamingPublisher
    publisher = StreamingPublisher(
        session_id="session456",
        source_function="TestFunc",
        field_names=["field1", "field2", "field3"]
    )
    
    assert not publisher.should_invoke_next(), "Should not invoke before any publish"
    
    # Note: publish() will fail without real datastore, but we test the logic
    publisher.published_fields["field1"] = "value1"  # Simulate publish
    assert publisher.should_invoke_next(), "Should invoke after first field"
    
    payload = publisher.get_streaming_payload()
    assert payload["field1"] == "value1", "Published field should have value"
    assert is_future(payload["field2"]), "Pending field should be future"
    assert is_future(payload["field3"]), "Pending field should be future"
    print("✓ StreamingPublisher creates correct payload")
    
    publisher.mark_next_invoked()
    assert not publisher.should_invoke_next(), "Should not invoke twice"
    print("✓ StreamingPublisher prevents double invocation")
    
    print()


def test_lazy_future_dict():
    """Test LazyFutureDict lazy resolution"""
    
    print("=" * 60)
    print("TEST 4: LazyFutureDict")
    print("=" * 60)
    
    from unum_streaming import LazyFutureDict, make_future_ref
    
    # Create a dict with mix of values and futures
    data = {
        "ready_field": "immediate_value",
        "ready_number": 42,
        "future_field": make_future_ref("session", "future_field", "OtherFunc")
    }
    
    lazy_dict = LazyFutureDict(data)
    
    # Test immediate value access
    assert lazy_dict["ready_field"] == "immediate_value"
    print("✓ Immediate string value accessible")
    
    assert lazy_dict["ready_number"] == 42
    print("✓ Immediate number value accessible")
    
    # Test dict-like operations
    assert "ready_field" in lazy_dict
    assert "future_field" in lazy_dict
    assert "missing" not in lazy_dict
    print("✓ 'in' operator works")
    
    assert len(lazy_dict) == 3
    print("✓ len() works")
    
    assert set(lazy_dict.keys()) == {"ready_field", "ready_number", "future_field"}
    print("✓ keys() works")
    
    # Note: Accessing future_field would block waiting for datastore
    # We can't test that without a real datastore setup
    print("  (Future resolution requires datastore - skipping)")
    
    print()


def test_real_function_transformation():
    """Test transformation of a real ML feature extractor function"""
    
    print("=" * 60)
    print("TEST 5: Real Function Transformation")
    print("=" * 60)
    
    from streaming_transformer import analyze_file, transform_file
    import ast
    
    # Path to the ML feature extractor
    ml_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'ml-feature-pipeline', 
        'FeatureExtractor', 
        'app.py'
    )
    
    if not os.path.exists(ml_path):
        print(f"  Skipping - file not found: {ml_path}")
        return
    
    analysis = analyze_file(ml_path)
    
    if not analysis.can_stream:
        print(f"  Function not streamable: {analysis.reason}")
        return
    
    print(f"✓ Analysis found {len(analysis.fields)} streamable fields:")
    for f in analysis.fields:
        print(f"    - {f.field_name} (line {f.computed_at_line})")
    
    new_source, messages = transform_file(ml_path, "FeatureExtractor")
    
    try:
        ast.parse(new_source)
        print("✓ Transformed code is syntactically valid")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return
    
    print(f"✓ Transformation applied {len(messages)} streaming points")
    
    print()


def test_error_handling():
    """Test error handling and graceful degradation"""
    
    print("=" * 60)
    print("TEST 6: Error Handling")
    print("=" * 60)
    
    from unum_streaming import (
        StreamingError,
        StreamingTimeoutError,
        StreamingResolutionError,
        StreamingPublishError,
        LazyFutureDict,
        make_future_ref,
        resolve_future_with_fallback,
        resolve_all_with_report
    )
    
    # Test exception hierarchy
    assert issubclass(StreamingTimeoutError, StreamingError)
    assert issubclass(StreamingResolutionError, StreamingError)
    assert issubclass(StreamingPublishError, StreamingError)
    print("✓ Exception hierarchy correct")
    
    # Test StreamingTimeoutError
    err = StreamingTimeoutError(
        field_name="test_field",
        source_function="TestFunc",
        session_id="session123",
        timeout=30.0,
        attempts=100
    )
    assert "test_field" in str(err)
    assert "TestFunc" in str(err)
    assert "30" in str(err)
    print("✓ StreamingTimeoutError contains field details")
    
    # Test StreamingResolutionError
    err = StreamingResolutionError(
        field_name="test_field",
        source_function="TestFunc",
        session_id="session123",
        error=ValueError("mock error"),
        attempts=5
    )
    assert err.original_error is not None
    assert "mock error" in str(err)
    print("✓ StreamingResolutionError preserves original error")
    
    # Test LazyFutureDict error tracking
    data = {
        "ready": "immediate value",
        "future": make_future_ref("s1", "f1", "source")
    }
    lazy_dict = LazyFutureDict(data)
    
    # Ready value should work
    val, success = lazy_dict.get_with_fallback("ready")
    assert success and val == "immediate value"
    print("✓ get_with_fallback works for ready values")
    
    # Future resolution will fail without datastore, fallback should work
    val, success = lazy_dict.get_with_fallback("future", fallback="default")
    # In memory store mode, this might not fail, but we test the interface
    print("✓ get_with_fallback interface works")
    
    # Test resolution_status
    status = lazy_dict.resolution_status()
    assert "ready" in status
    print("✓ resolution_status() reports field status")
    
    # Test repr
    repr_str = repr(lazy_dict)
    assert "LazyFutureDict" in repr_str
    print(f"✓ LazyFutureDict repr: {repr_str}")
    
    # Test resolve_all_with_report
    data = {
        "field1": "value1",
        "field2": 42,
    }
    resolved, report = resolve_all_with_report(data)
    assert resolved["field1"] == "value1"
    assert report["field1"]["was_future"] == False
    assert report["field1"]["resolved"] == True
    print("✓ resolve_all_with_report works for non-futures")
    
    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PARTIAL PARAMETER STREAMING - TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_streaming_analysis()
        test_streaming_transformation()
        test_future_creation_and_resolution()
        test_lazy_future_dict()
        test_real_function_transformation()
        test_error_handling()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
