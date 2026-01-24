"""
Unit tests for Hello-World Unum app
Tests business logic without full Unum orchestration
"""
import sys
sys.path.insert(0, '../hello')
sys.path.insert(0, '../world')

from hello.app import lambda_handler as hello_handler
from world.app import lambda_handler as world_handler


def test_hello_function():
    """Test Hello function returns correct output"""
    result = hello_handler({}, None)
    assert result == "Hello"
    print("✓ Hello function test passed")


def test_world_function():
    """Test World function concatenates correctly"""
    event = "Hello"
    result = world_handler(event, None)
    assert result == "Hello world!"
    print("✓ World function test passed")


def test_workflow():
    """Test complete workflow logic"""
    # Step 1: Hello
    hello_output = hello_handler({}, None)
    
    # Step 2: World with Hello's output
    final_output = world_handler(hello_output, None)
    
    assert final_output == "Hello world!"
    print("✓ Complete workflow test passed")


if __name__ == "__main__":
    test_hello_function()
    test_world_function()
    test_workflow()
    print("\n✅ All tests passed!")
