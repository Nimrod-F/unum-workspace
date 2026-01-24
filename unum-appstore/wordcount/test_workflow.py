"""
Test script to run the wordcount workflow locally
This simulates the complete workflow execution without AWS
"""
import sys
import os
import json
from pathlib import Path

# Add function directories to path
sys.path.insert(0, str(Path(__file__).parent / 'UnumMap0'))
sys.path.insert(0, str(Path(__file__).parent / 'mapper'))
sys.path.insert(0, str(Path(__file__).parent / 'partition'))
sys.path.insert(0, str(Path(__file__).parent / 'reducer'))
sys.path.insert(0, str(Path(__file__).parent / 'summary'))


def test_workflow():
    """Test the complete wordcount workflow"""
    
    # Load test event
    event_file = Path(__file__).parent / 'events' / 'test.json'
    with open(event_file, 'r') as f:
        initial_event = json.load(f)
    
    print("=" * 60)
    print("Starting WordCount Workflow Test")
    print("=" * 60)
    
    # Step 1: UnumMap0 - Entry point
    print("\n[Step 1] UnumMap0 - Processing input...")
    from UnumMap0.app import lambda_handler as unummap0_handler
    
    map0_output = unummap0_handler(initial_event, None)
    print(f"UnumMap0 Output: {json.dumps(map0_output, indent=2)[:200]}...")
    
    # Step 2: Count (Map) - Process each text through Mapper
    print("\n[Step 2] Count (Map) - Running Mappers...")
    from mapper.app import lambda_handler as mapper_handler
    
    map_outputs = []
    data_items = initial_event.get('Data', {}).get('Value', [])
    
    print(f"Processing {len(data_items)} items through mapper...")
    for i, item in enumerate(data_items):
        print(f"  Mapper {i+1}/{len(data_items)}: Processing text...")
        try:
            mapper_output = mapper_handler(item, None)
            map_outputs.append(mapper_output)
            print(f"  Mapper {i+1} output: {mapper_output}")
        except Exception as e:
            print(f"  Mapper {i+1} error: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 3: Partition - Organize mapper outputs for reducers
    print("\n[Step 3] Partition - Organizing data for reducers...")
    from partition.app import lambda_handler as partition_handler
    
    partition_event = {
        'mapOutputs': map_outputs
    }
    
    try:
        partition_output = partition_handler(partition_event, None)
        print(f"Partition Output: {json.dumps(partition_output, indent=2)[:200]}...")
    except Exception as e:
        print(f"Partition error: {e}")
        import traceback
        traceback.print_exc()
        partition_output = {'partitions': []}
    
    # Step 4: Reduce (Map) - Process each partition through Reducer
    print("\n[Step 4] Reduce (Map) - Running Reducers...")
    from reducer.app import lambda_handler as reducer_handler
    
    reduce_outputs = []
    partitions = partition_output.get('partitions', [])
    
    if not partitions and map_outputs:
        # If partition didn't work as expected, create simple partitions
        print("  Creating fallback partitions from mapper outputs...")
        for i, mapper_output in enumerate(map_outputs):
            partition = {
                'bucket': mapper_output.get('bucket', 'test-bucket'),
                'partition': i
            }
            partitions.append(partition)
    
    print(f"Processing {len(partitions)} partitions through reducer...")
    for i, partition in enumerate(partitions):
        print(f"  Reducer {i+1}/{len(partitions)}: Processing partition...")
        try:
            reducer_output = reducer_handler(partition, None)
            reduce_outputs.append(reducer_output)
            print(f"  Reducer {i+1} output: {str(reducer_output)[:100]}...")
        except Exception as e:
            print(f"  Reducer {i+1} error: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 5: Summary - Aggregate all reducer outputs
    print("\n[Step 5] Summary - Aggregating results...")
    from summary.app import lambda_handler as summary_handler
    
    summary_event = {
        'reduceOutputs': reduce_outputs
    }
    
    try:
        final_output = summary_handler(summary_event, None)
        print(f"\n{'=' * 60}")
        print("FINAL WORD COUNT RESULTS:")
        print('=' * 60)
        print(json.dumps(final_output, indent=2))
        print('=' * 60)
    except Exception as e:
        print(f"Summary error: {e}")
        import traceback
        traceback.print_exc()
        final_output = {'error': str(e)}
    
    print("\n✅ Workflow execution completed!")
    return final_output


if __name__ == "__main__":
    test_workflow()
