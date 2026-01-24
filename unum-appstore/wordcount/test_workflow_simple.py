"""
Simplified local test for wordcount workflow without AWS dependencies
This simulates the workflow logic without S3 storage
"""
import json
import re
from collections import defaultdict
from pathlib import Path


def simple_mapper(text):
    """Simple mapper function - splits text into words"""
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [(word, 1) for word in words]


def simple_partition(map_results, num_reducers=3):
    """Partition words across reducers using hash"""
    partitions = [defaultdict(list) for _ in range(num_reducers)]
    
    for words in map_results:
        for word, count in words:
            # Hash word to determine which reducer
            reducer_id = hash(word) % num_reducers
            partitions[reducer_id][word].append(count)
    
    return partitions


def simple_reducer(partition):
    """Reduce function - sum counts for each word"""
    result = {}
    for word, counts in partition.items():
        result[word] = sum(counts)
    return result


def simple_summary(reduce_results):
    """Summary function - combine all reducer outputs"""
    final_counts = defaultdict(int)
    for result in reduce_results:
        for word, count in result.items():
            final_counts[word] += count
    
    # Sort by count (descending) then by word (alphabetically)
    sorted_counts = dict(sorted(final_counts.items(), 
                                key=lambda x: (-x[1], x[0])))
    return sorted_counts


def test_wordcount_workflow():
    """Test the complete wordcount workflow"""
    
    # Load test event
    event_file = Path(__file__).parent / 'events' / 'test.json'
    with open(event_file, 'r') as f:
        initial_event = json.load(f)
    
    print("=" * 70)
    print("WordCount Workflow - Simplified Local Test")
    print("=" * 70)
    
    # Extract text items
    data_items = initial_event.get('Data', {}).get('Value', [])
    print(f"\n📥 Input: {len(data_items)} text documents")
    
    # Step 1: Map Phase - Process each text through mapper
    print("\n[Step 1] MAP PHASE - Mapping words...")
    map_results = []
    total_words = 0
    
    for i, item in enumerate(data_items):
        text = item.get('text', '')
        word_counts = simple_mapper(text)
        map_results.append(word_counts)
        total_words += len(word_counts)
        print(f"  Mapper {i+1}: Processed {len(word_counts)} words")
    
    print(f"  Total words processed: {total_words}")
    
    # Step 2: Partition Phase - Organize for reducers
    print("\n[Step 2] PARTITION PHASE - Distributing to reducers...")
    num_reducers = 3
    partitions = simple_partition(map_results, num_reducers)
    
    for i, partition in enumerate(partitions):
        unique_words = len(partition)
        total_counts = sum(len(counts) for counts in partition.values())
        print(f"  Reducer {i}: {unique_words} unique words, {total_counts} total occurrences")
    
    # Step 3: Reduce Phase - Count words in each partition
    print("\n[Step 3] REDUCE PHASE - Counting words...")
    reduce_results = []
    
    for i, partition in enumerate(partitions):
        result = simple_reducer(partition)
        reduce_results.append(result)
        print(f"  Reducer {i}: Counted {len(result)} unique words")
    
    # Step 4: Summary Phase - Combine all results
    print("\n[Step 4] SUMMARY PHASE - Aggregating results...")
    final_result = simple_summary(reduce_results)
    
    print(f"\n{'=' * 70}")
    print("FINAL WORD COUNT RESULTS")
    print('=' * 70)
    print(f"\nTotal unique words: {len(final_result)}")
    print(f"\nTop 20 most frequent words:")
    print(f"{'Word':<20} {'Count':>10}")
    print('-' * 32)
    
    for i, (word, count) in enumerate(list(final_result.items())[:20]):
        print(f"{word:<20} {count:>10}")
    
    print('\n' + '=' * 70)
    print(f"\n✅ Workflow completed successfully!")
    print(f"   - Processed {len(data_items)} documents")
    print(f"   - Counted {sum(final_result.values())} total words")
    print(f"   - Found {len(final_result)} unique words")
    print('=' * 70)
    
    return final_result


if __name__ == "__main__":
    result = test_wordcount_workflow()
