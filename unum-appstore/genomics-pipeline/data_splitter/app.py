"""
Data Splitter - Splits genomic data into individual samples for parallel processing.

Inspired by 1000Genomes workflow from SeBS-Flow and scientific computing literature.
Creates payloads for multiple individuals with varying computational costs.
"""
import json
import time
import random


def lambda_handler(event, context):
    """
    Split genomic data into individual samples.
    
    Expected event:
    {
        "chromosome": 22,
        "num_individuals": 6,
        "region_start": 16050000,
        "region_end": 16100000
    }
    """
    chromosome = event.get('chromosome', 22)
    num_individuals = event.get('num_individuals', 6)
    region_start = event.get('region_start', 16050000)
    region_end = event.get('region_end', 16100000)
    
    start_time = time.time()
    
    # Simulate reading VCF header
    time.sleep(0.3)
    
    # Create payloads for parallel individual processing
    # Each individual has different coverage depths = different processing times
    individual_configs = [
        {'id': 'NA12878', 'coverage': 60, 'delay_factor': 3.5},   # High coverage - SLOW
        {'id': 'NA12891', 'coverage': 30, 'delay_factor': 1.5},   # Medium coverage
        {'id': 'NA12892', 'coverage': 30, 'delay_factor': 1.8},   # Medium coverage
        {'id': 'HG00096', 'coverage': 15, 'delay_factor': 0.5},   # Low coverage - FAST
        {'id': 'HG00097', 'coverage': 15, 'delay_factor': 0.4},   # Low coverage - FAST
        {'id': 'HG00099', 'coverage': 45, 'delay_factor': 2.5},   # High-medium coverage
    ]
    
    payloads = []
    for i, config in enumerate(individual_configs[:num_individuals]):
        payloads.append({
            'individual_id': config['id'],
            'individual_index': i,
            'chromosome': chromosome,
            'region_start': region_start,
            'region_end': region_end,
            'coverage_depth': config['coverage'],
            'delay_factor': config['delay_factor'],
            'num_variants_estimate': int(config['coverage'] * 100 + random.randint(0, 500)),
            'splitter_timestamp': time.time()
        })
    
    processing_time = int((time.time() - start_time) * 1000)
    
    print(f'[SPLITTER] Chromosome {chromosome}:{region_start}-{region_end}')
    print(f'[SPLITTER] Created {len(payloads)} individual payloads')
    print(f'[SPLITTER] Delay factors: {[p["delay_factor"] for p in payloads]}')
    
    return payloads


if __name__ == '__main__':
    result = lambda_handler({'chromosome': 22, 'num_individuals': 6}, None)
    print(f"Generated {len(result)} payloads:")
    for p in result:
        print(f"  - {p['individual_id']}: coverage={p['coverage_depth']}x, ~{p['delay_factor']}s")
