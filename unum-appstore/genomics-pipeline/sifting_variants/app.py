"""
Sifting Variants - Filters and categorizes variants.

Sequential step between the two fan-in points.
Takes merged data and prepares for parallel analysis.
"""
import json
import time
import random


def lambda_handler(event, context):
    """
    Filter and sift variants, then prepare for parallel analysis.
    
    Creates payloads for:
    1. Mutation Overlap analysis (~2-4s - SLOW)
    2. Frequency Analysis (~0.3-0.5s - FAST)
    """
    start_time = time.time()
    
    total_variants = event.get('total_variants', 10000)
    merged_variants = event.get('merged_variants', {})
    
    # Simulate sifting process
    time.sleep(0.5)
    
    # Filter variants
    num_coding = int(total_variants * 0.02)  # ~2% in coding regions
    num_regulatory = int(total_variants * 0.05)  # ~5% in regulatory
    num_intronic = int(total_variants * 0.35)  # ~35% intronic
    num_intergenic = int(total_variants * 0.58)  # ~58% intergenic
    
    # Functional impact predictions
    high_impact = int(num_coding * 0.15)
    moderate_impact = int(num_coding * 0.45)
    low_impact = int(num_coding * 0.40)
    
    sifted_data = {
        'total_variants': total_variants,
        'by_region': {
            'coding': num_coding,
            'regulatory': num_regulatory,
            'intronic': num_intronic,
            'intergenic': num_intergenic,
        },
        'by_impact': {
            'high': high_impact,
            'moderate': moderate_impact,
            'low': low_impact,
        },
        'individuals': merged_variants.get('individuals', []),
    }
    
    # Create payloads for parallel analysis
    # IMPORTANT: Very different processing times!
    payloads = [
        {
            'analysis_type': 'mutation_overlap',
            'delay_factor': 3.0,  # ~2-4s - SLOW (compares with databases)
            'sifted_data': sifted_data,
            'sifting_timestamp': time.time()
        },
        {
            'analysis_type': 'frequency_analysis',
            'delay_factor': 0.4,  # ~0.3-0.5s - FAST
            'sifted_data': sifted_data,
            'sifting_timestamp': time.time()
        },
    ]
    
    processing_time = int((time.time() - start_time) * 1000)
    
    print(f'[SIFTING] Processed {total_variants} variants in {processing_time}ms')
    print(f'[SIFTING] High impact: {high_impact}, Moderate: {moderate_impact}')
    print(f'[SIFTING] Created 2 analysis payloads: mutation_overlap (~3s), frequency (~0.4s)')
    
    return payloads


if __name__ == '__main__':
    result = lambda_handler({'total_variants': 15000, 'merged_variants': {'individuals': []}}, None)
    print(f"Generated {len(result)} analysis payloads:")
    for p in result:
        print(f"  - {p['analysis_type']}: ~{p['delay_factor']}s")
