"""
Individuals Processor - Processes variant calls for a single individual.

Processing time varies based on coverage depth:
- Low coverage (15x): ~0.4-0.5s (FAST)
- Medium coverage (30x): ~1.5-1.8s
- High coverage (60x): ~3.5s (SLOW)
"""
import json
import time
import random


def lambda_handler(event, context):
    """Process variants for a single individual."""
    start_time = time.time()
    
    individual_id = event.get('individual_id', 'unknown')
    individual_index = event.get('individual_index', 0)
    coverage_depth = event.get('coverage_depth', 30)
    delay_factor = event.get('delay_factor', 1.5)
    chromosome = event.get('chromosome', 22)
    num_variants_estimate = event.get('num_variants_estimate', 3000)
    
    # Processing time proportional to coverage depth
    actual_delay = delay_factor * (0.85 + random.random() * 0.3)
    time.sleep(actual_delay)
    
    # Generate mock variant calls
    num_snps = int(num_variants_estimate * 0.85)
    num_indels = int(num_variants_estimate * 0.15)
    
    # Simulate quality metrics
    mean_quality = 25 + coverage_depth * 0.2 + random.uniform(-2, 2)
    pass_rate = min(0.98, 0.85 + coverage_depth * 0.002)
    
    result = {
        'individual_id': individual_id,
        'individual_index': individual_index,
        'chromosome': chromosome,
        'coverage_depth': coverage_depth,
        'variants': {
            'total': num_snps + num_indels,
            'snps': num_snps,
            'indels': num_indels,
            'transitions': int(num_snps * 0.65),
            'transversions': int(num_snps * 0.35),
        },
        'quality_metrics': {
            'mean_quality': round(mean_quality, 2),
            'pass_rate': round(pass_rate, 4),
            'het_hom_ratio': round(1.5 + random.uniform(-0.3, 0.3), 3),
        },
        'processing_time_ms': int((time.time() - start_time) * 1000),
        'timestamp': time.time()
    }
    
    print(f'[INDIVIDUAL] {individual_id}: {result["variants"]["total"]} variants in {result["processing_time_ms"]}ms')
    
    return result


if __name__ == '__main__':
    result = lambda_handler({
        'individual_id': 'NA12878',
        'coverage_depth': 60,
        'delay_factor': 3.5
    }, None)
    print(json.dumps(result, indent=2))
