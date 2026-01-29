"""
Frequency Analysis - FAST analysis (~0.3-0.5s)

Calculates allele frequencies and population statistics.
This is quick computation - mostly arithmetic operations.
"""
import json
import time
import random


def lambda_handler(event, context):
    """Calculate allele frequencies - FAST operation."""
    start_time = time.time()
    
    analysis_type = event.get('analysis_type', 'frequency_analysis')
    delay_factor = event.get('delay_factor', 0.4)
    sifted_data = event.get('sifted_data', {})
    
    total_variants = sifted_data.get('total_variants', 10000)
    individuals = sifted_data.get('individuals', [])
    num_individuals = len(individuals) if individuals else 6
    
    # FAST operation
    actual_delay = delay_factor * (0.8 + random.random() * 0.4)
    time.sleep(actual_delay)
    
    # Calculate frequency statistics
    total_alleles = num_individuals * 2  # Diploid
    
    # Generate mock frequency distribution
    freq_bins = {
        'singleton': int(total_variants * 0.25),  # Only in 1 individual
        'rare': int(total_variants * 0.35),  # AF < 5%
        'low_freq': int(total_variants * 0.15),  # AF 5-10%
        'common': int(total_variants * 0.20),  # AF 10-50%
        'high_freq': int(total_variants * 0.05),  # AF > 50%
    }
    
    # Hardy-Weinberg statistics
    hw_stats = {
        'in_equilibrium': int(total_variants * 0.92),
        'deviation': int(total_variants * 0.08),
    }
    
    result = {
        'analysis_type': 'frequency_analysis',
        'total_variants': total_variants,
        'num_individuals': num_individuals,
        'total_alleles': total_alleles,
        'frequency_distribution': freq_bins,
        'hardy_weinberg': hw_stats,
        'statistics': {
            'mean_af': round(random.uniform(0.15, 0.25), 4),
            'median_af': round(random.uniform(0.05, 0.10), 4),
            'pi_diversity': round(random.uniform(0.0008, 0.0012), 6),
            'theta_watterson': round(random.uniform(0.0010, 0.0015), 6),
        },
        'processing_time_ms': int((time.time() - start_time) * 1000),
        'timestamp': time.time()
    }
    
    print(f'[FREQUENCY] Analyzed {total_variants} variants across {num_individuals} individuals')
    print(f'[FREQUENCY] Processing time: {result["processing_time_ms"]}ms')
    
    return result


if __name__ == '__main__':
    result = lambda_handler({
        'analysis_type': 'frequency_analysis',
        'delay_factor': 0.4,
        'sifted_data': {'total_variants': 15000, 'individuals': [{}] * 6}
    }, None)
    print(json.dumps(result, indent=2))
