"""
Apply Filters - Medium operation (~1-2s)

Applies enhancement filters to the image.
"""
import json
import time
import random


def lambda_handler(event, context):
    """Apply image filters - medium duration operation."""
    start_time = time.time()
    
    image_id = event.get('image_id', 'unknown')
    delay_factor = event.get('delay_factor', 1.5)
    
    # Medium duration operation
    actual_delay = delay_factor * (0.8 + random.random() * 0.4)
    time.sleep(actual_delay)
    
    # Simulate filter application
    filters_applied = [
        {'name': 'auto_enhance', 'intensity': 0.7},
        {'name': 'sharpen', 'intensity': 0.3},
        {'name': 'denoise', 'intensity': 0.5},
        {'name': 'color_correction', 'temperature': 5500, 'tint': 0},
    ]
    
    result = {
        'operation': 'apply_filters',
        'image_id': image_id,
        'filters_applied': filters_applied,
        'num_filters': len(filters_applied),
        'enhanced_url': f's3://enhanced/{image_id}_enhanced.jpg',
        'quality_score': round(0.7 + random.random() * 0.25, 2),
        'processing_time_ms': int((time.time() - start_time) * 1000),
        'timestamp': time.time()
    }
    
    print(f'[FILTERS] Applied {len(filters_applied)} filters to {image_id} in {result["processing_time_ms"]}ms')
    
    return result


if __name__ == '__main__':
    result = lambda_handler({'image_id': 'test', 'delay_factor': 1.5}, None)
    print(json.dumps(result, indent=2))
