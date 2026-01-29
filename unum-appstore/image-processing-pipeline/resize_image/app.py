"""
Resize Image - Medium-fast operation (~300-500ms)

Resizes image to target dimensions.
"""
import json
import time
import random


def lambda_handler(event, context):
    """Resize image - medium-fast operation."""
    start_time = time.time()
    
    image_id = event.get('image_id', 'unknown')
    delay_factor = event.get('delay_factor', 0.4)
    original_width = event.get('width', 4000)
    original_height = event.get('height', 3000)
    
    # Medium-fast operation
    actual_delay = delay_factor * (0.8 + random.random() * 0.4)
    time.sleep(actual_delay)
    
    # Target size
    target_width = 1920
    target_height = int(1920 * original_height / original_width)
    
    result = {
        'operation': 'resize_image',
        'image_id': image_id,
        'original_size': f'{original_width}x{original_height}',
        'resized_size': f'{target_width}x{target_height}',
        'resized_url': f's3://resized/{image_id}_1920.jpg',
        'compression_ratio': round(original_width * original_height / (target_width * target_height), 2),
        'estimated_file_size_kb': int(target_width * target_height * 3 * 0.15 / 1024),
        'processing_time_ms': int((time.time() - start_time) * 1000),
        'timestamp': time.time()
    }
    
    print(f'[RESIZE] Resized to {target_width}x{target_height} for {image_id} in {result["processing_time_ms"]}ms')
    
    return result


if __name__ == '__main__':
    result = lambda_handler({'image_id': 'test', 'delay_factor': 0.4, 'width': 4000, 'height': 3000}, None)
    print(json.dumps(result, indent=2))
