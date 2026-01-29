"""
Generate Thumbnail - Fast operation (~100-200ms)

Creates a small thumbnail. Quick but not instant.
"""
import json
import time
import random


def lambda_handler(event, context):
    """Generate thumbnail - fast operation."""
    start_time = time.time()
    
    image_id = event.get('image_id', 'unknown')
    delay_factor = event.get('delay_factor', 0.15)
    original_width = event.get('width', 4000)
    original_height = event.get('height', 3000)
    
    # Fast operation
    actual_delay = delay_factor * (0.8 + random.random() * 0.4)
    time.sleep(actual_delay)
    
    # Calculate thumbnail dimensions
    thumb_size = 150
    if original_width > original_height:
        thumb_width = thumb_size
        thumb_height = int(thumb_size * original_height / original_width)
    else:
        thumb_height = thumb_size
        thumb_width = int(thumb_size * original_width / original_height)
    
    result = {
        'operation': 'generate_thumbnail',
        'image_id': image_id,
        'original_size': f'{original_width}x{original_height}',
        'thumbnail_size': f'{thumb_width}x{thumb_height}',
        'thumbnail_url': f's3://thumbnails/{image_id}_thumb.jpg',
        'file_size_kb': int(thumb_width * thumb_height * 0.5 / 1024),  # Rough estimate
        'processing_time_ms': int((time.time() - start_time) * 1000),
        'timestamp': time.time()
    }
    
    print(f'[THUMBNAIL] Generated {thumb_width}x{thumb_height} for {image_id} in {result["processing_time_ms"]}ms')
    
    return result


if __name__ == '__main__':
    result = lambda_handler({'image_id': 'test', 'delay_factor': 0.15, 'width': 4000, 'height': 3000}, None)
    print(json.dumps(result, indent=2))
