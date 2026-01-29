"""
Extract Metadata - FASTEST operation (~50-100ms)

Reads EXIF and image metadata. This completes almost instantly
and demonstrates immediate availability with future-based execution.
"""
import json
import time
import random


def lambda_handler(event, context):
    """Extract image metadata - fastest operation."""
    start_time = time.time()
    
    image_id = event.get('image_id', 'unknown')
    delay_factor = event.get('delay_factor', 0.05)
    
    # Very fast operation
    actual_delay = delay_factor * (0.8 + random.random() * 0.4)
    time.sleep(actual_delay)
    
    # Generate mock metadata
    metadata = {
        'operation': 'extract_metadata',
        'image_id': image_id,
        'exif': {
            'make': 'Canon',
            'model': 'EOS 5D Mark IV',
            'focal_length': '50mm',
            'aperture': 'f/2.8',
            'iso': 400,
            'shutter_speed': '1/250',
            'date_taken': '2024-01-15 14:30:00',
            'gps_lat': 51.5074,
            'gps_lon': -0.1278,
        },
        'file_info': {
            'format': event.get('format', 'JPEG'),
            'width': event.get('width', 4000),
            'height': event.get('height', 3000),
            'color_space': event.get('color_space', 'RGB'),
            'bit_depth': event.get('bit_depth', 8),
            'has_alpha': False,
        },
        'processing_time_ms': int((time.time() - start_time) * 1000),
        'timestamp': time.time()
    }
    
    print(f'[METADATA] Extracted metadata for {image_id} in {metadata["processing_time_ms"]}ms')
    
    return metadata


if __name__ == '__main__':
    result = lambda_handler({'image_id': 'test', 'delay_factor': 0.05}, None)
    print(json.dumps(result, indent=2))
