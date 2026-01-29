"""
Image Loader - Loads image and fans out to parallel processing functions.

Each parallel operation has dramatically different processing times:
- Metadata extraction: ~50-100ms (fastest)
- Thumbnail generation: ~100-200ms (fast)
- Resize: ~200-500ms (medium-fast)
- Apply Filters: ~1-2s (medium)
- Face Detection: ~2-5s (slowest)
"""
import json
import time
import random


def lambda_handler(event, context):
    """
    Load image and create payloads for parallel processing operations.
    
    Expected event:
    {
        "image_id": "sample_image",
        "image_url": "s3://bucket/image.jpg",
        "width": 4000,
        "height": 3000,
        "file_size_kb": 2500
    }
    """
    image_id = event.get('image_id', f'img_{int(time.time())}')
    image_url = event.get('image_url', 's3://sample-bucket/sample.jpg')
    width = event.get('width', 4000)
    height = event.get('height', 3000)
    file_size_kb = event.get('file_size_kb', 2500)
    
    start_time = time.time()
    
    # Simulate image loading
    time.sleep(0.2)
    
    # Generate mock image data
    image_data = {
        'image_id': image_id,
        'url': image_url,
        'width': width,
        'height': height,
        'file_size_kb': file_size_kb,
        'format': 'JPEG',
        'color_space': 'RGB',
        'bit_depth': 8,
    }
    
    # Create payloads for each parallel operation with expected delays
    operations = [
        {
            'operation': 'extract_metadata',
            'delay_factor': 0.05,  # ~50-100ms - FASTEST
            'description': 'Read EXIF and image metadata'
        },
        {
            'operation': 'generate_thumbnail',
            'delay_factor': 0.15,  # ~100-200ms - fast
            'description': 'Create 150x150 thumbnail'
        },
        {
            'operation': 'resize_image',
            'delay_factor': 0.4,   # ~300-500ms - medium-fast
            'description': 'Resize to 1920x1080'
        },
        {
            'operation': 'apply_filters',
            'delay_factor': 1.5,   # ~1-2s - medium
            'description': 'Apply enhancement filters'
        },
        {
            'operation': 'detect_faces',
            'delay_factor': 3.5,   # ~2-5s - SLOWEST
            'description': 'Detect and analyze faces'
        },
    ]
    
    payloads = []
    for op in operations:
        payloads.append({
            **image_data,
            'operation': op['operation'],
            'delay_factor': op['delay_factor'],
            'description': op['description'],
            'loader_timestamp': time.time()
        })
    
    processing_time = int((time.time() - start_time) * 1000)
    
    print(f'[LOADER] Image {image_id}: {width}x{height}, {file_size_kb}KB')
    print(f'[LOADER] Created {len(payloads)} operation payloads')
    print(f'[LOADER] Expected delays: {[p["delay_factor"] for p in payloads]}s')
    
    return payloads


if __name__ == '__main__':
    result = lambda_handler({
        "image_id": "test_image",
        "width": 4000,
        "height": 3000
    }, None)
    print(f"Generated {len(result)} payloads:")
    for p in result:
        print(f"  - {p['operation']}: ~{p['delay_factor']}s")
