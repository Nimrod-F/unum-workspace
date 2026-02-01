"""
Thumbnail - FASTEST branch (~50-80ms)

Simple image resize operation - computationally lightweight.
This should complete first and trigger the Publisher in FUTURE mode.
"""
import json
import time
import base64
from io import BytesIO
from PIL import Image


def lambda_handler(event, context):
    """
    Create thumbnail - FASTEST operation.
    
    Operations:
    - Resize to 128x128 thumbnail
    - Simple interpolation (no heavy filtering)
    
    Expected time: ~50-80ms (depending on input size)
    """
    start_time = time.time()
    
    bucket = event.get('bucket')
    key = event.get('key')
    image_b64 = event.get('image_b64')
    
    print(f'[Thumbnail] Starting - FASTEST branch (~50ms)')
    print(f'[Thumbnail] Processing: {key}')
    
    # Decode image
    decode_start = time.time()
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes))
    original_size = image.size
    decode_time = (time.time() - decode_start) * 1000
    
    print(f'[Thumbnail] Decoded {original_size[0]}x{original_size[1]} in {decode_time:.0f}ms')
    
    # === REAL COMPUTATION: Thumbnail generation ===
    compute_start = time.time()
    
    # Create thumbnail (128x128) - fast operation
    thumb = image.copy()
    thumb.thumbnail((128, 128), Image.Resampling.LANCZOS)
    
    # Save to bytes
    output_buffer = BytesIO()
    thumb.save(output_buffer, format='JPEG', quality=85)
    thumb_bytes = output_buffer.getvalue()
    
    compute_time = (time.time() - compute_start) * 1000
    # === END COMPUTATION ===
    
    total_time = (time.time() - start_time) * 1000
    
    result = {
        'operation': 'thumbnail',
        'branch': 'Thumbnail',
        'input_key': key,
        'input_size': original_size,
        'output_size': thumb.size,
        'output_bytes': len(thumb_bytes),
        'decode_time_ms': int(decode_time),
        'compute_time_ms': int(compute_time),
        'total_time_ms': int(total_time),
        'timestamp': time.time()
    }
    
    print(f'[Thumbnail] COMPLETE - Compute: {compute_time:.0f}ms, Total: {total_time:.0f}ms')
    print(f'[Thumbnail] This is the FASTEST branch - should trigger Publisher in FUTURE mode')
    
    return result


if __name__ == '__main__':
    # Local test with a sample image
    from PIL import Image
    import io
    
    # Create test image
    test_img = Image.new('RGB', (1920, 1080), color='blue')
    buffer = io.BytesIO()
    test_img.save(buffer, format='JPEG')
    test_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    result = lambda_handler({
        'bucket': 'test',
        'key': 'test.jpg',
        'image_b64': test_b64
    }, None)
    print(json.dumps(result, indent=2))
