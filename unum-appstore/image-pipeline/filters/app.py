"""
Filters - Medium complexity branch (~150-200ms)

Applies multiple convolution filters - moderately heavy computation.
"""
import json
import time
import base64
from io import BytesIO
from PIL import Image, ImageFilter


def lambda_handler(event, context):
    """
    Apply multiple image filters - MEDIUM complexity.
    
    Operations:
    - Gaussian blur (3x3 or 5x5 kernel convolution)
    - Sharpen filter (3x3 kernel convolution)
    - Smooth filter (box blur)
    
    Each filter requires convolution over entire image.
    Expected time: ~150-200ms
    """
    start_time = time.time()
    
    bucket = event.get('bucket')
    key = event.get('key')
    image_b64 = event.get('image_b64')
    
    print(f'[Filters] Starting - MEDIUM branch (~150-200ms)')
    print(f'[Filters] Processing: {key}')
    
    # Decode image
    decode_start = time.time()
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes))
    original_size = image.size
    decode_time = (time.time() - decode_start) * 1000
    
    print(f'[Filters] Decoded {original_size[0]}x{original_size[1]} in {decode_time:.0f}ms')
    
    # === REAL COMPUTATION: Multiple filter operations ===
    compute_start = time.time()
    results = []
    
    # 1. Gaussian Blur - convolution with gaussian kernel
    blur_start = time.time()
    blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
    blur_time = (time.time() - blur_start) * 1000
    results.append({'filter': 'gaussian_blur', 'time_ms': int(blur_time)})
    
    # 2. Sharpen - edge enhancement convolution
    sharpen_start = time.time()
    sharpened = image.filter(ImageFilter.SHARPEN)
    sharpen_time = (time.time() - sharpen_start) * 1000
    results.append({'filter': 'sharpen', 'time_ms': int(sharpen_time)})
    
    # 3. Smooth - box blur convolution
    smooth_start = time.time()
    smoothed = image.filter(ImageFilter.SMOOTH_MORE)
    smooth_time = (time.time() - smooth_start) * 1000
    results.append({'filter': 'smooth', 'time_ms': int(smooth_time)})
    
    # 4. Detail enhancement
    detail_start = time.time()
    detailed = image.filter(ImageFilter.DETAIL)
    detail_time = (time.time() - detail_start) * 1000
    results.append({'filter': 'detail', 'time_ms': int(detail_time)})
    
    # 5. Edge enhance
    edge_start = time.time()
    edge_enhanced = image.filter(ImageFilter.EDGE_ENHANCE)
    edge_time = (time.time() - edge_start) * 1000
    results.append({'filter': 'edge_enhance', 'time_ms': int(edge_time)})
    
    compute_time = (time.time() - compute_start) * 1000
    # === END COMPUTATION ===
    
    # Save final result
    output_buffer = BytesIO()
    sharpened.save(output_buffer, format='JPEG', quality=85)
    output_bytes = output_buffer.getvalue()
    
    total_time = (time.time() - start_time) * 1000
    
    result = {
        'operation': 'filters',
        'branch': 'Filters',
        'input_key': key,
        'input_size': original_size,
        'filters_applied': results,
        'output_bytes': len(output_bytes),
        'decode_time_ms': int(decode_time),
        'compute_time_ms': int(compute_time),
        'total_time_ms': int(total_time),
        'timestamp': time.time()
    }
    
    print(f'[Filters] COMPLETE - Compute: {compute_time:.0f}ms, Total: {total_time:.0f}ms')
    print(f'[Filters] Filter times: {[f["filter"] + ":" + str(f["time_ms"]) + "ms" for f in results]}')
    
    return result


if __name__ == '__main__':
    from PIL import Image
    import io
    
    # Create test image
    test_img = Image.new('RGB', (1920, 1080), color='green')
    buffer = io.BytesIO()
    test_img.save(buffer, format='JPEG')
    test_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    result = lambda_handler({
        'bucket': 'test',
        'key': 'test.jpg',
        'image_b64': test_b64
    }, None)
    print(json.dumps({k: v for k, v in result.items() if k != 'image_b64'}, indent=2))
