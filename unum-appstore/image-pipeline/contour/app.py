"""
Contour - SLOWEST branch (~200-400ms)

Heavy edge detection and contour extraction - most computationally expensive.
This is the SLOWEST branch due to complex convolution kernels.

BENCHMARK SUPPORT:
Set ARTIFICIAL_DELAY_MS environment variable to add simulated delay.
This allows testing Future-Based execution benefits with varying branch times.
"""
import json
import time
import base64
import os
from io import BytesIO
from PIL import Image, ImageFilter


def get_artificial_delay() -> int:
    """Get artificial delay from environment variable (in milliseconds)"""
    try:
        return int(os.environ.get('ARTIFICIAL_DELAY_MS', '0'))
    except (ValueError, TypeError):
        return 0


def lambda_handler(event, context):
    """
    Apply contour/edge detection filters - HEAVIEST computation.
    
    Operations:
    - CONTOUR: Full edge detection convolution
    - FIND_EDGES: Sobel/Prewitt edge detection
    - EMBOSS: 3D effect via convolution
    - Multiple passes for quality
    
    These are the most computationally expensive PIL operations.
    Expected time: ~200-400ms (SLOWEST branch)
    """
    start_time = time.time()
    
    bucket = event.get('bucket')
    key = event.get('key')
    image_b64 = event.get('image_b64')
    
    print(f'[Contour] Starting - SLOWEST branch (~200-400ms)')
    print(f'[Contour] Processing: {key}')
    
    # Decode image
    decode_start = time.time()
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes))
    original_size = image.size
    decode_time = (time.time() - decode_start) * 1000
    
    print(f'[Contour] Decoded {original_size[0]}x{original_size[1]} in {decode_time:.0f}ms')
    
    # === REAL COMPUTATION: Heavy edge detection operations ===
    compute_start = time.time()
    results = []
    
    # 1. CONTOUR - most expensive operation
    contour_start = time.time()
    contour_img = image.filter(ImageFilter.CONTOUR)
    contour_time = (time.time() - contour_start) * 1000
    results.append({'filter': 'contour', 'time_ms': int(contour_time)})
    
    # 2. FIND_EDGES - edge detection
    edges_start = time.time()
    edges_img = image.filter(ImageFilter.FIND_EDGES)
    edges_time = (time.time() - edges_start) * 1000
    results.append({'filter': 'find_edges', 'time_ms': int(edges_time)})
    
    # 3. EMBOSS - 3D relief effect
    emboss_start = time.time()
    emboss_img = image.filter(ImageFilter.EMBOSS)
    emboss_time = (time.time() - emboss_start) * 1000
    results.append({'filter': 'emboss', 'time_ms': int(emboss_time)})
    
    # 4. Apply contour again on blurred version (more expensive)
    contour2_start = time.time()
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
    contour2_img = blurred.filter(ImageFilter.CONTOUR)
    contour2_time = (time.time() - contour2_start) * 1000
    results.append({'filter': 'contour_blurred', 'time_ms': int(contour2_time)})
    
    # 5. Custom kernel convolution - even heavier
    kernel_start = time.time()
    # Laplacian edge detection kernel
    kernel = ImageFilter.Kernel(
        size=(3, 3),
        kernel=[0, -1, 0, -1, 4, -1, 0, -1, 0],
        scale=1,
        offset=128
    )
    laplacian_img = image.filter(kernel)
    kernel_time = (time.time() - kernel_start) * 1000
    results.append({'filter': 'laplacian', 'time_ms': int(kernel_time)})
    
    # 6. Sobel X and Y edge detection
    sobel_start = time.time()
    sobel_x = ImageFilter.Kernel(
        size=(3, 3),
        kernel=[-1, 0, 1, -2, 0, 2, -1, 0, 1],
        scale=1,
        offset=128
    )
    sobel_y = ImageFilter.Kernel(
        size=(3, 3),
        kernel=[-1, -2, -1, 0, 0, 0, 1, 2, 1],
        scale=1,
        offset=128
    )
    sobel_x_img = image.filter(sobel_x)
    sobel_y_img = image.filter(sobel_y)
    sobel_time = (time.time() - sobel_start) * 1000
    results.append({'filter': 'sobel_xy', 'time_ms': int(sobel_time)})
    
    compute_time = (time.time() - compute_start) * 1000
    # === END COMPUTATION ===
    
    # === ARTIFICIAL DELAY (for benchmarking) ===
    artificial_delay_ms = get_artificial_delay()
    if artificial_delay_ms > 0:
        print(f'[Contour] Applying artificial delay: {artificial_delay_ms}ms')
        time.sleep(artificial_delay_ms / 1000.0)
    # === END ARTIFICIAL DELAY ===
    
    # Save final result
    output_buffer = BytesIO()
    contour_img.save(output_buffer, format='JPEG', quality=85)
    output_bytes = output_buffer.getvalue()
    
    total_time = (time.time() - start_time) * 1000
    
    result = {
        'operation': 'contour',
        'branch': 'Contour',
        'input_key': key,
        'input_size': original_size,
        'filters_applied': results,
        'output_bytes': len(output_bytes),
        'decode_time_ms': int(decode_time),
        'compute_time_ms': int(compute_time),
        'artificial_delay_ms': artificial_delay_ms,
        'total_time_ms': int(total_time),
        'timestamp': time.time()
    }
    
    print(f'[Contour] COMPLETE - Compute: {compute_time:.0f}ms, Delay: {artificial_delay_ms}ms, Total: {total_time:.0f}ms')
    print(f'[Contour] This is the SLOWEST branch - triggers Publisher in CLASSIC mode')
    print(f'[Contour] Filter times: {[f["filter"] + ":" + str(f["time_ms"]) + "ms" for f in results]}')
    
    return result


if __name__ == '__main__':
    from PIL import Image
    import io
    
    # Create test image (larger = more noticeable time difference)
    test_img = Image.new('RGB', (1920, 1080), color='red')
    # Add some variation for realistic edge detection
    for x in range(0, 1920, 100):
        for y in range(0, 1080, 100):
            test_img.putpixel((x, y), (255, 255, 255))
    
    buffer = io.BytesIO()
    test_img.save(buffer, format='JPEG')
    test_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    result = lambda_handler({
        'bucket': 'test',
        'key': 'test.jpg',
        'image_b64': test_b64
    }, None)
    print(json.dumps({k: v for k, v in result.items() if k != 'image_b64'}, indent=2))
