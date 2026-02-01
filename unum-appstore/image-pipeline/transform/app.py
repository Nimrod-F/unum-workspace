"""
Transform - Light-medium branch (~100-150ms)

Geometric transformations - moderately light computation.
"""
import json
import time
import base64
from io import BytesIO
from PIL import Image


def lambda_handler(event, context):
    """
    Apply geometric transformations - LIGHT-MEDIUM complexity.
    
    Operations:
    - Rotate 90°, 180°, 270°
    - Flip horizontal and vertical
    - Transpose
    
    These are pixel remapping operations - faster than convolutions.
    Expected time: ~100-150ms
    """
    start_time = time.time()
    
    bucket = event.get('bucket')
    key = event.get('key')
    image_b64 = event.get('image_b64')
    
    print(f'[Transform] Starting - LIGHT-MEDIUM branch (~100-150ms)')
    print(f'[Transform] Processing: {key}')
    
    # Decode image
    decode_start = time.time()
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes))
    original_size = image.size
    decode_time = (time.time() - decode_start) * 1000
    
    print(f'[Transform] Decoded {original_size[0]}x{original_size[1]} in {decode_time:.0f}ms')
    
    # === REAL COMPUTATION: Geometric transformations ===
    compute_start = time.time()
    results = []
    
    # 1. Rotate 90°
    rot90_start = time.time()
    rot90 = image.transpose(Image.Transpose.ROTATE_90)
    rot90_time = (time.time() - rot90_start) * 1000
    results.append({'transform': 'rotate_90', 'time_ms': int(rot90_time)})
    
    # 2. Rotate 180°
    rot180_start = time.time()
    rot180 = image.transpose(Image.Transpose.ROTATE_180)
    rot180_time = (time.time() - rot180_start) * 1000
    results.append({'transform': 'rotate_180', 'time_ms': int(rot180_time)})
    
    # 3. Rotate 270°
    rot270_start = time.time()
    rot270 = image.transpose(Image.Transpose.ROTATE_270)
    rot270_time = (time.time() - rot270_start) * 1000
    results.append({'transform': 'rotate_270', 'time_ms': int(rot270_time)})
    
    # 4. Flip left-right
    flip_lr_start = time.time()
    flip_lr = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    flip_lr_time = (time.time() - flip_lr_start) * 1000
    results.append({'transform': 'flip_lr', 'time_ms': int(flip_lr_time)})
    
    # 5. Flip top-bottom
    flip_tb_start = time.time()
    flip_tb = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    flip_tb_time = (time.time() - flip_tb_start) * 1000
    results.append({'transform': 'flip_tb', 'time_ms': int(flip_tb_time)})
    
    # 6. Transpose (swap x and y)
    transpose_start = time.time()
    transposed = image.transpose(Image.Transpose.TRANSPOSE)
    transpose_time = (time.time() - transpose_start) * 1000
    results.append({'transform': 'transpose', 'time_ms': int(transpose_time)})
    
    # 7. Arbitrary rotation (more expensive - uses interpolation)
    rotate_start = time.time()
    rotated_45 = image.rotate(45, expand=True, resample=Image.Resampling.BICUBIC)
    rotate_time = (time.time() - rotate_start) * 1000
    results.append({'transform': 'rotate_45', 'time_ms': int(rotate_time)})
    
    compute_time = (time.time() - compute_start) * 1000
    # === END COMPUTATION ===
    
    # Save final result
    output_buffer = BytesIO()
    rot180.save(output_buffer, format='JPEG', quality=85)
    output_bytes = output_buffer.getvalue()
    
    total_time = (time.time() - start_time) * 1000
    
    result = {
        'operation': 'transform',
        'branch': 'Transform',
        'input_key': key,
        'input_size': original_size,
        'transforms_applied': results,
        'output_bytes': len(output_bytes),
        'decode_time_ms': int(decode_time),
        'compute_time_ms': int(compute_time),
        'total_time_ms': int(total_time),
        'timestamp': time.time()
    }
    
    print(f'[Transform] COMPLETE - Compute: {compute_time:.0f}ms, Total: {total_time:.0f}ms')
    print(f'[Transform] Transform times: {[t["transform"] + ":" + str(t["time_ms"]) + "ms" for t in results]}')
    
    return result


if __name__ == '__main__':
    from PIL import Image
    import io
    
    # Create test image
    test_img = Image.new('RGB', (1920, 1080), color='yellow')
    buffer = io.BytesIO()
    test_img.save(buffer, format='JPEG')
    test_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    result = lambda_handler({
        'bucket': 'test',
        'key': 'test.jpg',
        'image_b64': test_b64
    }, None)
    print(json.dumps({k: v for k, v in result.items() if k != 'image_b64'}, indent=2))
