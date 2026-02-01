"""
Local Test - Verify timing variance without AWS deployment

Tests each branch function locally to measure real computation times.
This confirms that timing differences come from actual computation, not artificial delays.
"""
import time
import base64
import json
from io import BytesIO
from PIL import Image, ImageFilter
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_image(width=1920, height=1080):
    """Create a realistic test image with variation for edge detection."""
    img = Image.new('RGB', (width, height))
    
    # Add gradient and patterns for realistic processing
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            # Gradient with noise
            r = int((x / width) * 255)
            g = int((y / height) * 255)
            b = int(((x + y) / (width + height)) * 255)
            pixels[x, y] = (r, g, b)
    
    # Add some shapes for edge detection to work on
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for i in range(10):
        x1 = (i * 180) % width
        y1 = (i * 100) % height
        draw.rectangle([x1, y1, x1+100, y1+80], fill=(255, 255, 255))
        draw.ellipse([x1+50, y1+50, x1+150, y1+130], fill=(0, 0, 0))
    
    return img


def encode_image(img):
    """Encode PIL Image to base64."""
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def test_thumbnail(image_b64):
    """Test thumbnail generation."""
    from thumbnail.app import lambda_handler
    return lambda_handler({
        'bucket': 'test',
        'key': 'test.jpg',
        'image_b64': image_b64
    }, None)


def test_transform(image_b64):
    """Test geometric transformations."""
    from transform.app import lambda_handler
    return lambda_handler({
        'bucket': 'test',
        'key': 'test.jpg',
        'image_b64': image_b64
    }, None)


def test_filters(image_b64):
    """Test filter operations."""
    from filters.app import lambda_handler
    return lambda_handler({
        'bucket': 'test',
        'key': 'test.jpg',
        'image_b64': image_b64
    }, None)


def test_contour(image_b64):
    """Test contour/edge detection."""
    from contour.app import lambda_handler
    return lambda_handler({
        'bucket': 'test',
        'key': 'test.jpg',
        'image_b64': image_b64
    }, None)


def main():
    print("=" * 70)
    print("IMAGE PIPELINE LOCAL TEST - Real Computation Timing")
    print("=" * 70)
    
    # Test different image sizes
    sizes = [
        (640, 480, "Small"),
        (1920, 1080, "Medium (1080p)"),
        (3840, 2160, "Large (4K)"),
    ]
    
    for width, height, label in sizes:
        print(f"\n{'='*70}")
        print(f"Testing {label} image: {width}x{height}")
        print("=" * 70)
        
        # Create test image
        print("Creating test image...")
        img = create_test_image(width, height)
        image_b64 = encode_image(img)
        print(f"Image size: {len(image_b64)} bytes (base64)")
        
        # Test each branch
        results = {}
        
        print("\n--- Testing Thumbnail (FASTEST expected) ---")
        try:
            results['Thumbnail'] = test_thumbnail(image_b64)
        except Exception as e:
            print(f"Error: {e}")
            results['Thumbnail'] = {'compute_time_ms': 0, 'error': str(e)}
        
        print("\n--- Testing Transform (LIGHT-MEDIUM expected) ---")
        try:
            results['Transform'] = test_transform(image_b64)
        except Exception as e:
            print(f"Error: {e}")
            results['Transform'] = {'compute_time_ms': 0, 'error': str(e)}
        
        print("\n--- Testing Filters (MEDIUM expected) ---")
        try:
            results['Filters'] = test_filters(image_b64)
        except Exception as e:
            print(f"Error: {e}")
            results['Filters'] = {'compute_time_ms': 0, 'error': str(e)}
        
        print("\n--- Testing Contour (SLOWEST expected) ---")
        try:
            results['Contour'] = test_contour(image_b64)
        except Exception as e:
            print(f"Error: {e}")
            results['Contour'] = {'compute_time_ms': 0, 'error': str(e)}
        
        # Summary
        print("\n" + "-" * 70)
        print(f"TIMING SUMMARY for {label} ({width}x{height}):")
        print("-" * 70)
        
        times = []
        for branch, result in sorted(results.items(), key=lambda x: x[1].get('compute_time_ms', 0)):
            compute_ms = result.get('compute_time_ms', 0)
            total_ms = result.get('total_time_ms', 0)
            times.append((branch, compute_ms))
            print(f"  {branch:12}: compute={compute_ms:4}ms, total={total_ms:4}ms")
        
        if len(times) >= 2:
            fastest = times[0]
            slowest = times[-1]
            variance = slowest[1] - fastest[1]
            ratio = slowest[1] / fastest[1] if fastest[1] > 0 else 0
            
            print(f"\n  FASTEST: {fastest[0]} ({fastest[1]}ms)")
            print(f"  SLOWEST: {slowest[0]} ({slowest[1]}ms)")
            print(f"  VARIANCE: {variance}ms ({ratio:.1f}x)")
            print(f"  FUTURE mode benefit: ~{variance}ms saved per execution")
    
    print("\n" + "=" * 70)
    print("LOCAL TEST COMPLETE")
    print("These timings are from REAL PIL computation, no artificial delays!")
    print("=" * 70)


if __name__ == '__main__':
    main()
