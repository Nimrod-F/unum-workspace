"""
Create and upload a test image to S3 for benchmarking.
"""
import boto3
from PIL import Image, ImageDraw
from io import BytesIO
import sys

# Configuration
BUCKET = "unum-benchmark-images"
KEY = "test-images/sample-1920x1080.jpg"
PROFILE = "research-profile"
REGION = "eu-central-1"


def create_test_image(width=1920, height=1080):
    """Create a realistic test image with patterns for edge detection."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    
    # Gradient background
    for x in range(width):
        for y in range(height):
            r = int((x / width) * 200)
            g = int((y / height) * 200)
            b = int(((x + y) / (width + height)) * 200)
            pixels[x, y] = (r + 30, g + 30, b + 30)
    
    # Add shapes for edge detection
    draw = ImageDraw.Draw(img)
    
    # Rectangles
    for i in range(15):
        x1 = (i * 120) % width
        y1 = (i * 70) % height
        color = ((i * 37) % 256, (i * 73) % 256, (i * 113) % 256)
        draw.rectangle([x1, y1, x1+80, y1+60], fill=color, outline='white')
    
    # Circles
    for i in range(10):
        x = (i * 180 + 50) % width
        y = (i * 100 + 50) % height
        color = ((i * 53) % 256, (i * 97) % 256, (i * 137) % 256)
        draw.ellipse([x, y, x+100, y+100], fill=color, outline='black')
    
    # Lines for edge detection
    for i in range(20):
        x1, y1 = (i * 90) % width, 0
        x2, y2 = (i * 90 + 200) % width, height
        draw.line([(x1, y1), (x2, y2)], fill='white', width=2)
    
    return img


def upload_to_s3(img, bucket, key, profile, region):
    """Upload image to S3."""
    session = boto3.Session(profile_name=profile, region_name=region)
    s3 = session.client('s3')
    
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType='image/jpeg'
    )
    
    return len(buffer.getvalue())


def main():
    print("Creating test image...")
    img = create_test_image(1920, 1080)
    
    print(f"Uploading to s3://{BUCKET}/{KEY}...")
    size = upload_to_s3(img, BUCKET, KEY, PROFILE, REGION)
    
    print(f"✓ Uploaded {size:,} bytes")
    print(f"✓ Image ready at s3://{BUCKET}/{KEY}")


if __name__ == '__main__':
    main()
