"""
Detect Faces - SLOWEST operation (~2-5s)

Performs face detection and analysis. This is the most computationally
intensive operation and takes the longest to complete.
"""
import json
import time
import random


def lambda_handler(event, context):
    """Detect faces - slowest operation."""
    start_time = time.time()
    
    image_id = event.get('image_id', 'unknown')
    delay_factor = event.get('delay_factor', 3.5)
    width = event.get('width', 4000)
    height = event.get('height', 3000)
    
    # Slowest operation - simulates ML inference
    actual_delay = delay_factor * (0.8 + random.random() * 0.4)
    time.sleep(actual_delay)
    
    # Generate mock face detections
    num_faces = random.randint(0, 5)
    faces = []
    
    for i in range(num_faces):
        x = random.randint(0, width - 200)
        y = random.randint(0, height - 200)
        w = random.randint(80, 200)
        h = int(w * 1.2)  # Faces are typically taller
        
        faces.append({
            'face_id': i,
            'bbox': [x, y, x + w, y + h],
            'confidence': round(0.85 + random.random() * 0.15, 3),
            'landmarks': {
                'left_eye': [x + w * 0.3, y + h * 0.35],
                'right_eye': [x + w * 0.7, y + h * 0.35],
                'nose': [x + w * 0.5, y + h * 0.55],
                'mouth': [x + w * 0.5, y + h * 0.75],
            },
            'attributes': {
                'age_estimate': random.randint(20, 60),
                'gender': random.choice(['male', 'female']),
                'emotion': random.choice(['neutral', 'happy', 'surprised']),
                'glasses': random.choice([True, False]),
            }
        })
    
    result = {
        'operation': 'detect_faces',
        'image_id': image_id,
        'num_faces': num_faces,
        'faces': faces,
        'model_used': 'FaceNet-v3',
        'inference_device': 'CPU',
        'processing_time_ms': int((time.time() - start_time) * 1000),
        'timestamp': time.time()
    }
    
    print(f'[FACES] Detected {num_faces} faces in {image_id} in {result["processing_time_ms"]}ms')
    
    return result


if __name__ == '__main__':
    result = lambda_handler({'image_id': 'test', 'delay_factor': 3.5, 'width': 4000, 'height': 3000}, None)
    print(json.dumps({k: v for k, v in result.items() if k != 'faces'}, indent=2))
    print(f"Faces detected: {result['num_faces']}")
