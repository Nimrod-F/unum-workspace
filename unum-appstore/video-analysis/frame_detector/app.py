"""
Frame Detector - Performs object detection on video frames.

Simulates running a Faster R-CNN or YOLO model on frame batches.
Processing time varies based on scene complexity (number of objects).
"""
import json
import time
import random


# Simulated object classes for detection
OBJECT_CLASSES = [
    'person', 'car', 'truck', 'bicycle', 'motorcycle',
    'bus', 'traffic_light', 'stop_sign', 'dog', 'cat',
    'chair', 'table', 'bottle', 'cup', 'laptop'
]


def detect_objects_in_frame(frame, complexity_factor):
    """Simulate object detection on a single frame."""
    # Number of objects based on complexity
    n_objects = int(3 + complexity_factor * 2 + random.randint(0, 3))
    
    detections = []
    for _ in range(n_objects):
        obj_class = random.choice(OBJECT_CLASSES)
        confidence = 0.5 + random.random() * 0.5  # 0.5-1.0
        
        # Random bounding box
        x = random.randint(0, frame.get('width', 1920) - 100)
        y = random.randint(0, frame.get('height', 1080) - 100)
        w = random.randint(50, 200)
        h = random.randint(50, 200)
        
        if confidence > 0.5:  # Only include high-confidence detections
            detections.append({
                'class': obj_class,
                'confidence': round(confidence, 3),
                'bbox': [x, y, x + w, y + h]
            })
    
    return detections


def lambda_handler(event, context):
    """
    Perform object detection on a batch of frames.
    
    Processing time varies significantly based on scene complexity:
    - Simple scenes (few objects): ~0.3-0.5s
    - Medium scenes: ~1.5-2s
    - Complex scenes (many objects): ~4-6s
    """
    start_time = time.time()
    
    batch_id = event.get('batch_id', 0)
    video_id = event.get('video_id', 'unknown')
    frames = event.get('frames', [])
    processing_delay = event.get('processing_delay', 1.0)
    
    # Simulate variable processing time
    # Add some randomness to make it more realistic
    actual_delay = processing_delay * (0.8 + random.random() * 0.4)
    time.sleep(actual_delay)
    
    # Process each frame
    all_detections = []
    for frame in frames:
        frame_detections = detect_objects_in_frame(frame, processing_delay)
        all_detections.append({
            'frame_id': frame.get('frame_id'),
            'detections': frame_detections,
            'num_detections': len(frame_detections)
        })
    
    # Aggregate statistics
    total_detections = sum(fd['num_detections'] for fd in all_detections)
    
    # Count detections by class
    class_counts = {}
    for fd in all_detections:
        for det in fd['detections']:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
    
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    print(f'[DETECTOR] Batch {batch_id}: processed {len(frames)} frames in {processing_time_ms}ms')
    print(f'[DETECTOR] Found {total_detections} objects')
    
    return {
        'batch_id': batch_id,
        'video_id': video_id,
        'n_frames_processed': len(frames),
        'total_detections': total_detections,
        'class_counts': class_counts,
        'frame_detections': all_detections,
        'processing_time_ms': processing_time_ms,
        'expected_delay': processing_delay,
        'timestamp': time.time()
    }


if __name__ == '__main__':
    result = lambda_handler({
        'batch_id': 0,
        'video_id': 'test',
        'frames': [
            {'frame_id': 0, 'width': 1920, 'height': 1080, 'objects_hint': 5},
            {'frame_id': 1, 'width': 1920, 'height': 1080, 'objects_hint': 5},
        ],
        'processing_delay': 1.0
    }, None)
    print(json.dumps({k: v for k, v in result.items() if k != 'frame_detections'}, indent=2))
