"""
Video Decoder - Extracts frames from video for parallel processing.

Simulates decoding a video into frame batches that will be processed
in parallel by object detection functions.
"""
import json
import time
import random


def lambda_handler(event, context):
    """
    Decode video into frame batches for parallel processing.
    
    Expected event:
    {
        "video_id": "sample_video",
        "total_frames": 30,
        "batch_size": 5,
        "frame_complexity": "varied"  # "uniform" or "varied"
    }
    
    Returns frame batches with varying complexity (affects detection time).
    """
    video_id = event.get('video_id', 'sample_video')
    total_frames = event.get('total_frames', 30)
    batch_size = event.get('batch_size', 5)
    frame_complexity = event.get('frame_complexity', 'varied')
    
    start_time = time.time()
    
    # Simulate video decoding time
    time.sleep(0.3)
    
    # Calculate number of batches
    n_batches = (total_frames + batch_size - 1) // batch_size
    
    # Generate frame batches with varying complexity
    # Some frames have more objects (slower detection), some have fewer (faster)
    batches = []
    for batch_idx in range(n_batches):
        start_frame = batch_idx * batch_size
        end_frame = min(start_frame + batch_size, total_frames)
        
        # Assign complexity that affects processing time
        if frame_complexity == 'varied':
            # Create natural variance - some batches have complex scenes
            # Batch 0: Simple scene (fast) - ~0.3s
            # Batch 1: Medium scene - ~1.5s  
            # Batch 2: Complex scene (many objects) - ~4s
            # Batch 3: Medium scene - ~2s
            # Batch 4: Simple scene (fast) - ~0.5s
            # Batch 5: Very complex - ~6s
            complexity_map = {
                0: 0.3,   # Simple - fast
                1: 1.5,   # Medium
                2: 4.0,   # Complex - slow
                3: 2.0,   # Medium
                4: 0.5,   # Simple - fast
                5: 6.0,   # Very complex - slowest
            }
            base_delay = complexity_map.get(batch_idx, 1.0 + random.random() * 2)
        else:
            base_delay = 1.0
        
        # Generate mock frame data
        frames = []
        for frame_idx in range(start_frame, end_frame):
            frames.append({
                'frame_id': frame_idx,
                'width': 1920,
                'height': 1080,
                'objects_hint': int(5 + base_delay * 3),  # More delay = more objects expected
            })
        
        batches.append({
            'batch_id': batch_idx,
            'video_id': video_id,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'n_frames': len(frames),
            'frames': frames,
            'processing_delay': base_delay,  # Hint for detector
            'timestamp': time.time()
        })
    
    processing_time = int((time.time() - start_time) * 1000)
    
    print(f'[DECODER] Decoded {total_frames} frames into {n_batches} batches')
    print(f'[DECODER] Processing delays per batch: {[b["processing_delay"] for b in batches]}')
    
    return batches


if __name__ == '__main__':
    result = lambda_handler({
        "video_id": "test_video",
        "total_frames": 30,
        "batch_size": 5,
        "frame_complexity": "varied"
    }, None)
    print(f"Generated {len(result)} batches")
    for batch in result:
        print(f"  Batch {batch['batch_id']}: frames {batch['start_frame']}-{batch['end_frame']}, delay={batch['processing_delay']}s")
