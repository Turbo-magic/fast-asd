#!/usr/bin/env python3
"""
Local Face Detection System
Runs without Sieve dependencies using S3FD face detector
"""

import os
import sys
import cv2
import numpy as np
import threading
import queue
import json
from pathlib import Path

# Add talknet directory to Python path
talknet_path = os.path.join(os.path.dirname(__file__), 'talknet')
sys.path.insert(0, talknet_path)

# Import local modules
from utils import get_video_dimensions, get_video_length, create_video_segments
from custom_types import VideoSegment, Frame, Box

# No additional imports needed - using S3FD from TalkNet

# Configuration
FACE_SIZE_THRESHOLD = 0.4
MAX_NUM_FACES = 5
PROCESSING_FPS = 2

class LocalFaceDetector:
    """Local face detection using TalkNet's S3FD face detector"""
    
    def __init__(self):
        print("Initializing S3FD face detector...")
        # Import TalkNet's S3FD face detector
        from demoTalkNet import initialize_detector
        self.face_detector = initialize_detector()
        print("S3FD face detector loaded successfully!")
        
    def detect_faces(self, video_path, start_frame, end_frame, confidence_threshold=0.5):
        """
        Detect faces using S3FD face detector
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        results = []
        current_frame = start_frame
        
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame for S3FD detector
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use S3FD to detect faces
            bboxes = self.face_detector.detect_faces(frame_rgb, conf_th=confidence_threshold, scales=[0.25])
            
            boxes = []
            for bbox in bboxes:
                # S3FD returns [x1, y1, x2, y2, confidence]
                if len(bbox) >= 5:
                    x1, y1, x2, y2, conf = bbox[:5]
                    boxes.append({
                        'x1': float(x1),
                        'y1': float(y1), 
                        'x2': float(x2),
                        'y2': float(y2),
                        'confidence': float(conf),
                        'class_name': 'face'
                    })
            
            results.append({
                'frame_number': current_frame,
                'boxes': boxes
            })
            
            current_frame += 1
            
        cap.release()
        return results


def process_video_local(
    video_path: str,
    speed_boost: bool = False,
    max_num_faces: int = 5,
    start_time: float = 0,
    end_time: float = -1,
    processing_fps: float = 2,
    face_size_threshold: float = 0.4,
):
    """
    Local implementation of face detection with scene detection
    """
    
    # Handle file format conversion (keep existing logic)
    if video_path.endswith(".webm"):
        print("Converting webm file to mp4...")
        import subprocess
        new_path = video_path.replace(".webm", ".mp4")
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-c", "copy", new_path])
        video_path = new_path
    
    # Add .mp4 extension if missing
    if not any(video_path.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.flv', '.mkv', '.wmv', '.mpg', '.mpeg', '.m4v', '.webm']):
        print("Adding .mp4 extension to file without extension...")
        new_path = video_path + ".mp4"
        if not os.path.exists(new_path):
            import shutil
            shutil.copy2(video_path, new_path)
        video_path = new_path

    # Get video properties
    width, height = get_video_dimensions(video_path)
    original_video_length = get_video_length(video_path)
    
    if end_time == -1:
        end_time = original_video_length

    # Validate time parameters
    if start_time < 0 or start_time > original_video_length:
        raise ValueError(f"start_time must be between 0 and {original_video_length}")
    if end_time < 0 or end_time > original_video_length:
        raise ValueError(f"end_time must be between 0 and {original_video_length}")
    if start_time >= end_time:
        raise ValueError(f"start_time must be less than end_time")
    
    # Create video segment
    original_video = VideoSegment(
        path=video_path,
        start=start_time,
        end=end_time,
    )
    original_video_fps = original_video.fps()

    # Scene detection (reuse existing logic)
    scene_detection_result = queue.Queue()
    
    def scene_detection_wrapper(video_path, result_queue, **kwargs):
        print("Cutting video into scene segments...")
        from scene_detection import scene_detection
        result = list(scene_detection(video_path, **kwargs))
        result_queue.put(result)
        print("Done cutting video into scene segments")
    
    scene_detection_thread = threading.Thread(
        target=scene_detection_wrapper, 
        args=(video_path, scene_detection_result), 
        kwargs={'threshold': 25.0}
    )
    scene_detection_thread.start()
    scene_detection_thread.join()
    scene_future = scene_detection_result.get()

    segments = create_video_segments(
        video_path, scene_future, 
        start_time=start_time, end_time=end_time, 
        fps=original_video_fps, original_video_length=original_video_length
    )

    # Initialize face detector
    face_detector = LocalFaceDetector()

    # Calculate face detection intervals - every second throughout the video
    face_detection_intervals = []
    
    current_time = start_time
    while current_time < end_time:
        frame_time = current_time
        start_frame = int(frame_time * original_video_fps)
        end_frame = start_frame
        
        face_detection_intervals.append({
            'start_time': frame_time,
            'end_time': frame_time,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'segment_index': -1,
            'type': 'face'
        })
        
        current_time += 1

    print("------------------")
    print("Video Information")
    print("Video Length: {:.2f}s".format(original_video_length))
    print("Video FPS: {:.2f}".format(original_video_fps))
    print("Start Time: ", start_time)
    print("End Time: ", end_time)
    print("Number of Scenes: ", len(segments))
    print(f"Number of Face Detection Intervals: {len(face_detection_intervals)}")
    print("------------------")
    print("Processing video...")

    # Prepare scene shots output
    shots = []
    for segment_index, segment in enumerate(segments):
        shots.append({
            "startTimestamp": round(segment.start * 1000),
            "endTimestamp": round(segment.end * 1000),
            "startFrame": segment.start_frame if segment.start_frame else int(segment.start * original_video_fps),
            "endFrame": (segment.end_frame if segment.end_frame else int(segment.end * original_video_fps)) - 1,
            "shotSegment": {
                "Confidence": 99.92,
                "Index": segment_index
            },
            "type": "SHOT"
        })

    # Process face detection results (every second) to get face positions
    faces = []
    print(f"Processing {len(face_detection_intervals)} face detection intervals...")
    
    for face_idx, face_interval in enumerate(face_detection_intervals):
        print(f"Processing face detection {face_idx + 1}/{len(face_detection_intervals)} at {face_interval['start_time']:.2f}s")
        
        try:
            # Use local face detector
            target_frame = face_interval['start_frame']
            start_frame = max(0, target_frame - 1)
            end_frame = target_frame + 1
            
            face_detection_result = face_detector.detect_faces(
                video_path, start_frame, end_frame, confidence_threshold=0.5
            )
            
        except Exception as e:
            print(f"WARNING: Face detection failed for interval {face_idx}: {e}")
            continue
        
        # Process results (keep existing processing logic)
        for frame in face_detection_result:
            frame_number = frame["frame_number"]
            
            if frame_number != target_frame:
                continue
                
            face_boxes = []
            
            for box in frame["boxes"]:
                if box.get('class_name') != 'face':
                    continue
                if box.get('confidence', 0) <= 0.3:
                    continue
                    
                # Convert to percentages
                x1_pct = max(0, box["x1"]) / width
                y1_pct = max(0, box["y1"]) / height  
                x2_pct = min(width, box["x2"]) / width
                y2_pct = min(height, box["y2"]) / height
                
                face_boxes.append({
                    "x1": x1_pct,
                    "y1": y1_pct,
                    "x2": x2_pct,
                    "y2": y2_pct,
                    "confidence": box.get('confidence', 0),
                })
            
            # Sort by size and limit
            face_boxes = sorted(face_boxes, key=lambda x: (x['x2'] - x['x1']) * (x['y2'] - x['y1']), reverse=True)
            if len(face_boxes) > max_num_faces:
                face_boxes = face_boxes[:max_num_faces]
            
            if face_boxes:
                # Determine scene number
                scene_number = -1
                for seg_idx, segment in enumerate(segments):
                    if (frame_number / original_video_fps >= segment.start and 
                        frame_number / original_video_fps < segment.end):
                        scene_number = seg_idx
                        break
                
                faces.append({
                    "frame_number": frame_number,
                    "timestamp": round(frame_number / original_video_fps * 1000),
                    "faces": face_boxes,
                    "scene_number": scene_number
                })
                break

    # Return results with only shots and faces
    return {
        "shots": shots,
        "faces": faces
    }

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Local Face Detection with Scene Detection')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--start_time', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--end_time', type=float, default=-1, help='End time in seconds')
    parser.add_argument('--max_faces', type=int, default=5, help='Maximum faces to detect')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file {args.video_path} not found")
        return
    
    print(f"Processing video: {args.video_path}")
    result = process_video_local(
        args.video_path,
        start_time=args.start_time,
        end_time=args.end_time,
        max_num_faces=args.max_faces
    )
    
    print("\n=== RESULTS ===")
    print(f"Shots: {len(result['shots'])}")
    print(f"Face detections: {len(result['faces'])}")
    
    # Write results to output.txt
    output_file = "output.txt"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {output_file}")
    
    # Show sample results
    if result['faces']:
        print(f"\nSample face detection: {result['faces'][0]}")
    else:
        print("\nNo faces detected")

if __name__ == "__main__":
    main()