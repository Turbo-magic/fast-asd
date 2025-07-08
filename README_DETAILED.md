# Active Speaker Detection System

## Overview

This is a state-of-the-art active speaker detection system built on the Sieve platform that analyzes video content to identify who is speaking at any given moment. The system combines face detection, scene segmentation, and speaker detection to provide comprehensive analysis of video content.

## What It Does

The system processes video files and returns three types of data:
1. **Scene Shots**: Video segments where scenes change
2. **Face Detections**: Locations of faces throughout the video
3. **Active Speakers**: Who is speaking at specific moments

## Architecture Overview

The system uses a multi-stage pipeline approach:

```
Video Input → Scene Detection → Face Detection → Speaker Detection → Results
```

### Key Components

- **Scene Detection**: Divides video into logical segments
- **Face Detection**: Uses YOLOv8 to identify faces in each scene
- **Speaker Detection**: Uses TalkNet-ASD to determine who is speaking
- **Data Processing**: Combines and formats results for output

## Step-by-Step Process

### 1. Input Processing & Validation
- **File Format Handling**: 
  - Converts `.webm` files to `.mp4` using FFmpeg
  - Adds `.mp4` extension to files without extensions
  - Validates input parameters (start_time, end_time, etc.)

### 2. Video Analysis Setup
- **Video Properties**: Extracts dimensions, length, and FPS
- **Parameter Validation**: Ensures start/end times are within video bounds
- **Model Configuration**: Sets up YOLOv8-face for face detection

### 3. Scene Detection
- **Threading**: Runs scene detection in a separate thread for efficiency
- **Threshold-based**: Uses a 15.0 threshold to detect scene changes
- **Segment Creation**: Divides video into logical segments based on scene cuts

### 4. Detection Interval Planning
The system creates two types of detection intervals:

#### Face Detection Intervals
- **Frequency**: Every second throughout each scene
- **Duration**: 0.1-second intervals around each second mark
- **Purpose**: Provides continuous face position tracking

#### Speaker Detection Intervals
- **Frequency**: Only the first second of each scene
- **Duration**: 1 second per scene
- **Purpose**: Determines who is speaking at the beginning of each scene

### 5. Object Detection (Face Detection)
- **Model**: Uses YOLOv8-face model
- **Processing**: Runs at specified FPS (default: 2 FPS)
- **Confidence**: Filters results with 0.5 confidence threshold
- **Output**: Returns bounding boxes for detected faces

### 6. Speaker Detection
- **Model**: Uses TalkNet-ASD (Active Speaker Detection)
- **Input**: Face detection results converted to string format
- **Processing**: Analyzes audio-visual correlation to determine speaking status
- **Output**: Returns speaking scores for each detected face

### 7. Data Processing & Formatting

#### Face Data Processing
- **Coordinate Conversion**: Converts pixel coordinates to percentages
- **Filtering**: Removes low-confidence detections (< 0.3)
- **Size Filtering**: Applies face size threshold
- **Limiting**: Keeps only the largest faces (up to max_num_faces)

#### Speaker Data Processing
- **Score Analysis**: Processes raw speaking scores
- **Coordinate Conversion**: Converts to percentage-based coordinates
- **Active Status**: Determines if person is actively speaking (score > 0)
- **Size Sorting**: Orders by face size (largest first)

### 8. Error Handling & Retry Logic
- **Automatic Retries**: Up to 10 retry attempts for failed detections
- **Graceful Degradation**: Continues processing even if some intervals fail
- **Logging**: Provides detailed progress and error information

## Output Format

The system returns a structured JSON object with three main sections:

### Shots (Scene Segments)
```json
{
  "startTimestamp": 1000,
  "endTimestamp": 5000,
  "startFrame": 30,
  "endFrame": 150,
  "shotSegment": {
    "Confidence": 99.92,
    "Index": 0
  },
  "type": "SHOT"
}
```

### Faces (Face Detections)
```json
{
  "frame_number": 60,
  "timestamp": 2000,
  "faces": [
    {
      "x1": 0.2,
      "y1": 0.3,
      "x2": 0.4,
      "y2": 0.7,
      "confidence": 0.95
    }
  ],
  "scene_number": 0
}
```

### Speakers (Active Speaker Detections)
```json
{
  "frame_number": 30,
  "timestamp": 1000,
  "speakers": [
    {
      "x1": 0.2,
      "y1": 0.3,
      "x2": 0.4,
      "y2": 0.7,
      "speaking_score": 0.85,
      "active": true
    }
  ],
  "scene_number": 0
}
```

## Key Parameters

### Input Parameters
- `file`: Video file to process
- `speed_boost`: Use faster but less accurate detection (default: false)
- `max_num_faces`: Maximum faces to return per frame (default: 5)
- `start_time`: Start processing time in seconds (default: 0)
- `end_time`: End processing time in seconds (default: -1 = end of video)
- `processing_fps`: Detection processing framerate (default: 2)
- `face_size_threshold`: Minimum face size threshold (default: 0.4)

### Model Configuration
- **Face Detection**: `sieve/yolov8` with YOLOv8-face model
- **Speaker Detection**: `sieve/talknet-asd` for active speaker detection
- **Scene Detection**: Custom scene detection with 15.0 threshold

## Performance Optimizations

### Efficient Processing Strategy
1. **Selective Detection**: Only processes first second of each scene for speaker detection
2. **Sampled Face Detection**: Processes every second for face tracking
3. **Parallel Processing**: Uses threading for scene detection
4. **Memory Management**: Implements in-memory thresholds for large videos

### Error Recovery
- **Automatic Retries**: Failed detections are retried up to 10 times
- **Confidence Adjustments**: Increases confidence thresholds on retries
- **Graceful Degradation**: Continues processing even with partial failures

## Use Cases

This system is ideal for:
- **Video Content Analysis**: Understanding who speaks when in videos
- **Meeting Recordings**: Identifying active speakers in conference calls
- **Educational Content**: Tracking instructor vs. student speaking patterns
- **Entertainment**: Analyzing dialogue distribution in movies/shows
- **Accessibility**: Creating speaker-aware video annotations

## Technical Requirements

### Dependencies
- Python 3.9
- OpenCV 4.7.0.72
- FFmpeg
- NumPy 1.23.5
- FilterPy 1.4.5
- SceneDetect with OpenCV support

### System Packages
- FFmpeg for video processing
- OpenGL libraries for computer vision
- GLib for system utilities

## Limitations & Considerations

### Performance
- Processing time scales with video length and complexity
- Large videos may require significant computational resources
- Speaker detection is limited to first second of each scene

### Accuracy
- Face detection accuracy depends on video quality and face visibility
- Speaker detection requires clear audio-visual correlation
- Scene detection may miss subtle scene changes

### File Support
- Primarily designed for MP4 format
- WebM files are automatically converted
- Other formats may require preprocessing

## Future Enhancements

Potential improvements could include:
- **Continuous Speaker Detection**: Process entire scenes, not just first second
- **Multi-modal Analysis**: Combine with audio-only speaker diarization
- **Real-time Processing**: Stream processing capabilities
- **Enhanced Accuracy**: Integration with more advanced models
- **Custom Thresholds**: Per-video adaptive parameter tuning 