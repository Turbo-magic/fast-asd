# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a production-ready active speaker detection system with two main components:

1. **Active Speaker Detection Application** (`main.py`) - A Sieve-based function that processes videos to detect and track active speakers
2. **TalkNet Implementation** (`talknet/`) - A standalone, optimized implementation of the TalkNet model for active speaker detection

The system uses computer vision and audio processing to identify which people are speaking in video content, leveraging both face detection and speaker detection models.

## Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Testing the TalkNet Model
```bash
cd talknet
python main.py
```

### Running Active Speaker Detection Locally
```bash
# Requires Sieve API key and account setup
sieve login
python main.py
```

### Dependencies
- Install system dependencies: `ffmpeg`, `libgl1-mesa-glx`, `libglib2.0-0`
- Main Python packages: `opencv-python`, `numpy`, `scenedetect`, `supervision`

## Architecture

### Core Components

**main.py** - Main application entry point containing:
- `process()` - Primary function that orchestrates video processing
- Scene detection and segmentation logic
- Parallel processing of object detection and speaker detection
- Integration with Sieve platform for cloud-based ML models

**custom_types.py** - Pydantic models defining core data structures:
- `VideoSegment` - Represents time-based video segments
- `Box` - Bounding box with geometric operations
- `Frame` - Video frame with associated detection boxes

**utils.py** - Utility functions:
- Video metadata extraction (`get_video_dimensions`, `get_video_length`)
- Video segmentation (`create_video_segments`)
- Box tracking using ByteTrack algorithm

**scene_detection.py** - Scene change detection using PySceneDetect:
- Content-based scene detection
- Adaptive threshold options

### TalkNet Module (`talknet/`)

Standalone implementation of TalkNet active speaker detection model:
- `main.py` - Sieve model wrapper
- `demoTalkNet.py` - Core TalkNet inference logic
- `model/` - Neural network components (audio encoder, visual encoder, attention layers)
- `utils/` - Training utilities and performance evaluation

### Processing Pipeline

1. **Video Preprocessing** - Handle file format conversion (.webm to .mp4)
2. **Scene Detection** - Identify scene cuts using content-based detection
3. **Parallel Processing**:
   - Object detection for face detection using YOLOv8
   - Speaker detection using TalkNet model
4. **Post-processing** - Smoothing, tracking, and result formatting

### Key Dependencies

- **Sieve Platform** - Cloud-based ML model hosting and execution
- **PySceneDetect** - Scene change detection
- **OpenCV** - Video processing and computer vision
- **Supervision** - Object tracking and detection utilities
- **PyTorch** - Deep learning framework (TalkNet model)

## Configuration

### Model Parameters
- `SPEAKER_DETECTION_MODEL = "sieve/talknet-asd"`
- `OBJECT_DETECTION_MODEL = "sieve/yolov8"`
- `SPEAKER_DETECTION_IN_MEMORY_THRESHOLD = 3000`

### Processing Parameters
- Scene detection threshold: 15.0 (adjustable)
- Processing FPS: 2 (default)
- Face size threshold: 0.4 (minimum face size for detection)

## File Structure

```
├── main.py                     # Main application
├── custom_types.py             # Data models
├── utils.py                    # Utility functions
├── scene_detection.py          # Scene detection logic
├── requirements.txt            # Python dependencies
└── talknet/                    # TalkNet implementation
    ├── main.py                 # Sieve model wrapper
    ├── demoTalkNet.py          # Core inference
    ├── model/                  # Neural network components
    └── utils/                  # Training utilities
```

## Development Notes

- The system is designed for production use with the Sieve platform
- Local development requires Sieve API credentials
- TalkNet model supports variable frame rates (unlike original 25 FPS limitation)
- Video processing is optimized for performance with parallel execution
- Scene-based processing reduces computational overhead by only analyzing key intervals