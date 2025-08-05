# Local Active Speaker Detection

This is a standalone version of the active speaker detection system that runs entirely on your local machine without requiring Sieve infrastructure.

## 🚀 Quick Start

### 1. Setup (one-time)
```bash
# Install system dependencies
sudo /usr/bin/apt update
sudo /usr/bin/apt install -y ffmpeg python3-full python3-dev libgl1-mesa-dev libglib2.0-dev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements_compatible.txt

# Download TalkNet model
mkdir -p ~/.cache/models
gdown --id 1J-PDWDAkYCdT8T2Nxn3Q_-iOHH_t-9YP -O ~/.cache/models/pretrain_TalkSet.model
```

### 2. Run Detection

#### Option A: Direct execution
```bash
# Activate virtual environment
source venv/bin/activate

# Run detection
python local_main.py your_video.mp4
```

#### Option B: Convenience script (recommended)
```bash
# Run directly (automatically uses virtual environment)
./run_local.py your_video.mp4
```

### 3. Command Line Options
```bash
# Basic usage
./run_local.py video.mp4

# Process specific time range
./run_local.py video.mp4 --start_time 10 --end_time 60

# Limit maximum faces detected per frame
./run_local.py video.mp4 --max_faces 3

# Help
./run_local.py --help
```

## 🧠 Models Used

- **Face Detection**: YOLOv8-face (auto-downloads ~6MB)
- **Speaker Detection**: TalkNet pretrained model (~100MB, downloads once)

## 📊 Output Format

The system returns a JSON structure with three main components:

```json
{
  "shots": [
    {
      "startTimestamp": 0,
      "endTimestamp": 5000,
      "startFrame": 0,
      "endFrame": 125,
      "shotSegment": {"Confidence": 99.92, "Index": 0},
      "type": "SHOT"
    }
  ],
  "faces": [
    {
      "frame_number": 30,
      "timestamp": 1200,
      "faces": [
        {
          "x1": 0.2, "y1": 0.3, "x2": 0.4, "y2": 0.7,
          "confidence": 0.95
        }
      ],
      "scene_number": 0
    }
  ],
  "speakers": [
    {
      "frame_number": 30,
      "timestamp": 1200,
      "faces": [
        {
          "x1": 0.2, "y1": 0.3, "x2": 0.4, "y2": 0.7,
          "speaking_score": 0.85,
          "active": true
        }
      ],
      "scene_number": 0
    }
  ]
}
```

## 🔧 Integration

### Python API
```python
from local_main import process_video_local

# Process a video
result = process_video_local(
    video_path="path/to/video.mp4",
    start_time=0,
    end_time=30,
    max_num_faces=5,
    face_size_threshold=0.4
)

print(f"Detected {len(result['faces'])} face instances")
print(f"Detected {len(result['speakers'])} speaker instances")
print(f"Found {len(result['shots'])} scene shots")
```

## 🐛 Troubleshooting

### Setup Issues

**Virtual environment creation fails:**
```bash
# Install python3-venv if missing
sudo apt install python3-venv python3-pip
```

**Package installation fails:**
```bash
# Update pip in virtual environment
source venv/bin/activate
pip install --upgrade pip
```

**System dependencies missing:**
```bash
# Install manually
sudo apt update
sudo apt install ffmpeg libgl1-mesa-dev libglib2.0-dev
```

### Runtime Issues

**CUDA/GPU errors:**
- The system will automatically fall back to CPU if GPU is not available
- For better performance, ensure you have PyTorch with CUDA support

**Out of memory:**
- Reduce video length or use `--max_faces` to limit detections
- Process shorter segments at a time

**Model download fails:**
- Check internet connection
- Manually download TalkNet model:
  ```bash
  mkdir -p ~/.cache/models
  wget https://drive.google.com/uc?id=1J-PDWDAkYCdT8T2Nxn3Q_-iOHH_t-9YP -O ~/.cache/models/pretrain_TalkSet.model
  ```

## 📋 System Requirements

- **OS**: Linux (Ubuntu/Debian recommended)
- **Python**: 3.8+
- **RAM**: 4GB+ (8GB+ recommended)
- **Storage**: ~1GB for models and dependencies
- **GPU**: Optional (CUDA-capable for better performance)

## 🔄 Migration from Sieve

This local version maintains the same API and output format as the original Sieve-based system:

- ✅ Same scene detection logic
- ✅ Same face detection intervals (every second)
- ✅ Same speaker detection (first second of each scene)
- ✅ Same output JSON structure
- ✅ Same coordinate system (percentage-based)

## 📈 Performance

**Typical processing speeds:**
- Face detection: ~2-5 FPS
- Speaker detection: Real-time to 2x real-time
- Overall: Depends on video length and complexity

**Optimization tips:**
- Use shorter time segments for faster processing
- Reduce `max_faces` if you don't need to track many people
- Use GPU acceleration when available