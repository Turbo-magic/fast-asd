#!/bin/bash
# Installation script for Local Face Detection System on Amazon Linux 2023

set -e  # Exit on any error

echo "🚀 Installing Local Face Detection System on Amazon Linux 2023..."

# Update system
echo "📦 Updating system packages..."
sudo dnf update -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo dnf install -y \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    make \
    cmake \
    git \
    wget \
    unzip

# Install FFmpeg (from RPM Fusion or build from source)
# echo "🎬 Installing FFmpeg..."
# if ! command -v ffmpeg &> /dev/null; then
#     # Try installing from Amazon Linux extras first
#     sudo dnf install -y amazon-linux-extras
#     sudo amazon-linux-extras install -y epel || true
    
#     # Install FFmpeg
#     sudo dnf install -y ffmpeg ffmpeg-devel || {
#         echo "⚠️  FFmpeg not available in repos, installing from source..."
#         cd /tmp
#         wget -O ffmpeg-release-amd64-static.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
#         tar -xf ffmpeg-release-amd64-static.tar.xz
#         sudo cp ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/
#         sudo cp ffmpeg-*-amd64-static/ffprobe /usr/local/bin/
#         sudo chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe
#         rm -rf ffmpeg-*
#     }
# fi

# Install OpenGL and graphics libraries
echo "🖥️  Installing graphics libraries..."
sudo dnf install -y \
    mesa-libGL-devel \
    mesa-libGLU-devel \
    libX11-devel \
    libXext-devel \
    libXrender-devel \
    libICE-devel \
    libSM-devel \
    ncurses-compat-libs \
    glib2-devel

# Install OpenCV dependencies
echo "📸 Installing OpenCV dependencies..."
sudo dnf install -y \
    opencv-devel \
    opencv-python3 || {
        echo "⚠️  OpenCV not available, will install via pip"
    }

# Create project directory if it doesn't exist
PROJECT_DIR="/opt/face-detection"
echo "📁 Setting up project directory at $PROJECT_DIR..."
sudo mkdir -p $PROJECT_DIR
sudo chown $USER:$USER $PROJECT_DIR

# Navigate to project directory
cd $PROJECT_DIR

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "📦 Installing Python dependencies..."

# Install core dependencies first
pip install numpy>=1.24.0
pip install scipy>=1.10.0

# Install computer vision libraries
pip install opencv-python>=4.8.0

# Install PyTorch (CPU version for compatibility)
echo "🧠 Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ML and video processing libraries
pip install scikit-learn>=1.3.0
pip install pandas>=2.0.0
pip install tqdm

# Install video processing dependencies
pip install "imageio[ffmpeg]"
pip install "av>=10.0.0"

# Install face detection dependencies
pip install python_speech_features
pip install gdown

# Install other required packages
pip install pydantic>=2.0.0
pip install filterpy>=1.4.5
pip install "scenedetect[opencv]"
pip install lap>=0.4.0
pip install sortedcontainers
pip install supervision
pip install "vidgear[core]"

# Download TalkNet pretrained model
echo "📥 Downloading TalkNet pretrained model..."
mkdir -p ~/.cache/models
if [ ! -f ~/.cache/models/pretrain_TalkSet.model ]; then
    echo "Downloading TalkNet model (this may take a few minutes)..."
    gdown --id 1J-PDWDAkYCdT8T2Nxn3Q_-iOHH_t-9YP -O ~/.cache/models/pretrain_TalkSet.model
    echo "✅ TalkNet model downloaded successfully!"
else
    echo "✅ TalkNet model already exists"
fi

# Create models directory and download S3FD face detection model
echo "📥 Downloading S3FD face detection model..."
mkdir -p model/faceDetector/s3fd
if [ ! -f model/faceDetector/s3fd/sfd_face.pth ]; then
    echo "Downloading S3FD face model (this may take a few minutes)..."
    wget -O model/faceDetector/s3fd/sfd_face.pth https://storage.googleapis.com/mango-public-models/sfd_face.pth
    echo "✅ S3FD face model downloaded successfully!"
else
    echo "✅ S3FD face model already exists"
fi

# Set proper permissions
echo "🔐 Setting permissions..."
chmod +x local_main.py || echo "⚠️  local_main.py not found - copy your files to $PROJECT_DIR"

# Create systemd service file (optional)
echo "⚙️  Creating systemd service template..."
sudo tee /etc/systemd/system/face-detection.service > /dev/null <<EOF
[Unit]
Description=Local Face Detection Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python local_main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Systemd service template created at /etc/systemd/system/face-detection.service"

# Create convenience script
echo "📝 Creating convenience scripts..."
cat > run_face_detection.sh << 'EOF'
#!/bin/bash
# Convenience script to run face detection with virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Run face detection with all arguments passed through
python local_main.py "$@"
EOF

chmod +x run_face_detection.sh

# Create installation verification script
cat > verify_installation.py << 'EOF'
#!/usr/bin/env python3
"""Verify installation of face detection system"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    tests = [
        ("numpy", "import numpy"),
        ("opencv", "import cv2"),
        ("torch", "import torch"),
        ("pydantic", "import pydantic"),
        ("scenedetect", "import scenedetect"),
        ("gdown", "import gdown"),
    ]
    
    print("🧪 Testing package imports...")
    failed = []
    
    for name, import_cmd in tests:
        try:
            exec(import_cmd)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0

def test_models():
    """Test if required models are available"""
    print("\n🧪 Testing model availability...")
    
    talknet_path = os.path.expanduser("~/.cache/models/pretrain_TalkSet.model")
    s3fd_path = "model/faceDetector/s3fd/sfd_face.pth"
    
    success = True
    
    if os.path.exists(talknet_path):
        print("✅ TalkNet model found")
    else:
        print(f"❌ TalkNet model not found at {talknet_path}")
        success = False
    
    if os.path.exists(s3fd_path):
        print("✅ S3FD face model found")
    else:
        print(f"❌ S3FD face model not found at {s3fd_path}")
        success = False
    
    return success

def test_system_deps():
    """Test system dependencies"""
    import subprocess
    
    print("\n🧪 Testing system dependencies...")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ ffmpeg available")
        else:
            print("❌ ffmpeg not working")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ ffmpeg not found or not responding")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def main():
    print("🔍 Installation Verification")
    print("=" * 40)
    
    all_good = True
    
    if not test_system_deps():
        all_good = False
    
    if not test_imports():
        all_good = False
    
    if not test_models():
        all_good = False
    
    print("\n" + "=" * 40)
    if all_good:
        print("🎉 All tests passed! Installation is ready.")
        print(f"📍 Project location: {os.getcwd()}")
        print("🚀 Run: ./run_face_detection.sh your_video.mp4")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x verify_installation.py

deactivate

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📍 Project installed at: $PROJECT_DIR"
echo ""
echo "Next steps:"
echo "1. Copy your face detection files to: $PROJECT_DIR"
echo "2. Test installation: cd $PROJECT_DIR && python verify_installation.py"
echo "3. Run face detection: ./run_face_detection.sh your_video.mp4"
echo ""
echo "📋 Files to copy to $PROJECT_DIR:"
echo "   - local_main.py"
echo "   - utils.py"
echo "   - custom_types.py"
echo "   - scene_detection.py"
echo "   - talknet/ (entire directory)"
echo ""
echo "🔧 Optional: Enable service with 'sudo systemctl enable face-detection'"