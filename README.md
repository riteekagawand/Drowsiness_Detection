# Drowsiness Detection (OpenCV / dlib)

Real-time drowsiness monitoring via webcam with an audible alert.

## Features
- OpenCV-based detector with 3 tiers: Haar -> optional DNN -> simple heuristic fallback
- Optional dlib 68-landmark EAR method (`Drowsiness_Detection.py`)
- Audible alarm using `pygame` when drowsiness persists
- Automatic calibration and smoothing (in `Drowsiness_Detection_OpenCV.py`)

## Project Structure
- `Drowsiness_Detection_OpenCV.py`: OpenCV-only pipeline with calibration and fallbacks
- `models/shape_predictor_68_face_landmarks.dat`: dlib landmarks model (used by `Drowsiness_Detection.py`)
- `music.wav`: alert sound played on drowsiness
- `requirement.txt`: Python dependencies

## Requirements
- Python 3.9+ recommended
- Webcam
- Windows/macOS/Linux

## Quick start (clone, install, run)
```bash
# 1) Clone the repository
git clone https://github.com/riteekagawand/Drowsiness_Detection
cd drowsiness-detection

# 2) (Optional) Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# or Windows (CMD)
.\.venv\Scripts\activate.bat
# or Git Bash
source .venv/Scripts/activate

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirement.txt

# 4) Run the OpenCV version (recommended)
python Drowsiness_Detection_OpenCV.py

