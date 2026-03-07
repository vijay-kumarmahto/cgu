# 🚀 Quick Start Guide - Multi-Modal Deepfake Detector

## One-Time Setup

```bash
# 1. Run automated setup (downloads all models)
./setup.sh

# This will:
# - Create virtual environment
# - Install all dependencies
# - Download ViT + SigLIP (image models)
# - Download GenConViT-ED + VAE (video models)
# - Download Wav2Vec2 (audio model)
```

## Running the Application

```bash
# Option 1: Quick run
./run.sh

# Option 2: Manual
source .venv/bin/activate
python start.py
```

Then open **http://localhost:7860** in your browser.

---

## Using the Web Interface

### 📷 Image Tab
1. Upload an image
2. Click **Detect**
3. View results:
   - ViT prediction
   - SigLIP prediction (if used)
   - Final ensemble verdict
4. Provide feedback:
   - ✅ **Correct** - logs the prediction
   - ❌ **Wrong** - triggers automatic fine-tuning
   - 🔄 **Check Training Status** - see training progress

### 🎥 Video Tab
1. Upload a video
2. Click **Detect**
3. View results:
   - Overall verdict (FAKE/REAL)
   - Confidence score
   - Analysis based on 10 sampled frames

### 🎤 Audio Tab
1. Upload an audio file (WAV, MP3, FLAC, OGG, M4A)
2. Click **Detect**
3. View results:
   - Verdict (Fake/Real)
   - Confidence percentage
   - Probability breakdown

---

## Testing Individual Modules

### Test Audio Detection
```bash
python test_audio.py
```

### Evaluate on Datasets

**Image:**
```bash
cd ImageDetection
python evaluate_dataset.py
# Dataset: ~/Downloads/DEEPFAKE_images/Dataset/Test/
```

**Video:**
```bash
cd VideoDetection
python evaluate_dataset.py
# Dataset: ~/Downloads/DEEPFAKE_videos/Dataset/Test/
```

**Audio:**
```bash
cd AudioDetection
python evaluate_dataset.py
# Dataset: ~/Downloads/DEEPFAKE_audio/Dataset/Test/
```

---

## Configuration

### Image Detection Threshold
```bash
# Default: 0.90 (90% confidence)
DEEPFAKE_THRESHOLD=0.85 python start.py
```

### Training Parameters
Edit `ImageDetection/trainer.py`:
- `DRIFT_CHECK_EVERY = 10` - Check drift every N corrections
- `DRIFT_TOLERANCE = 0.07` - Max accuracy drop before rollback
- `MAX_PER_CLASS = 50` - Replay buffer size per class

---

## Programmatic Usage

### Image Detection
```python
from ImageDetection.ensemble import run
from PIL import Image

img = Image.open("test.jpg")
result = run(img)
print(result["verdict"])  # "Fake" or "Real"
print(result["fake_prob"])  # 0.0 - 1.0
```

### Video Detection
```python
from VideoDetection.inference import predict_video

label, confidence, score = predict_video("test.mp4")
print(f"{label}: {confidence:.2f}%")
```

### Audio Detection
```python
from AudioDetection.inference import analyze_audio

result = analyze_audio("test.wav")
print(result["verdict"])  # "Fake" or "Real"
print(result["confidence"])  # 0.0 - 1.0
```

---

## Troubleshooting

### Models Not Found
```bash
# Re-run setup to download models
./setup.sh
```

### Dependencies Missing
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Port Already in Use
```bash
# Kill existing instances
pkill -f app.py
# Or restart with different port
python app.py --server-port 7861
```

### Audio Processing Errors
```bash
# Install audio backends
pip install soundfile librosa
# On Ubuntu/Debian:
sudo apt-get install libsndfile1
```

---

## File Locations

### Models
- Image: `ImageDetection/models/vit/`, `ImageDetection/models/siglip/`
- Video: `VideoDetection/weights/`
- Audio: `AudioDetection/models/`

### Feedback Data
- Images: `ImageDetection/feedback/images/`
- CSV Log: `ImageDetection/feedback/feedback_log.csv`
- Model Backups: `ImageDetection/models/*/model_backup.safetensors`

### Results
- Image: `ImageDetection/results/`
- Video: `VideoDetection/results/`
- Audio: `AudioDetection/results/`

---

## Feature Status

| Feature | Status | Training |
|---------|--------|----------|
| 📷 Image Detection | ✅ Complete | ✅ Automatic |
| 🎥 Video Detection | ✅ Complete | ⚠️ Not implemented |
| 🎤 Audio Detection | ✅ Complete | ⚠️ Not implemented |

---

## Support

- **Documentation**: See `README.md` for detailed information
- **Audio Module**: See `AudioDetection/README.md`
- **Completion Summary**: See `AUDIO_COMPLETION.md`

---

## Quick Commands Cheat Sheet

```bash
# Setup
./setup.sh

# Run
./run.sh

# Test audio
python test_audio.py

# Evaluate image model
cd ImageDetection && python evaluate_dataset.py

# Evaluate video model
cd VideoDetection && python evaluate_dataset.py

# Evaluate audio model
cd AudioDetection && python evaluate_dataset.py

# Custom threshold
DEEPFAKE_THRESHOLD=0.85 python start.py
```

---

**🎉 You're all set! The system supports complete multi-modal deepfake detection.**
