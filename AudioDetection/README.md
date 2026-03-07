# Audio Deepfake Detection

Audio deepfake detection using **Wav2Vec2** model fine-tuned for fake/real audio classification.

---

## Model

- **Architecture**: Wav2Vec2ForSequenceClassification
- **Input**: Audio files (16kHz sample rate)
- **Labels**: 
  - Class 0: Fake
  - Class 1: Real

---

## Features

- ✅ Real-time audio analysis
- ✅ Support for multiple audio formats (WAV, MP3, FLAC, OGG, M4A)
- ✅ Confidence scores for predictions
- ⚠️ Fine-tuning not yet implemented (feedback buttons available but training disabled)

---

## Usage

### Via Web UI
```bash
python start.py
# Navigate to Audio tab, upload audio file, click Detect
```

### Programmatic
```python
from AudioDetection.inference import analyze_audio

result = analyze_audio("path/to/audio.wav")
print(result)
# {
#   "verdict": "Fake" | "Real",
#   "fake_prob": 0.85,
#   "real_prob": 0.15,
#   "confidence": 0.85
# }
```

---

## Evaluation

```bash
cd AudioDetection
python evaluate_dataset.py
```

Expected dataset structure:
```
~/Downloads/DEEPFAKE_audio/Dataset/Test/
    Fake/
    Real/
```

Results saved to `results/audio_eval_TIMESTAMP.csv`

---

## Model Details

- **Sampling Rate**: 16kHz (automatically resampled)
- **Feature Extraction**: Wav2Vec2FeatureExtractor
- **Hardware**: Auto-detects CUDA/CPU, uses FP16 on GPU, FP32 on CPU

---

## Future Enhancements

- [ ] Implement fine-tuning on user feedback
- [ ] Add spectral analysis visualization
- [ ] Support for longer audio files (chunking)
- [ ] Real-time streaming analysis
