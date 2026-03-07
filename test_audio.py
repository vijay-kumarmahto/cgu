#!/usr/bin/env python3
"""
test_audio.py
=============
Quick test to verify audio detection module is working.
"""

import sys
import os

# Add AudioDetection to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_AUDIO_DIR = os.path.join(_HERE, "AudioDetection")
if _AUDIO_DIR not in sys.path:
    sys.path.insert(0, _AUDIO_DIR)

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        print("  ✅ torch")
        import librosa
        print("  ✅ librosa")
        import soundfile
        print("  ✅ soundfile")
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        print("  ✅ transformers (Wav2Vec2)")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_model_files():
    """Test that model files exist."""
    print("\nTesting model files...")
    model_dir = os.path.join(_AUDIO_DIR, "models")
    required_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
    
    all_ok = True
    for fname in required_files:
        fpath = os.path.join(model_dir, fname)
        if os.path.isfile(fpath):
            print(f"  ✅ {fname}")
        else:
            print(f"  ❌ {fname} missing")
            all_ok = False
    
    return all_ok

def test_model_loading():
    """Test that the model can be loaded."""
    print("\nTesting model loading...")
    try:
        from models import predict_audio
        print("  ✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Audio Detection Module Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Model Files", test_model_files),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    for name, test_fn in tests:
        result = test_fn()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Audio detection is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
