"""
evaluate_dataset.py
===================
Benchmark audio deepfake detector on a test dataset.

Expected structure:
  ~/Downloads/DEEPFAKE_audio/Dataset/Test/
      Fake/
      Real/

Results saved to results/ as CSV.
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import importlib.util

# Add AudioDetection to path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Import inference using absolute path
_inference_path = os.path.join(_HERE, 'inference.py')
_spec = importlib.util.spec_from_file_location('audio_inference', _inference_path)
_audio_inference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audio_inference)
analyze_audio = _audio_inference.analyze_audio

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path.home() / "Downloads" / "DEEPFAKE_audio" / "Dataset" / "Test"
RESULTS_DIR = Path(_HERE) / "results"
LIMIT = 100  # Set to None to process all files

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def evaluate():
    if not DATASET_ROOT.exists():
        print(f"❌ Dataset not found: {DATASET_ROOT}")
        print("\nExpected structure:")
        print("  ~/Downloads/DEEPFAKE_audio/Dataset/Test/")
        print("      Fake/")
        print("      Real/")
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    # Collect files
    fake_files = [f for f in (DATASET_ROOT / "Fake").glob("*") if f.suffix.lower() in AUDIO_EXTS]
    real_files = [f for f in (DATASET_ROOT / "Real").glob("*") if f.suffix.lower() in AUDIO_EXTS]

    if LIMIT:
        fake_files = fake_files[:LIMIT]
        real_files = real_files[:LIMIT]

    all_files = [(f, "Fake") for f in fake_files] + [(f, "Real") for f in real_files]
    
    print(f"\n📊 Evaluating Audio Detector")
    print(f"   Fake: {len(fake_files)} files")
    print(f"   Real: {len(real_files)} files")
    print(f"   Total: {len(all_files)} files\n")

    results = []
    correct = 0

    for audio_path, ground_truth in tqdm(all_files, desc="Processing"):
        try:
            result = analyze_audio(str(audio_path))
            predicted = result["verdict"]
            
            is_correct = (predicted == ground_truth)
            if is_correct:
                correct += 1

            results.append({
                "file": audio_path.name,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "fake_prob": result["fake_prob"],
                "real_prob": result["real_prob"],
                "confidence": result["confidence"],
                "correct": is_correct,
            })

        except Exception as e:
            print(f"\n⚠️  Error processing {audio_path.name}: {e}")
            results.append({
                "file": audio_path.name,
                "ground_truth": ground_truth,
                "predicted": "ERROR",
                "fake_prob": 0.0,
                "real_prob": 0.0,
                "confidence": 0.0,
                "correct": False,
            })

    # Calculate metrics
    df = pd.DataFrame(results)
    accuracy = correct / len(all_files) if all_files else 0.0

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"audio_eval_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"📊 Results")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(all_files)})")
    print(f"\nResults saved to: {csv_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    evaluate()
