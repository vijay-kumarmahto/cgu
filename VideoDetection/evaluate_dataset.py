import os
import time
import pandas as pd
from pathlib import Path
from inference import predict_video
from models import GenConViTDetector

DATASET_PATH = Path.home() / "Downloads" / "DEEPFAKE_videos" / "Test"
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

def evaluate_dataset():
    detector = GenConViTDetector()
    
    fake_dir = DATASET_PATH / "Fake"
    real_dir = DATASET_PATH / "Real"
    
    if not fake_dir.exists() or not real_dir.exists():
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return
    
    results = []
    correct = 0
    total = 0
    
    print("\n📊 Evaluating Fake videos...")
    for video_file in fake_dir.iterdir():
        if video_file.suffix.lower() in VIDEO_EXTENSIONS:
            try:
                start = time.time()
                label, confidence, score = predict_video(str(video_file))
                elapsed = time.time() - start
                
                is_correct = (label == "FAKE")
                correct += is_correct
                total += 1
                
                results.append({
                    'filename': video_file.name,
                    'true_label': 'FAKE',
                    'predicted_label': label,
                    'confidence': f"{confidence:.2f}%",
                    'raw_score': f"{score:.4f}",
                    'correct': is_correct,
                    'time_sec': f"{elapsed:.2f}"
                })
                
                print(f"  {video_file.name}: {label} ({confidence:.1f}%) - {'✓' if is_correct else '✗'}")
            except Exception as e:
                print(f"  ❌ Error processing {video_file.name}: {e}")
    
    print("\n📊 Evaluating Real videos...")
    for video_file in real_dir.iterdir():
        if video_file.suffix.lower() in VIDEO_EXTENSIONS:
            try:
                start = time.time()
                label, confidence, score = predict_video(str(video_file))
                elapsed = time.time() - start
                
                is_correct = (label == "REAL")
                correct += is_correct
                total += 1
                
                results.append({
                    'filename': video_file.name,
                    'true_label': 'REAL',
                    'predicted_label': label,
                    'confidence': f"{confidence:.2f}%",
                    'raw_score': f"{score:.4f}",
                    'correct': is_correct,
                    'time_sec': f"{elapsed:.2f}"
                })
                
                print(f"  {video_file.name}: {label} ({confidence:.1f}%) - {'✓' if is_correct else '✗'}")
            except Exception as e:
                print(f"  ❌ Error processing {video_file.name}: {e}")
    
    df = pd.DataFrame(results)
    output_csv = Path(__file__).parent / "evaluation_results.csv"
    df.to_csv(output_csv, index=False)
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📈 RESULTS")
    print(f"{'='*50}")
    print(f"Total videos: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\n✅ Results saved to: {output_csv}")

if __name__ == "__main__":
    evaluate_dataset()
