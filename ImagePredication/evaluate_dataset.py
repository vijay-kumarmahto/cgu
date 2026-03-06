"""
evaluate_dataset.py
===================
Benchmarks ensemble mode on your local test dataset.
Uses hardware-aware batching:
  - CPU  → batch size 4  (cache-friendly, avoids OOM)
  - CUDA → batch size 16 (GPU parallelism, fits easily in VRAM)

Dataset expected structure:
    ~/Downloads/DEEPFAKE_images/Dataset/Test/
        Fake/
        Real/

Outputs
-------
  results/ensemble_fake_results.csv
  results/ensemble_real_results.csv
  Console: per-folder accuracy + timing + final summary table.
"""

import os
import csv
import time
import sys
from PIL import Image
from tqdm import tqdm

# ── Path setup ─────────────────────────────────────────────────────────────────
# evaluate_dataset.py is in ImagePredication/
# hardware.py is in the parent (DeepFake_Predication/)
# ensemble.py is in the same folder
_HERE       = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_HERE)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Hardware setup: suppress noise, configure torch, then load the rest ────────
import hardware
hardware.suppress_torchao_noise()
hardware.configure()

import ensemble

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.expanduser("~/Downloads/DEEPFAKE_images/Dataset/Test")
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set to an integer to cap images per folder, None to run ALL
LIMIT = 1000

# Batch size: CPU gets 4, GPU gets 16
BATCH_SIZE = 16 if hardware.PROFILE == "cuda" else 4

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
FAKE_DIR   = os.path.join(DATASET_ROOT, "Fake")
REAL_DIR   = os.path.join(DATASET_ROOT, "Real")


# ── Batch inference ────────────────────────────────────────────────────────────
def predict_batch(image_paths: list[str]) -> list[dict]:
    """
    Run ensemble inference on a batch of image paths.
    Falls back gracefully per-image on error.
    """
    results = []
    for path in image_paths:
        try:
            img    = Image.open(path).convert("RGB")
            result = ensemble.run(img)
        except Exception as exc:
            result = {
                "verdict":     f"ERROR: {exc}",
                "fake_prob":   0.0,
                "real_prob":   0.0,
                "path":        "error",
                "vit_fake":    0.0,
                "vit_real":    0.0,
                "siglip_fake": None,
                "siglip_real": None,
            }
        results.append(result)
    return results


# ── Core evaluation ────────────────────────────────────────────────────────────
def evaluate_folder(folder: str, ground_truth: str, csv_path: str) -> tuple[float, float]:
    """
    Evaluate all images in folder using ensemble mode with batching.

    Returns
    -------
    (accuracy %, avg_inference_ms)
    """
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)])

    if not files:
        print(f"  [WARN] No images found in {folder}")
        return 0.0, 0.0

    if LIMIT is not None:
        files = files[:LIMIT]

    correct    = 0
    rows       = []
    total_time = 0.0

    batches = [files[i:i + BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]

    for batch_fnames in tqdm(batches, desc=f"[ENSEMBLE] {ground_truth}", unit="batch"):
        batch_paths = [os.path.join(folder, f) for f in batch_fnames]

        t0      = time.perf_counter()
        results = predict_batch(batch_paths)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        for fname, result in zip(batch_fnames, results):
            verdict        = result["verdict"]
            fake_pct       = round(result["fake_prob"] * 100, 2)
            real_pct       = round(result["real_prob"] * 100, 2)
            inference_path = result.get("path", "ensemble")

            is_correct = verdict == ground_truth
            if is_correct:
                correct += 1

            rows.append({
                "filename"           : fname,
                "fake_percent"       : fake_pct,
                "real_percent"       : real_pct,
                "verdict"            : verdict,
                "ground_truth"       : ground_truth,
                "correct_prediction" : "Yes" if is_correct else "No",
                "inference_path"     : inference_path,
            })

    total_images = len(files)
    accuracy     = round(correct / total_images * 100, 2) if total_images else 0.0
    avg_ms       = round((total_time / total_images) * 1000, 1) if total_images else 0.0

    fieldnames = [
        "filename", "fake_percent", "real_percent",
        "verdict", "ground_truth", "correct_prediction", "inference_path",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  → {ground_truth:4s} | accuracy: {accuracy:6.2f}% | "
          f"avg: {avg_ms:.1f}ms/img | total: {total_time:.1f}s | CSV: {csv_path}")
    return accuracy, avg_ms


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hardware.summary()
    print("=" * 65)
    print("  Ensemble Deepfake Detector — Dataset Evaluation")
    print(f"  Dataset    : {DATASET_ROOT}")
    print(f"  Limit      : {LIMIT if LIMIT else 'ALL'}")
    print(f"  Batch size : {BATCH_SIZE}  (profile={hardware.PROFILE})")
    print(f"  Threshold  : {ensemble.CONFIDENCE_THRESHOLD}")
    print(f"  Output     : {OUTPUT_DIR}")
    print("=" * 65)

    t_start = time.perf_counter()

    fake_acc, fake_ms = evaluate_folder(
        FAKE_DIR, "Fake",
        os.path.join(OUTPUT_DIR, "ensemble_fake_results.csv"),
    )
    real_acc, real_ms = evaluate_folder(
        REAL_DIR, "Real",
        os.path.join(OUTPUT_DIR, "ensemble_real_results.csv"),
    )

    overall    = round((fake_acc + real_acc) / 2, 2)
    overall_ms = round((fake_ms + real_ms) / 2, 1)
    total_secs = round(time.perf_counter() - t_start, 1)

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print(f"  {'Folder':<10} {'Accuracy':>10} {'Avg ms/img':>12}")
    print("  " + "-" * 38)
    print(f"  {'Fake':<10} {fake_acc:>9.2f}%  {fake_ms:>10.1f}ms")
    print(f"  {'Real':<10} {real_acc:>9.2f}%  {real_ms:>10.1f}ms")
    print(f"  {'Overall':<10} {overall:>9.2f}%  {overall_ms:>10.1f}ms")
    print(f"\n  Total wall time : {total_secs}s")
    print(f"  Results saved to: {OUTPUT_DIR}/")
    print("=" * 65)
