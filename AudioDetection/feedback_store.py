"""
feedback_store.py
=================
Manages user corrections for audio deepfake detection.
Stores audio files and maintains balanced replay buffer.
"""

import os
import csv
import shutil
import hashlib
from collections import deque
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
CSV_PATH = os.path.join(FEEDBACK_DIR, "corrections.csv")

MAX_PER_CLASS = 50  # Keep 50 audio files per class in replay buffer

_correction_count = 0
_buffer = {"Fake": deque(maxlen=MAX_PER_CLASS), "Real": deque(maxlen=MAX_PER_CLASS)}
_seen_hashes = set()


def _compute_hash(audio_path: str) -> str:
    """Compute MD5 hash of audio file."""
    md5 = hashlib.md5()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def record_correction(audio_path: str, predicted_label: str) -> str:
    """
    Record a wrong prediction. Flips label and saves audio file.
    Returns the correct label.
    """
    global _correction_count
    
    correct_label = "Real" if predicted_label == "Fake" else "Fake"
    _correction_count += 1
    
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    
    # Compute hash to avoid duplicates
    audio_hash = _compute_hash(audio_path)
    
    # Save audio file with hash-based naming
    ext = os.path.splitext(audio_path)[1]
    filename = f"{audio_hash}_{correct_label}{ext}"
    dest_path = os.path.join(FEEDBACK_DIR, filename)
    
    # Only save if not already in buffer
    if audio_hash not in _seen_hashes:
        shutil.copy2(audio_path, dest_path)
        _buffer[correct_label].append(dest_path)
        _seen_hashes.add(audio_hash)
    
    # Log to CSV
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if _correction_count == 1:
            writer.writerow(["timestamp", "correction_num", "predicted", "correct", "audio_hash", "audio_path"])
        writer.writerow([
            datetime.now().isoformat(),
            _correction_count,
            predicted_label,
            correct_label,
            audio_hash,
            dest_path
        ])
    
    return correct_label


def get_buffer() -> dict:
    """Return current replay buffer."""
    return _buffer


def buffer_size() -> int:
    """Total items in buffer."""
    return len(_buffer["Fake"]) + len(_buffer["Real"])


def correction_count() -> int:
    """Total corrections recorded this session."""
    return _correction_count


def total_logged() -> int:
    """Total corrections logged to CSV (persistent)."""
    if not os.path.exists(CSV_PATH):
        return 0
    with open(CSV_PATH, "r") as f:
        return max(0, sum(1 for _ in f) - 1)  # -1 for header


def load_buffer_from_disk():
    """Load existing corrections from disk into buffer on startup."""
    global _correction_count
    
    if not os.path.exists(FEEDBACK_DIR):
        return
    
    for filename in sorted(os.listdir(FEEDBACK_DIR)):
        if not ("_Fake" in filename or "_Real" in filename):
            continue
        
        # Parse label from filename
        if "_Fake" in filename:
            label = "Fake"
        elif "_Real" in filename:
            label = "Real"
        else:
            continue
        
        # Extract hash from filename
        audio_hash = filename.split("_")[0]
        
        path = os.path.join(FEEDBACK_DIR, filename)
        if audio_hash not in _seen_hashes and len(_buffer[label]) < MAX_PER_CLASS:
            _buffer[label].append(path)
            _seen_hashes.add(audio_hash)
    
    # Set correction count from CSV
    _correction_count = total_logged()
    
    if _correction_count > 0:
        print(f"[audio/feedback_store] Loaded {buffer_size()} corrections from disk", flush=True)
