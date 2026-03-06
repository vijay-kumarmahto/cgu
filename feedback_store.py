"""
feedback_store.py
=================
Manages all user feedback data:
  - Replay buffer : last N corrected images kept in memory for training
  - Image store   : saves corrected images to disk (feedback/images/)
  - CSV log       : appends every correction to feedback/feedback_log.csv

How correct label is determined (NO user input needed):
  Model predicted "Fake" + user clicked Wrong  →  correct label = "Real"
  Model predicted "Real" + user clicked Wrong  →  correct label = "Fake"

Public API
----------
  record_correction(image, predicted_label)
      → saves image, logs to CSV, adds to buffer
      → returns correct_label (the flipped label)

  get_replay_batch(n)
      → returns up to n random (image, label) pairs from buffer

  correction_count()
      → total corrections since startup

  total_logged()
      → total corrections ever logged (from CSV, survives restarts)
"""

import os
import csv
import random
import hashlib
import datetime
from collections import deque
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
IMAGE_DIR    = os.path.join(FEEDBACK_DIR, "images")
LOG_PATH     = os.path.join(FEEDBACK_DIR, "feedback_log.csv")

os.makedirs(IMAGE_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
BUFFER_SIZE = 100   # max images kept in replay buffer (FIFO)

# ── CSV fields ─────────────────────────────────────────────────────────────────
_CSV_FIELDS = [
    "timestamp",
    "image_hash",
    "predicted_label",
    "correct_label",
    "image_path",
]

# ── Internal state ─────────────────────────────────────────────────────────────
# Each entry: {"image": PIL.Image, "label": "Fake" | "Real"}
_buffer: deque = deque(maxlen=BUFFER_SIZE)

# Session correction count (resets on restart)
_session_count: int = 0


# ── CSV helpers ────────────────────────────────────────────────────────────────
def _ensure_csv() -> None:
    """Create CSV with header if it doesn't exist yet."""
    if not os.path.isfile(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_CSV_FIELDS).writeheader()


def _append_csv(row: dict) -> None:
    _ensure_csv()
    with open(LOG_PATH, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=_CSV_FIELDS).writerow(row)


def total_logged() -> int:
    """Count total rows in the CSV log (survives restarts)."""
    _ensure_csv()
    with open(LOG_PATH, "r", newline="") as fh:
        # subtract 1 for header row
        return max(0, sum(1 for _ in fh) - 1)


# ── Image hash ─────────────────────────────────────────────────────────────────
def _image_hash(image: Image.Image) -> str:
    """
    Fast MD5 hash of raw pixel bytes.
    Used as unique filename — same image uploaded twice gets the same hash
    (deduplication: won't save duplicate files or log duplicate corrections).
    """
    return hashlib.md5(image.tobytes()).hexdigest()[:16]


# ── Public API ─────────────────────────────────────────────────────────────────
def record_correction(image: Image.Image, predicted_label: str) -> str:
    """
    Record a user correction.

    Parameters
    ----------
    image           : PIL image that was predicted wrongly
    predicted_label : what the model said — "Fake" or "Real"

    Returns
    -------
    correct_label : the flipped label — "Real" if predicted "Fake", vice versa
    """
    global _session_count

    # ── Determine correct label (simple flip — no user input needed) ───────────
    correct_label = "Real" if predicted_label == "Fake" else "Fake"

    # ── Save image to disk ─────────────────────────────────────────────────────
    image    = image.convert("RGB")
    img_hash = _image_hash(image)
    fname    = f"{img_hash}_{correct_label}.jpg"
    img_path = os.path.join(IMAGE_DIR, fname)

    if not os.path.isfile(img_path):   # skip if exact same image already saved
        image.save(img_path, "JPEG", quality=95)

    # ── Append to CSV log ──────────────────────────────────────────────────────
    _append_csv({
        "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
        "image_hash":      img_hash,
        "predicted_label": predicted_label,
        "correct_label":   correct_label,
        "image_path":      img_path,
    })

    # ── Add to replay buffer ───────────────────────────────────────────────────
    # Store a copy so the original PIL object can be GC'd
    _buffer.append({
        "image": image.copy(),
        "label": correct_label,
    })

    _session_count += 1

    print(
        f"[feedback_store] Correction #{_session_count} recorded — "
        f"predicted={predicted_label}, correct={correct_label}, "
        f"buffer_size={len(_buffer)}",
        flush=True,
    )

    return correct_label


def get_replay_batch(n: int) -> list[dict]:
    """
    Return up to n random samples from the replay buffer.

    Each item: {"image": PIL.Image, "label": "Fake" | "Real"}

    If buffer has fewer than n items, returns all of them.
    Used by trainer.py to build mixed training batches.
    """
    if not _buffer:
        return []
    k = min(n, len(_buffer))
    return random.sample(list(_buffer), k)


def correction_count() -> int:
    """Number of corrections recorded this session (resets on restart)."""
    return _session_count


def load_buffer_from_disk() -> None:
    """
    Pre-populate the replay buffer from saved images on disk at startup.
    Reads the CSV log to get correct labels, loads images from IMAGE_DIR.
    Loads up to BUFFER_SIZE most recent entries.

    Call this once at startup (app.py) so the buffer is warm even after restart.
    """
    global _buffer
    _ensure_csv()

    rows = []
    try:
        with open(LOG_PATH, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    except Exception:
        return

    # Take the most recent BUFFER_SIZE entries
    rows = rows[-BUFFER_SIZE:]

    loaded = 0
    for row in rows:
        img_path = row.get("image_path", "")
        label    = row.get("correct_label", "")
        if not img_path or not label:
            continue
        if not os.path.isfile(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            _buffer.append({"image": img, "label": label})
            loaded += 1
        except Exception:
            continue

    if loaded:
        print(f"[feedback_store] Loaded {loaded} images into replay buffer from disk.", flush=True)
