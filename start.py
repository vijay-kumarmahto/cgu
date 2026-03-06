"""
start.py
========
Single entry point for the Ensemble Deepfake Detector.

Does three things before launching:
  1. Check all required packages are importable.
     If anything missing → print exact pip install command → exit.
  2. Check both model directories have required files.
     If anything missing → tell user exactly what and where → exit.
  3. Launch app.py in a subprocess (so hardware.configure() runs
     inside the correct process context).

Usage
-----
  python start.py
  DEEPFAKE_THRESHOLD=0.85 python start.py   # custom threshold
"""

import os
import sys
import signal
import subprocess
import time

# ── Suppress torchao C-level stderr noise BEFORE any import of torchao ────────
# Import hardware first (no heavy deps) so we can use the shared utility.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardware
hardware.suppress_torchao_noise()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 0. Kill any existing app instances ────────────────────────────────────────
def kill_existing() -> None:
    """
    Kill any running/suspended instances of app.py.
    Uses SIGTERM first, then SIGKILL if still alive, then waits for port release.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "app.py"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        current_pid = str(os.getpid())

        killed = []
        for pid in pids:
            if pid and pid != current_pid:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    killed.append(pid)
                except ProcessLookupError:
                    pass

        if killed:
            time.sleep(1)   # give SIGTERM time to work

            # SIGKILL any that are still alive (handles Ctrl+Z suspended procs)
            for pid in killed:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass   # already dead — good

            print(f"  🔪  Killed old app.py processes: {', '.join(killed)}")
            time.sleep(2)   # wait for OS to release ports
        else:
            print("  ✅  No old processes found.")

    except Exception:
        pass   # non-critical


# ── 1. Dependency check ────────────────────────────────────────────────────────
REQUIRED_PACKAGES = {
    "torch":           "torch>=2.0.0",
    "torchvision":     "torchvision>=0.15.0",
    "torchao":         "torchao>=0.16.0",
    "transformers":    "transformers>=4.38.0",
    "PIL":             "Pillow>=10.0.0",
    "gradio":          "gradio>=4.0.0",
    "tqdm":            "tqdm>=4.65.0",
    "accelerate":      "accelerate>=0.20.0",
    "psutil":          "psutil>=5.9.0",
}

def check_dependencies() -> bool:
    missing = []
    for module, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print("=" * 60)
        print("  ❌  Missing packages detected")
        print("=" * 60)
        print("\n  Install them with:\n")
        print(f"  pip install {' '.join(missing)}\n")
        return False

    print("  ✅  All dependencies present.")
    return True


# ── 2. Model file check ────────────────────────────────────────────────────────
REQUIRED_MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
]

IMAGE_DIR = os.path.join(BASE_DIR, "ImagePredication")

MODEL_DIRS = {
    "ViT":    os.path.join(IMAGE_DIR, "models", "vit"),
    "SigLIP": os.path.join(IMAGE_DIR, "models", "siglip"),
}

def check_models() -> bool:
    all_ok = True

    for name, path in MODEL_DIRS.items():
        if not os.path.isdir(path):
            print(f"  ❌  {name} model directory missing: {path}")
            all_ok = False
            continue

        for fname in REQUIRED_MODEL_FILES:
            full = os.path.join(path, fname)
            if not os.path.isfile(full):
                print(f"  ❌  Missing {fname} in {path}")
                all_ok = False

    if not all_ok:
        print("\n  Model files must be placed at:")
        for name, path in MODEL_DIRS.items():
            print(f"    {name}: {path}/")
            for f in REQUIRED_MODEL_FILES:
                print(f"      └── {f}")
        return False

    print("  ✅  All model files present.")
    return True


# ── 3. Launch ──────────────────────────────────────────────────────────────────
def launch() -> None:
    app_path = os.path.join(BASE_DIR, "app.py")
    env = os.environ.copy()
    env["TORCHAO_SUPPRESS_ERRORS"] = "1"
    env["PYTHONWARNINGS"] = "ignore"

    print("\n  🚀  Launching app...\n")
    try:
        result = subprocess.run([sys.executable, "-W", "ignore", app_path], env=env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        # Ctrl+C is forwarded to the subprocess automatically.
        # Catch it here to suppress the ugly traceback in the parent.
        print("\n  👋  Stopped.")
        sys.exit(0)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Ensemble Deepfake Detector — Startup Check")
    print("=" * 60)

    kill_existing()        # ← kill old processes first

    deps_ok   = check_dependencies()
    models_ok = check_models()

    # ── Check new feedback/trainer files are present ───────────────────────────
    new_files_ok = True
    for fname in ["feedback_store.py", "trainer.py"]:
        fpath = os.path.join(IMAGE_DIR, fname)
        if not os.path.isfile(fpath):
            print(f"  ❌  Missing required file: {fname}")
            new_files_ok = False
    if new_files_ok:
        print("  ✅  feedback_store.py and trainer.py present.")

    if not deps_ok or not models_ok or not new_files_ok:
        print("\n  ❌  Fix the issues above, then run start.py again.")
        sys.exit(1)

    print("=" * 60)
    launch()
