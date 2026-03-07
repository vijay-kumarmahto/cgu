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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 0. Check virtual environment ─────────────────────────────────────────────
def check_venv() -> bool:
    if sys.prefix == sys.base_prefix:
        print("  ❌  Virtual environment not active.")
        print("\n  Activate it with:\n")
        print("  source .venv/bin/activate   # Linux/macOS")
        print("  .venv\\Scripts\\activate      # Windows\n")
        return False
    print("  ✅  Virtual environment active.")
    return True


# ── 1. Kill any existing app instances ────────────────────────────────────────
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


# ── 2. Dependency check ────────────────────────────────────────────────────────
# Map import names to pip package names (when they differ)
IMPORT_TO_PIP = {
    "PIL": "Pillow",
}

def parse_requirements() -> list[str]:
    """Parse requirements.txt and return list of package specs."""
    req_path = os.path.join(BASE_DIR, "requirements.txt")
    if not os.path.isfile(req_path):
        return []
    
    packages = []
    with open(req_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                packages.append(line)
    return packages

def get_import_name(package_spec: str) -> str:
    """Extract import name from package spec (e.g., 'torch>=2.0.0' -> 'torch')."""
    # Remove version specifiers
    for sep in [">", "<", "=", "!", "~"]:
        if sep in package_spec:
            package_spec = package_spec.split(sep)[0]
    return package_spec.strip()

def check_dependencies() -> bool:
    packages = parse_requirements()
    if not packages:
        print("  ⚠️  requirements.txt not found or empty")
        return True  # Don't block if file missing
    
    missing = []
    for pkg_spec in packages:
        pip_package = get_import_name(pkg_spec)
        
        # Map pip package name to import name (Pillow -> PIL)
        import_name = "PIL" if pip_package == "Pillow" else pip_package
        
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_spec)
    
    if missing:
        print("=" * 60)
        print("  ❌  Missing packages detected")
        print("=" * 60)
        print("\n  Install them with:\n")
        print(f"  pip install {' '.join(missing)}\n")
        return False
    
    print("  ✅  All dependencies present.")
    return True


# ── 3. Model file check ────────────────────────────────────────────────────────
REQUIRED_MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
]

IMAGE_DIR = os.path.join(BASE_DIR, "ImageDetection")
VIDEO_DIR = os.path.join(BASE_DIR, "VideoDetection")
AUDIO_DIR = os.path.join(BASE_DIR, "AudioDetection")

MODEL_DIRS = {
    "ViT":    os.path.join(IMAGE_DIR, "models", "vit"),
    "SigLIP": os.path.join(IMAGE_DIR, "models", "siglip"),
    "Audio":  os.path.join(AUDIO_DIR, "models"),
}

VIDEO_WEIGHTS = {
    "GenConViT-ED":  os.path.join(VIDEO_DIR, "weights", "genconvit_ed_inference.pth"),
    "GenConViT-VAE": os.path.join(VIDEO_DIR, "weights", "genconvit_vae_inference.pth"),
}

def check_models() -> bool:
    all_ok = True

    # Check image models
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
    else:
        print("  ✅  All model files present (Image + Audio).")
    
    # Check video model weights (required)
    for name, path in VIDEO_WEIGHTS.items():
        if not os.path.isfile(path):
            print(f"  ❌  {name} weight missing: {path}")
            all_ok = False
    
    if all_ok:
        print("  ✅  All video model weights present.")
    
    return all_ok


# ── 4. Launch ──────────────────────────────────────────────────────────────────
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

    venv_ok = check_venv()
    if not venv_ok:
        print("\n  ❌  Activate virtual environment first.")
        sys.exit(1)

    kill_existing()

    deps_ok   = check_dependencies()
    models_ok = check_models()

    if not deps_ok or not models_ok:
        print("\n  ❌  Fix the issues above, then run start.py again.")
        sys.exit(1)

    print("=" * 60)
    launch()