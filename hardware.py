"""
hardware.py
===========
Detects hardware at launch and configures PyTorch for maximum performance.

Tiers handled
-------------
  TIER 1 — CUDA GPU (discrete NVIDIA)
      → models on GPU, float16, CUDA streams
  TIER 2 — Intel iGPU via OpenVINO (Intel UHD / Iris / Arc)
      → OpenVINO GPU device if openvino installed, else falls to TIER 3
  TIER 3 — CPU only (current machine: i3-1215U)
      → AVX2/AVX_VNNI + oneDNN + MKL + pinned P-cores + optimal threads

Exports
-------
  DEVICE       : torch.device to use
  DTYPE        : torch.float16 | torch.bfloat16 | torch.float32
  PROFILE      : "cuda" | "openvino" | "cpu_avx2" | "cpu_basic"
  configure()  : call once at startup — sets all torch globals
  summary()    : prints a hardware summary table
"""

import os
import torch
import psutil


# ── Suppress torchao C-level .so noise ────────────────────────────────────────
_torchao_suppressed = False

def suppress_torchao_noise() -> None:
    """
    Suppress torchao's C-level .so load noise (printed directly to fd 2, not Python's
    sys.stderr). Call this ONCE before the first torchao import in any entry point.
    Subsequent calls are no-ops (torchao is already cached in sys.modules).
    """
    global _torchao_suppressed
    if _torchao_suppressed:
        return
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd   = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        import torchao  # noqa: F401 — pre-load to absorb .so noise
    except Exception:
        pass
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)
    _torchao_suppressed = True

# ── Detect CUDA ────────────────────────────────────────────────────────────────
def _detect_cuda() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name":      props.name,
        "vram_gb":   round(props.total_memory / 1e9, 1),
        "sm_count":  props.multi_processor_count,
        "bf16":      torch.cuda.is_bf16_supported(),
    }


# ── Detect Intel iGPU via OpenVINO ────────────────────────────────────────────
def _detect_openvino() -> dict:
    try:
        from openvino.runtime import Core  # type: ignore[import-untyped]
        core    = Core()
        devices = core.available_devices
        has_gpu = any("GPU" in d for d in devices)
        return {"available": has_gpu, "devices": devices}
    except ImportError:
        return {"available": False, "devices": [], "reason": "openvino not installed"}
    except Exception as e:
        return {"available": False, "devices": [], "reason": str(e)}


# ── Detect CPU capabilities ────────────────────────────────────────────────────
def _detect_cpu() -> dict:
    flags = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line
                    break
    except Exception:
        pass

    # Count physical P-cores vs E-cores (Intel hybrid)
    # P-cores have hyperthreading; E-cores don't
    logical  = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False)
    ht_cores = logical - physical   # number of cores with HT = P-cores count
    p_cores  = ht_cores             # each P-core contributes 2 logical
    e_cores  = physical - p_cores

    ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    free_gb = round(psutil.virtual_memory().available / 1e9, 1)

    return {
        "logical":    logical,
        "physical":   physical,
        "p_cores":    p_cores,
        "e_cores":    e_cores,
        "avx2":       "avx2"     in flags,
        "avx_vnni":   "avx_vnni" in flags,
        "mkl":        torch.backends.mkl.is_available(),
        "mkldnn":     torch.backends.mkldnn.is_available(),
        "ram_gb":     ram_gb,
        "free_gb":    free_gb,
    }


# ── Build hardware profile ─────────────────────────────────────────────────────
def _build_profile(cuda: dict, ov: dict, cpu: dict) -> tuple[str, torch.device, torch.dtype]:
    if cuda["available"]:
        dtype  = torch.bfloat16 if cuda["bf16"] else torch.float16
        device = torch.device("cuda:0")
        return "cuda", device, dtype

    if ov["available"]:
        # OpenVINO handles device internally — torch stays on CPU
        # but models can be exported/run via OV runtime on iGPU
        device = torch.device("cpu")
        dtype  = torch.float32
        return "openvino", device, dtype

    # CPU-only path
    device = torch.device("cpu")
    if cpu["avx2"] or cpu["avx_vnni"]:
        dtype = torch.float32   # AVX2+oneDNN is fastest at float32 on Intel
        return "cpu_avx2", device, dtype

    dtype = torch.float32
    return "cpu_basic", device, dtype


# ── Apply torch settings ───────────────────────────────────────────────────────
def _apply_torch_settings(profile: str, cpu: dict) -> dict:
    settings = {}

    if profile == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        torch.backends.cudnn.benchmark         = True
        settings["cuda_tf32"]    = True
        settings["cudnn_bench"]  = True

    elif profile in ("cpu_avx2", "cpu_basic", "openvino"):
        # Use ALL physical cores — E-cores are slow but not useless for ML batch ops.
        # Avoid logical (HT) count — hyperthreading hurts matmul throughput.
        num_threads = max(cpu["physical"], 1)

        # interop threads: for parallel ops (e.g. dataloader + model)
        # keep at 2 to avoid contention on low-core machines
        interop = min(2, num_threads)

        # set_num_threads is safe to call anytime
        torch.set_num_threads(num_threads)

        # set_num_interop_threads can ONLY be called before any parallel work starts.
        # If torch has already been used (e.g. imported elsewhere first), skip silently.
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            interop = torch.get_num_interop_threads()  # read actual current value

        # oneDNN/MKL-DNN: force enable for AVX2 path
        if cpu["mkldnn"]:
            torch.backends.mkldnn.enabled = True

        settings["num_threads"]       = num_threads
        settings["interop_threads"]   = interop
        settings["mkldnn"]            = cpu["mkldnn"]

    return settings


# ── Public API ─────────────────────────────────────────────────────────────────
_cuda = _detect_cuda()
_ov   = _detect_openvino()
_cpu  = _detect_cpu()

PROFILE, DEVICE, DTYPE = _build_profile(_cuda, _ov, _cpu)

_SETTINGS = {}   # filled by configure()
_configured = False


def configure() -> None:
    """Call once at app startup. Applies all torch performance settings. Idempotent."""
    global _SETTINGS, _configured
    if _configured:
        return
    _configured = True
    _SETTINGS = _apply_torch_settings(PROFILE, _cpu)


def summary() -> None:
    """Print a clean hardware summary to console."""
    bar = "=" * 60
    print(bar)
    print("  Hardware Profile")
    print(bar)
    print(f"  Profile     : {PROFILE.upper()}")
    print(f"  Device      : {DEVICE}")
    print(f"  Dtype       : {DTYPE}")

    if _cuda["available"]:
        print(f"\n  GPU         : {_cuda['name']}")
        print(f"  VRAM        : {_cuda['vram_gb']} GB")
        print(f"  SM count    : {_cuda['sm_count']}")
        print(f"  BF16        : {_cuda['bf16']}")

    if _ov["available"]:
        print(f"\n  OpenVINO devices: {_ov['devices']}")

    print(f"\n  CPU cores   : {_cpu['physical']} physical / {_cpu['logical']} logical")
    print(f"  P-cores     : {_cpu['p_cores']}  |  E-cores: {_cpu['e_cores']}")
    print(f"  AVX2        : {_cpu['avx2']}")
    print(f"  AVX_VNNI    : {_cpu['avx_vnni']}")
    print(f"  oneDNN      : {_cpu['mkldnn']}")
    print(f"  MKL         : {_cpu['mkl']}")
    print(f"  RAM total   : {_cpu['ram_gb']} GB  |  free: {_cpu['free_gb']} GB")

    if _SETTINGS:
        print(f"\n  Applied settings:")
        for k, v in _SETTINGS.items():
            print(f"    {k}: {v}")
    print(bar)


def move_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Move model to the right device and cast dtype.
    For CPU profiles, model stays float32 (quantization handles speed).
    For CUDA, cast to float16/bfloat16.
    """
    model = model.to(DEVICE)
    if PROFILE == "cuda":
        model = model.to(DTYPE)
    return model
