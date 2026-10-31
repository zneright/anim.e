r"""Phase 1 Empirical Hardware Benchmark Runner for anim.e Studio (Hardened v3).

Safety & Accuracy Guarantees:
1. No fabricated GPU telemetry: if nvidia-smi fails, raises TelemetryError and aborts benchmarks.
2. Exact artifact resolution: resolves output files strictly from ComfyUI's /history/{prompt_id}
   metadata for that specific prompt_id; preserves all companion files in data/outputs/poc/ and never deletes files.
3. Strict cryptographic SHA-256 verification: requires every model to match a trusted upstream HuggingFace LFS
   SHA-256 OID. Missing upstream metadata is marked UNVERIFIED and halts execution immediately.
4. Pre-submission workflow schema & node-link validation against ComfyUI's live /object_info endpoint.
5. Strict pass/fail criteria: both Rung 0 and Rung 1 must pass; any failure halts subsequent rungs.
6. Preserves pre-installation disk space (37.95 GiB) separately from benchmark-start and post-benchmark disk space.
7. Distinguishes automated OpenCV/FFmpeg container validation from human visual inspection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT_DIR = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT_DIR / "engine"
COMFY_PORTABLE_DIR = ENGINE_DIR / "ComfyUI_windows_portable"
COMFY_DIR = COMFY_PORTABLE_DIR / "ComfyUI"
PYTHON_EXE = COMFY_PORTABLE_DIR / "python_embeded" / "python.exe"
POC_OUT_DIR = ROOT_DIR / "data" / "outputs" / "poc"
LOG_DIR = ROOT_DIR / "data" / "logs"

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"
MIN_FREE_GIB = 15.0
PRE_INSTALL_FREE_DISK_GIB = 37.95

CHECKPOINT_NAME = "Counterfeit-V3.0_fix_fp16.safetensors"
CHECKPOINT_REPO = "gsdf/Counterfeit-V3.0"

VAE_NAME = "vae-ft-mse-840000-ema-pruned.safetensors"
VAE_REPO = "stabilityai/sd-vae-ft-mse-original"

MOTION_MODULE_NAME = "mm_sd_v15_v2.ckpt"
MOTION_MODULE_REPO = "guoyww/animatediff"

POSITIVE_PROMPT = (
    "masterpiece, best quality, anime style, cinematic lighting, 1girl, silver hair, "
    "violet eyes, standing on a rooftop at twilight, gentle breeze blowing hair, "
    "glowing city lights in background, Makoto Shinkai style, detailed eyes, smooth motion"
)
NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
    "signature, watermark, username, blurry, static, deformed"
)


class TelemetryError(RuntimeError):
    """Raised when genuine GPU telemetry cannot be collected via nvidia-smi."""


def query_gpu() -> dict:
    """Queries genuine GPU telemetry via nvidia-smi. Never fabricates fallback metrics."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.PIPE,
        ).strip()
    except Exception as exc:
        raise TelemetryError(f"nvidia-smi execution failed: {exc}") from exc

    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        raise TelemetryError("nvidia-smi returned empty output")

    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 7:
        raise TelemetryError(f"Unexpected nvidia-smi output format: {lines[0]}")

    try:
        vram_used = float(parts[2])
        vram_total = float(parts[3])
        gpu_util = float(parts[4])
        temp_c = float(parts[5])
    except ValueError as exc:
        raise TelemetryError(f"Non-numeric GPU telemetry in '{lines[0]}': {exc}") from exc

    power_w: float | None = None
    if parts[6] not in ("[N/A]", "N/A", ""):
        try:
            power_w = float(parts[6])
        except ValueError:
            power_w = None

    return {
        "gpu_name": parts[0],
        "driver_version": parts[1],
        "vram_used_mib": vram_used,
        "vram_total_mib": vram_total,
        "gpu_util_pct": gpu_util,
        "temp_c": temp_c,
        "power_w": power_w,
    }


class TelemetryMonitor:
    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval_sec = interval_sec
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        first = query_gpu()
        self.samples = [first]
        self.errors = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(query_gpu())
            except TelemetryError as exc:
                self.errors.append(str(exc))
            time.sleep(self.interval_sec)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        try:
            self.samples.append(query_gpu())
        except TelemetryError as exc:
            self.errors.append(str(exc))

        if not self.samples:
            return {
                "telemetry_valid": False,
                "errors": self.errors or ["No telemetry samples collected"],
            }

        power_samples = [s["power_w"] for s in self.samples if s["power_w"] is not None]
        return {
            "telemetry_valid": len(self.errors) == 0 and len(self.samples) > 0,
            "gpu_name": self.samples[0]["gpu_name"],
            "driver_version": self.samples[0]["driver_version"],
            "vram_total_mib": self.samples[0]["vram_total_mib"],
            "baseline_vram_mib": self.samples[0]["vram_used_mib"],
            "peak_vram_mib": max(s["vram_used_mib"] for s in self.samples),
            "avg_gpu_util_pct": round(
                sum(s["gpu_util_pct"] for s in self.samples) / len(self.samples), 1
            ),
            "peak_gpu_util_pct": max(s["gpu_util_pct"] for s in self.samples),
            "max_temp_c": max(s["temp_c"] for s in self.samples),
            "max_power_w": max(power_samples) if power_samples else None,
            "sample_count": len(self.samples),
            "errors": self.errors,
        }


def compute_full_sha256(file_path: Path) -> str:
    """Computes full cryptographic SHA-256 hex digest of a file."""
    print(f"  [SHA-256] Computing full SHA-256 for {file_path.name} ...", flush=True)
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def fetch_published_hf_lfs_metadata(repo_id: str, filename: str) -> dict:
    """Queries HuggingFace API for published LFS SHA-256 oid and byte size."""
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    req = urllib.request.Request(url, headers={"User-Agent": "anim.e-verifier/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            items = json.loads(resp.read().decode("utf-8"))
        for item in items:
            if item.get("path") == filename:
                lfs = item.get("lfs") or {}
                oid = lfs.get("oid")
                size = lfs.get("size", item.get("size"))
                if oid and len(oid) == 64:
                    return {
                        "published_sha256": oid.lower(),
                        "published_size_bytes": size,
                        "source": url,
                    }
    except Exception as exc:
        return {"published_sha256": None, "published_size_bytes": None, "error": str(exc)}
    return {"published_sha256": None, "published_size_bytes": None, "error": "No LFS SHA-256 OID found"}


def verify_safetensors_header(file_path: Path) -> dict:
    with open(file_path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"File too small to be a valid safetensors file: {file_path}")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        if header_len <= 0 or header_len > 100 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header length {header_len} in {file_path}")
        header_json = json.loads(f.read(header_len).decode("utf-8"))
    tensor_count = len([k for k in header_json.keys() if k != "__metadata__"])
    return {"valid_safetensors": True, "tensor_count": tensor_count}


def verify_model_integrity(file_path: Path, repo_id: str, is_safetensors: bool) -> dict:
    """Strictly requires published upstream SHA-256 and exact match. Stops if unverified or mismatched."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required model file missing: {file_path}")
    actual_bytes = file_path.stat().st_size
    local_sha256 = compute_full_sha256(file_path).lower()
    hf_meta = fetch_published_hf_lfs_metadata(repo_id, file_path.name)
    pub_sha256 = hf_meta.get("published_sha256")
    pub_bytes = hf_meta.get("published_size_bytes")

    if not pub_sha256 or pub_bytes is None:
        raise RuntimeError(
            f"UNVERIFIED MODEL: Upstream SHA-256 or size metadata missing for {file_path.name} "
            f"(repo {repo_id}, error={hf_meta.get('error')}). Refusing to proceed without cryptographic authentication."
        )

    if actual_bytes != pub_bytes:
        raise RuntimeError(
            f"SIZE MISMATCH for {file_path.name}: actual {actual_bytes} bytes != published {pub_bytes} bytes"
        )

    if local_sha256 != pub_sha256.lower():
        raise RuntimeError(
            f"SHA-256 MISMATCH for {file_path.name}: local {local_sha256} != published {pub_sha256}"
        )

    result = {
        "file": file_path.name,
        "size_bytes": actual_bytes,
        "size_mib": round(actual_bytes / (1024**2), 2),
        "hash_type": "full_sha256",
        "local_sha256": local_sha256,
        "published_sha256": pub_sha256,
        "checksum_status": "VERIFIED",
        "checksum_verified_against_upstream": True,
    }
    if is_safetensors:
        result.update(verify_safetensors_header(file_path))
    print(
        f"  [VERIFIED] {file_path.name}: {result['size_mib']} MiB | SHA-256={local_sha256[:16]}... matches upstream LFS OID",
        flush=True,
    )
    return result


def run_12_point_verification(ffmpeg_exe: str, benchmark_start_free_gib: float) -> dict:
    print("=== RUNNING 12-POINT INSTALLATION & CRYPTOGRAPHIC VERIFICATION ===", flush=True)
    gpu_now = query_gpu()
    if benchmark_start_free_gib < MIN_FREE_GIB:
        raise RuntimeError(
            f"Disk space {benchmark_start_free_gib} GiB is below {MIN_FREE_GIB} GiB floor!"
        )

    ckpt_path = COMFY_DIR / "models" / "checkpoints" / CHECKPOINT_NAME
    vae_path = COMFY_DIR / "models" / "vae" / VAE_NAME
    mm_path = (
        COMFY_DIR
        / "custom_nodes"
        / "ComfyUI-AnimateDiff-Evolved"
        / "models"
        / MOTION_MODULE_NAME
    )

    ckpt_info = verify_model_integrity(ckpt_path, CHECKPOINT_REPO, is_safetensors=True)
    vae_info = verify_model_integrity(vae_path, VAE_REPO, is_safetensors=True)
    mm_info = verify_model_integrity(mm_path, MOTION_MODULE_REPO, is_safetensors=False)

    py_check = subprocess.check_output(
        [
            str(PYTHON_EXE),
            "-c",
            (
                "import json, sys, torch, cv2, imageio_ffmpeg, safetensors; "
                "props = torch.cuda.get_device_properties(0); "
                "print(json.dumps({"
                "'python': sys.version.split()[0], "
                "'torch': torch.__version__, "
                "'cuda_available': bool(torch.cuda.is_available()), "
                "'cuda_version': torch.version.cuda, "
                "'device_name': torch.cuda.get_device_name(0), "
                "'compute_capability': f'{props.major}.{props.minor}', "
                "'total_vram_mib': round(props.total_memory / (1024**2)), "
                "'opencv': cv2.__version__, "
                "'imageio_ffmpeg': imageio_ffmpeg.__version__"
                "}))"
            ),
        ],
        text=True,
    ).strip()
    py_data = json.loads(py_check)
    if not py_data.get("cuda_available"):
        raise RuntimeError("PyTorch reports torch.cuda.is_available() == False!")

    enc_out = subprocess.check_output(
        [ffmpeg_exe, "-encoders"], text=True, stderr=subprocess.STDOUT
    )
    has_libx264 = "libx264" in enc_out
    if not has_libx264:
        raise RuntimeError(f"FFmpeg at {ffmpeg_exe} does not support libx264 encoding!")

    verification = {
        "comfy_dir_exists": COMFY_DIR.exists(),
        "initial_gpu_state": gpu_now,
        "pre_install_free_disk_gib": PRE_INSTALL_FREE_DISK_GIB,
        "benchmark_start_free_disk_gib": benchmark_start_free_gib,
        "runtime": py_data,
        "checkpoint": ckpt_info,
        "vae": vae_info,
        "motion_module": mm_info,
        "ffmpeg": {
            "binary": ffmpeg_exe,
            "libx264_supported": has_libx264,
        },
    }
    print("[12-POINT VERIFICATION COMPLETED SUCCESSFULLY]", flush=True)
    return verification


def http_json(url: str, data: dict | None = None, timeout: float = 30.0) -> dict:
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
    else:
        req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def free_comfy_memory() -> None:
    try:
        http_json(f"{COMFY_BASE}/free", {"unload_models": True, "free_memory": True}, timeout=15.0)
        time.sleep(2.0)
    except Exception as exc:
        print(f"  [WARN] ComfyUI /free call returned error: {exc}", flush=True)


def validate_workflow_against_schema(workflow: dict, object_info: dict) -> None:
    """Validates every node class_type, required input keys, enum values, and node links against /object_info."""
    for node_id, node in workflow.items():
        class_type = node.get("class_type")
        if class_type not in object_info:
            raise RuntimeError(
                f"Workflow node #{node_id} uses unregistered class_type '{class_type}'"
            )
        node_schema = object_info[class_type]
        schema_inputs = node_schema.get("input", {})
        required_inputs = schema_inputs.get("required", {})
        optional_inputs = schema_inputs.get("optional", {})
        provided_inputs = node.get("inputs", {})

        for req_key in required_inputs:
            if req_key not in provided_inputs:
                raise RuntimeError(
                    f"Workflow node #{node_id} ({class_type}) is missing required input '{req_key}'"
                )

        all_schema_inputs = {**required_inputs, **optional_inputs}
        for key, val in provided_inputs.items():
            spec = all_schema_inputs.get(key)
            # Check if val is a connection link [src_node_id, slot_idx]
            if isinstance(val, list):
                if len(val) != 2 or str(val[0]) not in workflow or not isinstance(val[1], int):
                    raise RuntimeError(
                        f"Workflow node #{node_id} ({class_type}) input '{key}' has invalid link {val}"
                    )
                src_node = workflow[str(val[0])]
                src_class = src_node.get("class_type")
                src_outputs = object_info.get(src_class, {}).get("output", [])
                if val[1] < 0 or val[1] >= len(src_outputs):
                    raise RuntimeError(
                        f"Workflow node #{node_id} ({class_type}) input '{key}' links to out-of-range "
                        f"output slot {val[1]} on node #{val[0]} ({src_class}, outputs={src_outputs})"
                    )
            elif (
                isinstance(spec, list)
                and len(spec) > 0
                and isinstance(spec[0], list)
            ):
                allowed = spec[0]
                if val not in allowed:
                    raise RuntimeError(
                        f"Workflow node #{node_id} ({class_type}) input '{key}' value '{val}' "
                        f"is not in allowed ComfyUI options: {allowed}"
                    )


def build_sanity_workflow(width: int, height: int, steps: int, prefix: str) -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT_NAME},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": POSITIVE_PROMPT, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE_PROMPT, "clip": ["1", 1]},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 424242,
                "steps": steps,
                "cfg": 7.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "7": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["2", 0],
                "tile_size": 384,
                "overlap": 64,
                "temporal_size": 8,
                "temporal_overlap": 4,
            },
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"images": ["7", 0], "filename_prefix": prefix},
        },
    }


def build_video_workflow(
    width: int, height: int, frames: int, steps: int, fps: int, prefix: str
) -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT_NAME},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": POSITIVE_PROMPT, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE_PROMPT, "clip": ["1", 1]},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": frames},
        },
        "6": {
            "class_type": "ADE_AnimateDiffLoaderGen1",
            "inputs": {
                "model": ["1", 0],
                "model_name": MOTION_MODULE_NAME,
                "beta_schedule": "autoselect",
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 108080,
                "steps": steps,
                "cfg": 7.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["6", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["2", 0],
                "tile_size": 384,
                "overlap": 64,
                "temporal_size": 8,
                "temporal_overlap": 4,
            },
        },
        "9": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def verify_image_file(image_path: Path, expected_w: int, expected_h: int) -> dict:
    check_code = (
        "import cv2, json, numpy as np, sys; "
        "img = cv2.imread(sys.argv[1]); "
        "h, w, c = img.shape if img is not None else (0, 0, 0); "
        "has_nan = bool(np.isnan(img).any()) if img is not None else True; "
        "mean_val = float(np.mean(img)) if img is not None else 0.0; "
        "std_val = float(np.std(img)) if img is not None else 0.0; "
        "exp_w, exp_h = int(sys.argv[2]), int(sys.argv[3]); "
        "valid = bool(img is not None and not has_nan and w == exp_w and h == exp_h and mean_val > 5.0 and std_val > 5.0); "
        "print(json.dumps({'valid': valid, 'width': w, 'height': h, 'mean_pixel': round(mean_val, 2), "
        "'std_pixel': round(std_val, 2), 'visual_inspection_status': 'PENDING_USER_REVIEW'}))"
    )
    out = subprocess.check_output(
        [str(PYTHON_EXE), "-c", check_code, str(image_path), str(expected_w), str(expected_h)],
        text=True,
    ).strip()
    return json.loads(out)


def verify_video_file(
    video_path: Path, expected_w: int, expected_h: int, expected_frames: int, expected_fps: int
) -> dict:
    check_code = (
        "import cv2, json, numpy as np, sys; "
        "cap = cv2.VideoCapture(sys.argv[1]); "
        "w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); "
        "h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); "
        "fps = float(cap.get(cv2.CAP_PROP_FPS)); "
        "frames = []\n"
        "while True:\n"
        "    ok, frame = cap.read()\n"
        "    if not ok or frame is None: break\n"
        "    frames.append(frame)\n"
        "cap.release(); "
        "n = len(frames); "
        "exp_w, exp_h, exp_n, exp_fps = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]); "
        "min_frame_std = min((float(np.std(f)) for f in frames), default=0.0); "
        "mean_val = float(np.mean(frames[0])) if n > 0 else 0.0; "
        "motion_diff = float(np.mean(np.abs(frames[-1].astype(float) - frames[0].astype(float)))) if n > 1 else 0.0; "
        "duration_sec = round(n / fps, 2) if fps > 0 else 0.0; "
        "valid = bool(n == exp_n and w == exp_w and h == exp_h and abs(fps - exp_fps) < 0.5 and mean_val > 5.0 and min_frame_std > 5.0 and motion_diff > 0.2); "
        "print(json.dumps({'valid': valid, 'frames': n, 'width': w, 'height': h, 'fps': round(fps, 2), "
        "'duration_sec': duration_sec, 'mean_pixel': round(mean_val, 2), 'min_frame_std': round(min_frame_std, 2), "
        "'motion_diff_score': round(motion_diff, 2), 'visual_inspection_status': 'PENDING_USER_REVIEW'}))"
    )
    out = subprocess.check_output(
        [
            str(PYTHON_EXE),
            "-c",
            check_code,
            str(video_path),
            str(expected_w),
            str(expected_h),
            str(expected_frames),
            str(expected_fps),
        ],
        text=True,
    ).strip()
    return json.loads(out)


def resolve_prompt_output_files(history_entry: dict, comfy_out_dir: Path) -> list[Path]:
    """Extracts the exact output file paths recorded by ComfyUI in /history/{prompt_id}."""
    resolved_files: list[Path] = []
    base_dir = comfy_out_dir.resolve()
    outputs_by_node = history_entry.get("outputs", {})
    for _node_id, node_out in outputs_by_node.items():
        if not isinstance(node_out, dict):
            continue
        for key in ("gifs", "videos", "images"):
            items = node_out.get(key, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                if item.get("type", "output") != "output":
                    continue
                subfolder = item.get("subfolder", "")
                candidate = (base_dir / subfolder / item["filename"]).resolve()
                if candidate.is_relative_to(base_dir) and candidate.exists():
                    if candidate not in resolved_files:
                        resolved_files.append(candidate)
    return resolved_files


def persist_summary(
    summary_path: Path,
    benchmark_start_free_gib: float,
    verification_info: dict | None,
    results: list[dict],
    post_shutdown_vram_mib: float | None,
) -> None:
    POC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, _, current_free = shutil.disk_usage("C:\\")
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": (
            verification_info["initial_gpu_state"]["gpu_name"]
            if verification_info and "initial_gpu_state" in verification_info
            else "Unavailable"
        ),
        "pre_install_free_disk_gib": PRE_INSTALL_FREE_DISK_GIB,
        "benchmark_start_free_disk_gib": benchmark_start_free_gib,
        "final_free_disk_gib": round(current_free / (1024**3), 2),
        "post_shutdown_vram_mib": post_shutdown_vram_mib,
        "verification": verification_info,
        "results": results,
    }
    tmp_path = summary_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(summary_path)


def execute_workflow(
    rung_id: str,
    name: str,
    meta: dict,
    workflow: dict,
    object_info: dict,
    expected_ext: str,
    dest_filename: str,
    rung_timeout_sec: float,
) -> dict:
    print(f"\n>>> RUNNING BENCHMARK: {name} <<<", flush=True)
    validate_workflow_against_schema(workflow, object_info)

    baseline_gpu = query_gpu()
    print(f"  Baseline VRAM: {baseline_gpu['vram_used_mib']:.0f} MiB", flush=True)

    monitor = TelemetryMonitor(interval_sec=0.5)
    monitor.start()
    start_t = time.time()
    client_id = str(uuid.uuid4())

    try:
        res = http_json(f"{COMFY_BASE}/prompt", {"prompt": workflow, "client_id": client_id})
        prompt_id = res["prompt_id"]
        print(f"  Submitted prompt_id: {prompt_id}", flush=True)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        stats = monitor.stop()
        print(f"  [FAIL] Submission HTTP {exc.code}: {err_body}", flush=True)
        return {
            "rung_id": rung_id,
            "name": name,
            "parameters": meta,
            "status": "failed_submission",
            "error": f"HTTP {exc.code}: {err_body}",
            "duration_sec": round(time.time() - start_t, 2),
            "telemetry": stats,
        }
    except Exception as exc:
        stats = monitor.stop()
        print(f"  [FAIL] Submission error: {exc}", flush=True)
        return {
            "rung_id": rung_id,
            "name": name,
            "parameters": meta,
            "status": "failed_submission",
            "error": str(exc),
            "duration_sec": round(time.time() - start_t, 2),
            "telemetry": stats,
        }

    comfy_out_dir = COMFY_DIR / "output"
    history_entry: dict | None = None
    while True:
        elapsed = time.time() - start_t
        if elapsed > rung_timeout_sec:
            try:
                http_json(f"{COMFY_BASE}/interrupt", {})
            except Exception:
                pass
            stats = monitor.stop()
            print(f"  [FAIL] Timed out after {elapsed:.1f}s (limit {rung_timeout_sec}s)", flush=True)
            return {
                "rung_id": rung_id,
                "name": name,
                "prompt_id": prompt_id,
                "parameters": meta,
                "status": "failed_timeout",
                "error": f"Exceeded {rung_timeout_sec}s timeout",
                "duration_sec": round(elapsed, 2),
                "telemetry": stats,
            }

        hist = http_json(f"{COMFY_BASE}/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status_obj = entry.get("status", {})
            if status_obj.get("status_str") == "error":
                stats = monitor.stop()
                err_msgs = json.dumps(status_obj.get("messages", []))
                print(f"  [FAIL] ComfyUI execution error: {err_msgs}", flush=True)
                return {
                    "rung_id": rung_id,
                    "name": name,
                    "prompt_id": prompt_id,
                    "parameters": meta,
                    "status": "failed_execution",
                    "error": err_msgs,
                    "duration_sec": round(elapsed, 2),
                    "telemetry": stats,
                }
            if status_obj.get("completed", False) or entry.get("outputs"):
                history_entry = entry
                break
        time.sleep(1.0)

    duration_sec = round(time.time() - start_t, 2)
    stats = monitor.stop()

    prompt_files = resolve_prompt_output_files(history_entry or {}, comfy_out_dir)
    matching_target = [p for p in prompt_files if p.suffix.lower() == expected_ext.lower()]

    if not matching_target:
        print(
            f"  [FAIL] No {expected_ext} output recorded in ComfyUI history for prompt_id {prompt_id}. "
            f"Files recorded: {[str(p) for p in prompt_files]}",
            flush=True,
        )
        return {
            "rung_id": rung_id,
            "name": name,
            "prompt_id": prompt_id,
            "parameters": meta,
            "status": "failed_missing_output",
            "error": f"No {expected_ext} output recorded in /history/{prompt_id}",
            "duration_sec": duration_sec,
            "telemetry": stats,
        }

    primary_file = matching_target[0]
    POC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_path = POC_OUT_DIR / dest_filename
    shutil.move(str(primary_file), str(final_path))

    # Preserve all companion files (e.g. VHS preview PNGs / workflow images) in data/outputs/poc/
    companion_paths: list[str] = []
    for companion in prompt_files:
        if companion != primary_file and companion.exists():
            comp_dest = POC_OUT_DIR / f"{final_path.stem}_companion_{companion.name}"
            shutil.move(str(companion), str(comp_dest))
            companion_paths.append(str(comp_dest.relative_to(ROOT_DIR)))

    file_size_kb = round(final_path.stat().st_size / 1024.0, 1)
    if expected_ext.lower() == ".mp4":
        validation = verify_video_file(
            final_path,
            expected_w=meta["width"],
            expected_h=meta["height"],
            expected_frames=meta["frames"],
            expected_fps=meta["fps"],
        )
    else:
        validation = verify_image_file(
            final_path,
            expected_w=meta["width"],
            expected_h=meta["height"],
        )
    validation["file_size_kb"] = file_size_kb

    if not stats.get("telemetry_valid", False):
        status = "failed_telemetry"
        log_tag = "FAIL (TELEMETRY INVALID)"
    elif not validation.get("valid", False):
        status = "failed_validation"
        log_tag = "FAIL (AUTOMATED OUTPUT VALIDATION FAILED)"
    else:
        status = "passed"
        log_tag = "PASS (AUTOMATED VALIDATION PASSED — PENDING USER VISUAL INSPECTION)"

    print(
        f"  [{log_tag}] {name} finished in {duration_sec:.1f}s | "
        f"Peak VRAM: {stats.get('peak_vram_mib', 0):.0f} MiB | "
        f"Output: {final_path.name} ({file_size_kb} KB) | Validation: {validation}",
        flush=True,
    )
    return {
        "rung_id": rung_id,
        "name": name,
        "prompt_id": prompt_id,
        "parameters": meta,
        "status": status,
        "duration_sec": duration_sec,
        "output_path": str(final_path.relative_to(ROOT_DIR)),
        "companion_outputs": companion_paths,
        "file_size_kb": file_size_kb,
        "validation": validation,
        "telemetry": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Empirical GPU Benchmark Runner")
    parser.add_argument(
        "--boot-timeout-sec",
        type=float,
        default=float(os.environ.get("ANIME_BOOT_TIMEOUT_SEC", "180.0")),
    )
    parser.add_argument(
        "--rung-timeout-sec",
        type=float,
        default=float(os.environ.get("ANIME_RUNG_TIMEOUT_SEC", "600.0")),
    )
    parser.add_argument(
        "--max-vram-for-next-rung-mib",
        type=float,
        default=float(os.environ.get("ANIME_MAX_VRAM_NEXT_RUNG_MIB", "3980.0")),
    )
    args = parser.parse_args()

    _, _, free_now = shutil.disk_usage("C:\\")
    benchmark_start_free_gib = round(free_now / (1024**3), 2)
    if benchmark_start_free_gib < MIN_FREE_GIB:
        raise RuntimeError(
            f"Disk space {benchmark_start_free_gib:.2f} GiB is below {MIN_FREE_GIB} GiB floor!"
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = POC_OUT_DIR / "benchmark_results.json"
    comfy_log_path = LOG_DIR / "comfyui_phase1_poc.log"

    ladder_specs = [
        {
            "rung_id": "rung0",
            "name": "Rung 0: Basic SD 1.5 Sanity Test (384x384, 1 frame)",
            "kind": "image",
            "ext": ".png",
            "dest": "rung0_sanity_384x384_1f.png",
            "meta": {
                "width": 384,
                "height": 384,
                "resolution": "384x384",
                "frames": 1,
                "fps": 1,
                "steps": 15,
                "cfg": 7.0,
                "sampler": "euler_ancestral",
                "scheduler": "normal",
                "checkpoint": CHECKPOINT_NAME,
                "motion_module": "None (Static Sanity Check)",
            },
        },
        {
            "rung_id": "rung1",
            "name": "Rung 1: First AnimateDiff Video (384x384, 8 frames @ 8 FPS)",
            "kind": "video",
            "ext": ".mp4",
            "dest": "rung1_video_384x384_8f.mp4",
            "meta": {
                "width": 384,
                "height": 384,
                "resolution": "384x384",
                "frames": 8,
                "fps": 8,
                "steps": 15,
                "cfg": 7.0,
                "sampler": "euler_ancestral",
                "scheduler": "normal",
                "checkpoint": CHECKPOINT_NAME,
                "motion_module": MOTION_MODULE_NAME,
            },
        },
        {
            "rung_id": "rung2",
            "name": "Rung 2: Conservative 16-Frame Video (384x384, 16 frames @ 8 FPS)",
            "kind": "video",
            "ext": ".mp4",
            "dest": "rung2_video_384x384_16f.mp4",
            "meta": {
                "width": 384,
                "height": 384,
                "resolution": "384x384",
                "frames": 16,
                "fps": 8,
                "steps": 16,
                "cfg": 7.0,
                "sampler": "euler_ancestral",
                "scheduler": "normal",
                "checkpoint": CHECKPOINT_NAME,
                "motion_module": MOTION_MODULE_NAME,
            },
        },
        {
            "rung_id": "rung3a",
            "name": "Rung 3a: Portrait Video (384x512, 8 frames @ 8 FPS)",
            "kind": "video",
            "ext": ".mp4",
            "dest": "rung3a_video_384x512_8f.mp4",
            "meta": {
                "width": 384,
                "height": 512,
                "resolution": "384x512",
                "frames": 8,
                "fps": 8,
                "steps": 16,
                "cfg": 7.0,
                "sampler": "euler_ancestral",
                "scheduler": "normal",
                "checkpoint": CHECKPOINT_NAME,
                "motion_module": MOTION_MODULE_NAME,
            },
        },
        {
            "rung_id": "rung3b",
            "name": "Rung 3b: Square Standard Video (512x512, 8 frames @ 8 FPS)",
            "kind": "video",
            "ext": ".mp4",
            "dest": "rung3b_video_512x512_8f.mp4",
            "meta": {
                "width": 512,
                "height": 512,
                "resolution": "512x512",
                "frames": 8,
                "fps": 8,
                "steps": 16,
                "cfg": 7.0,
                "sampler": "euler_ancestral",
                "scheduler": "normal",
                "checkpoint": CHECKPOINT_NAME,
                "motion_module": MOTION_MODULE_NAME,
            },
        },
    ]

    results: list[dict] = [
        {
            "rung_id": spec["rung_id"],
            "name": spec["name"],
            "parameters": spec["meta"],
            "status": "not_run",
        }
        for spec in ladder_specs
    ]
    verification_info: dict | None = None
    post_shutdown_vram: float | None = None

    ffmpeg_exe = subprocess.check_output(
        [str(PYTHON_EXE), "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
        text=True,
    ).strip()

    verification_info = run_12_point_verification(ffmpeg_exe, benchmark_start_free_gib)
    time.sleep(3.0)
    persist_summary(
        summary_path, benchmark_start_free_gib, verification_info, results, post_shutdown_vram
    )

    env = os.environ.copy()
    env["VHS_FORCE_FFMPEG_PATH"] = ffmpeg_exe
    env["PATH"] = str(Path(ffmpeg_exe).parent) + os.pathsep + env.get("PATH", "")

    cmd = [
        str(PYTHON_EXE),
        "-s",
        str(COMFY_DIR / "main.py"),
        "--listen",
        COMFY_HOST,
        "--port",
        str(COMFY_PORT),
        "--lowvram",
        "--reserve-vram",
        "0.5",
        "--disable-auto-launch",
        "--preview-method",
        "none",
    ]

    print(f"[STARTING COMFYUI] {' '.join(cmd)}", flush=True)
    log_f = open(comfy_log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(COMFY_PORTABLE_DIR),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )

    try:
        boot_start = time.time()
        object_info: dict = {}
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI exited early with code {proc.returncode}. Check {comfy_log_path}"
                )
            try:
                object_info = http_json(f"{COMFY_BASE}/object_info", timeout=5.0)
                if "ADE_AnimateDiffLoaderGen1" in object_info and "VHS_VideoCombine" in object_info:
                    print(
                        f"[COMFYUI READY] Booted in {time.time() - boot_start:.1f}s. "
                        "Verified ADE_AnimateDiffLoaderGen1 and VHS_VideoCombine schemas.",
                        flush=True,
                    )
                    break
                missing = [
                    n
                    for n in ("ADE_AnimateDiffLoaderGen1", "VHS_VideoCombine")
                    if n not in object_info
                ]
                raise RuntimeError(f"Missing required custom nodes in ComfyUI: {missing}")
            except urllib.error.URLError:
                if time.time() - boot_start > args.boot_timeout_sec:
                    raise RuntimeError(
                        f"Timed out waiting {args.boot_timeout_sec}s for ComfyUI server to boot."
                    )
                time.sleep(2.0)

        for idx, spec in enumerate(ladder_specs):
            _, _, disk_free_now = shutil.disk_usage("C:\\")
            disk_free_gib = disk_free_now / (1024**3)
            print(
                f"[DISK GUARD - Before {spec['rung_id']}] Current Free: {disk_free_gib:.2f} GiB (Floor: {MIN_FREE_GIB:.2f} GiB)",
                flush=True,
            )
            if disk_free_gib < MIN_FREE_GIB:
                for rem_idx in range(idx, len(ladder_specs)):
                    results[rem_idx]["status"] = "skipped"
                    results[rem_idx]["error"] = (
                        f"Aborted due to 15.0 GiB minimum free disk guard (measured {disk_free_gib:.2f} GiB during active pagefile reservation)"
                    )
                persist_summary(
                    summary_path,
                    benchmark_start_free_gib,
                    verification_info,
                    results,
                    post_shutdown_vram,
                )
                break

            meta = spec["meta"]
            prefix = f"anim_e_{spec['rung_id']}_{uuid.uuid4().hex[:6]}"
            if spec["kind"] == "image":
                wf = build_sanity_workflow(meta["width"], meta["height"], meta["steps"], prefix)
            else:
                wf = build_video_workflow(
                    meta["width"],
                    meta["height"],
                    meta["frames"],
                    meta["steps"],
                    meta["fps"],
                    prefix,
                )

            rung_res = execute_workflow(
                rung_id=spec["rung_id"],
                name=spec["name"],
                meta=meta,
                workflow=wf,
                object_info=object_info,
                expected_ext=spec["ext"],
                dest_filename=spec["dest"],
                rung_timeout_sec=args.rung_timeout_sec,
            )
            results[idx] = rung_res
            free_comfy_memory()
            persist_summary(
                summary_path,
                benchmark_start_free_gib,
                verification_info,
                results,
                post_shutdown_vram,
            )

            if rung_res["status"] != "passed":
                reason = (
                    f"Skipped because {spec['rung_id']} did not pass "
                    f"(status={rung_res['status']}, error={rung_res.get('error', 'validation/telemetry failed')})"
                )
                for rem_idx in range(idx + 1, len(ladder_specs)):
                    results[rem_idx]["status"] = "skipped"
                    results[rem_idx]["error"] = reason
                persist_summary(
                    summary_path,
                    benchmark_start_free_gib,
                    verification_info,
                    results,
                    post_shutdown_vram,
                )
                break

            peak_vram = rung_res.get("telemetry", {}).get("peak_vram_mib", 0.0)
            if idx >= 1 and peak_vram >= args.max_vram_for_next_rung_mib:
                reason = (
                    f"Stopped progression after {spec['rung_id']} because peak VRAM "
                    f"({peak_vram:.0f} MiB) reached safety threshold ({args.max_vram_for_next_rung_mib:.0f} MiB)"
                )
                for rem_idx in range(idx + 1, len(ladder_specs)):
                    results[rem_idx]["status"] = "skipped"
                    results[rem_idx]["error"] = reason
                persist_summary(
                    summary_path,
                    benchmark_start_free_gib,
                    verification_info,
                    results,
                    post_shutdown_vram,
                )
                break

    finally:
        print("\n[SHUTTING DOWN COMFYUI SERVER] ...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_f.close()
        time.sleep(2.0)
        try:
            post_gpu = query_gpu()
            post_shutdown_vram = post_gpu["vram_used_mib"]
            print(f"[POST-SHUTDOWN GPU] VRAM Used: {post_shutdown_vram:.0f} MiB", flush=True)
        except TelemetryError as exc:
            print(f"[WARN] Post-shutdown GPU query failed: {exc}", flush=True)
        persist_summary(
            summary_path,
            benchmark_start_free_gib,
            verification_info,
            results,
            post_shutdown_vram,
        )
        print(f"[BENCHMARK SUMMARY PERSISTED] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
