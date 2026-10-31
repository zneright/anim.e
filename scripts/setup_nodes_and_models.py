r"""Step 2 of Phase 1 Setup: Install Custom Nodes, FFmpeg wrapper, and Sequentially Download Models.

Enforces:
- 15.0 GiB minimum free disk space floor at every step.
- Exact git commit recording in engine/manifest.json.
- Sequential HTTP download (with Range resume) of pruned FP16 SD 1.5 Anime Checkpoint, VAE, and AnimateDiff v2 Motion Module.
"""
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

ROOT_DIR = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT_DIR / "engine"
COMFY_PORTABLE_DIR = ENGINE_DIR / "ComfyUI_windows_portable"
COMFY_DIR = COMFY_PORTABLE_DIR / "ComfyUI"
PYTHON_EXE = COMFY_PORTABLE_DIR / "python_embeded" / "python.exe"
CUSTOM_NODES_DIR = COMFY_DIR / "custom_nodes"

MIN_FREE_GIB = 15.0

REPOS = [
    (
        "ComfyUI-AnimateDiff-Evolved",
        "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git",
        "Apache-2.0",
    ),
    (
        "ComfyUI-VideoHelperSuite",
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "GPL-3.0",
    ),
]

MODELS = [
    {
        "name": "Counterfeit-V3.0_fix_fp16.safetensors",
        "type": "checkpoint (SD 1.5 FP16 Pruned)",
        "license": "CreativeML OpenRAIL-M",
        "url": "https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fix_fp16.safetensors",
        "dest": COMFY_DIR / "models" / "checkpoints" / "Counterfeit-V3.0_fix_fp16.safetensors",
    },
    {
        "name": "vae-ft-mse-840000-ema-pruned.safetensors",
        "type": "vae (SD 1.5 FP16 Pruned)",
        "license": "CreativeML OpenRAIL-M / MIT",
        "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
        "dest": COMFY_DIR / "models" / "vae" / "vae-ft-mse-840000-ema-pruned.safetensors",
    },
    {
        "name": "mm_sd_v15_v2.ckpt",
        "type": "motion_module (AnimateDiff v2 SD 1.5)",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt",
        "dest": CUSTOM_NODES_DIR / "ComfyUI-AnimateDiff-Evolved" / "models" / "mm_sd_v15_v2.ckpt",
    },
]


def assert_disk_space(stage: str, needed_gib: float = 0.0) -> float:
    _, _, free = shutil.disk_usage("C:\\")
    free_gib = free / (1024**3)
    projected_free = free_gib - needed_gib
    print(
        f"[DISK GUARD - {stage}] Current Free: {free_gib:.2f} GiB | "
        f"Projected Free after op ({needed_gib:.2f} GiB): {projected_free:.2f} GiB (Floor: {MIN_FREE_GIB:.2f} GiB)",
        flush=True,
    )
    if projected_free < MIN_FREE_GIB:
        raise RuntimeError(
            f"ABORT: Disk guard triggered at '{stage}'. Projected free space {projected_free:.2f} GiB "
            f"would fall below {MIN_FREE_GIB:.2f} GiB floor."
        )
    return free_gib


def download_file(url: str, dest: Path, label: str) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "anim.e-setup/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        expected_bytes = int(resp.headers.get("Content-Length", 0))

    if dest.exists() and dest.stat().st_size == expected_bytes and expected_bytes > 0:
        print(
            f"[SKIP] {label} already downloaded and size-verified ({expected_bytes / (1024**2):.1f} MiB)",
            flush=True,
        )
        return expected_bytes

    temp_dest = dest.with_suffix(dest.suffix + ".part")
    existing_bytes = temp_dest.stat().st_size if temp_dest.exists() else 0
    if existing_bytes > expected_bytes and expected_bytes > 0:
        temp_dest.unlink()
        existing_bytes = 0

    remaining_gib = max(0, expected_bytes - existing_bytes) / (1024**3)
    assert_disk_space(f"Download {label}", needed_gib=remaining_gib)

    headers = {"User-Agent": "anim.e-setup/1.0"}
    mode = "wb"
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        mode = "ab"
        print(
            f"[RESUMING] {label} from {existing_bytes / (1024**2):.1f} MiB / {expected_bytes / (1024**2):.1f} MiB ...",
            flush=True,
        )
    else:
        print(f"[DOWNLOADING] {label} ({expected_bytes / (1024**2):.1f} MiB) from {url} ...", flush=True)

    req_get = urllib.request.Request(url, headers=headers)
    start_t = time.time()
    downloaded = existing_bytes
    session_bytes = 0
    last_print = 0.0

    with urllib.request.urlopen(req_get, timeout=60) as resp, open(temp_dest, mode) as out_f:
        while True:
            chunk = resp.read(1024 * 1024 * 4)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            session_bytes += len(chunk)
            now = time.time()
            if now - last_print >= 5.0 or downloaded == expected_bytes:
                pct = (downloaded / expected_bytes * 100.0) if expected_bytes else 0.0
                speed = (session_bytes / (1024**2)) / max(now - start_t, 0.1)
                print(
                    f"  -> {downloaded / (1024**2):.1f} / {expected_bytes / (1024**2):.1f} MiB "
                    f"({pct:.1f}%) @ {speed:.1f} MiB/s",
                    flush=True,
                )
                last_print = now

    if expected_bytes > 0 and downloaded != expected_bytes:
        raise RuntimeError(f"Download size mismatch for {label}: got {downloaded}, expected {expected_bytes}")

    temp_dest.replace(dest)
    print(f"[DONE] Saved {dest} ({downloaded / (1024**2):.1f} MiB)", flush=True)
    return downloaded


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    if not PYTHON_EXE.exists():
        raise RuntimeError(f"ComfyUI embedded Python not found at {PYTHON_EXE}")

    manifest: dict = {
        "comfyui_portable_version": "v0.43.0 (cu126)",
        "comfyui_commit": get_git_commit(COMFY_DIR),
        "custom_nodes": {},
        "python_packages": {},
        "models": [],
    }

    print("=== 1. INSTALLING CUSTOM NODES ===", flush=True)
    assert_disk_space("Clone custom nodes", needed_gib=0.25)
    for name, url, license_name in REPOS:
        target = CUSTOM_NODES_DIR / name
        if not target.exists():
            print(f"[GIT CLONE] {name} from {url} ...", flush=True)
            subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
        else:
            print(f"[SKIP] {name} already cloned.", flush=True)
        commit_sha = get_git_commit(target)
        manifest["custom_nodes"][name] = {
            "url": url,
            "commit": commit_sha,
            "license": license_name,
        }
        print(f"  -> {name} pinned at commit {commit_sha}", flush=True)

    print("\n=== 2. INSTALLING PINNED PYTHON DEPENDENCIES FOR VHS & ANIMATEDIFF ===", flush=True)
    subprocess.run(
        [
            str(PYTHON_EXE),
            "-m",
            "pip",
            "install",
            "--no-warn-script-location",
            "imageio-ffmpeg==0.6.0",
            "opencv-python-headless==4.11.0.86",
        ],
        check=True,
    )

    ffmpeg_path = subprocess.check_output(
        [str(PYTHON_EXE), "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
        text=True,
    ).strip()
    print(f"[FFMPEG VERIFIED] Binary at: {ffmpeg_path}", flush=True)
    manifest["python_packages"]["imageio-ffmpeg"] = "0.6.0"
    manifest["python_packages"]["opencv-python-headless"] = "4.11.0.86"
    manifest["ffmpeg_binary"] = ffmpeg_path

    print("\n=== 3. SEQUENTIAL MODEL DOWNLOADS (WITH 15 GiB DISK SAFEGUARD) ===", flush=True)
    for m in MODELS:
        size_bytes = download_file(m["url"], m["dest"], m["name"])
        manifest["models"].append(
            {
                "name": m["name"],
                "type": m["type"],
                "license": m["license"],
                "url": m["url"],
                "size_bytes": size_bytes,
                "size_mib": round(size_bytes / (1024**2), 2),
                "path": str(m["dest"].relative_to(ROOT_DIR)),
            }
        )

    py_info = json.loads(
        subprocess.check_output(
            [
                str(PYTHON_EXE),
                "-c",
                "import json, sys, torch; print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0)}))",
            ],
            text=True,
        ).strip()
    )
    manifest["runtime"] = py_info

    manifest_path = ENGINE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[MANIFEST SAVED] {manifest_path}", flush=True)
    assert_disk_space("Post-model download check", needed_gib=0.0)


if __name__ == "__main__":
    main()
