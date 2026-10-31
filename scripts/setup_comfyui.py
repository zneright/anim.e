r"""Step 1 of Phase 1 Setup: Download (with HTTP Range Resume), Extract, Cleanup, and Verify ComfyUI Portable.

Enforces:
- 15.0 GiB minimum free space safeguard at every step (including extraction buffer).
- Sequential download with HTTP Content-Length verification and Range resume.
- Immediate deletion of the .7z archive after verified extraction.
"""
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

ROOT_DIR = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT_DIR / "engine"
TOOLS_DIR = ENGINE_DIR / "tools"
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = ROOT_DIR / "docs"

MIN_FREE_GIB = 15.0

SEVEN_ZIP_URL = "https://www.7-zip.org/a/7zr.exe"
COMFY_VERSION = "v0.43.0"
COMFY_URL = f"https://github.com/Comfy-Org/ComfyUI/releases/download/{COMFY_VERSION}/ComfyUI_windows_portable_nvidia_cu126.7z"


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


def download_file(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "anim.e-setup/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        expected_bytes = int(resp.headers.get("Content-Length", 0))

    if dest.exists() and dest.stat().st_size == expected_bytes and expected_bytes > 0:
        print(f"[SKIP] {label} already downloaded and size-verified ({expected_bytes / (1024**2):.1f} MiB)", flush=True)
        return

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


def main() -> None:
    for d in [
        ENGINE_DIR,
        TOOLS_DIR,
        DATA_DIR / "outputs" / "poc",
        DATA_DIR / "thumbnails",
        DATA_DIR / "uploads",
        DATA_DIR / "temp",
        DATA_DIR / "logs",
        DOCS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    seven_zip_exe = TOOLS_DIR / "7zr.exe"
    download_file(SEVEN_ZIP_URL, seven_zip_exe, "7zr.exe")

    comfy_dir = ENGINE_DIR / "ComfyUI_windows_portable"
    python_exe = comfy_dir / "python_embeded" / "python.exe"

    if not python_exe.exists():
        archive_path = ENGINE_DIR / f"ComfyUI_windows_portable_nvidia_cu126_{COMFY_VERSION}.7z"
        # Check disk space for remaining download + extraction (~5.3 GiB)
        assert_disk_space("Pre-check for ComfyUI download + extraction", needed_gib=6.8)
        download_file(COMFY_URL, archive_path, f"ComfyUI Portable {COMFY_VERSION} (CUDA 12.6)")

        assert_disk_space("Extract ComfyUI Portable", needed_gib=5.5)
        print("[EXTRACTING] ComfyUI_windows_portable ... (this takes 1-3 minutes)", flush=True)
        subprocess.run(
            [str(seven_zip_exe), "x", str(archive_path), f"-o{ENGINE_DIR}", "-y"],
            check=True,
        )

        if not python_exe.exists():
            raise RuntimeError(f"Extraction failed: {python_exe} not found!")

        print(f"[CLEANUP] Deleting archive {archive_path.name} to reclaim 1.80 GiB ...", flush=True)
        archive_path.unlink()
        assert_disk_space("Post-extraction archive cleanup", needed_gib=0.0)
    else:
        print("[SKIP] ComfyUI_windows_portable is already extracted.", flush=True)

    print("\n=== VERIFYING EMBEDDED PYTHON & PYTORCH CUDA ===", flush=True)
    verify_code = (
        "import sys, torch; "
        "print('Python:', sys.version); "
        "print('PyTorch:', torch.__version__); "
        "print('CUDA Available:', torch.cuda.is_available()); "
        "print('CUDA Version:', torch.version.cuda); "
        "print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); "
        "print('Device Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"
    )
    subprocess.run([str(python_exe), "-c", verify_code], check=True)


if __name__ == "__main__":
    main()
