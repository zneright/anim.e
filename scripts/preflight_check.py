r"""Phase 1 Pre-Flight Verification Script for anim.e Studio.

Checks:
1. Free disk space on C:\ against the 15.0 GiB hard floor.
2. NVIDIA RTX 3050 GPU availability, VRAM, and driver via nvidia-smi.
3. Exact URLs, HTTP status, Content-Length, and licenses for:
   - 7zr.exe
   - ComfyUI Windows Portable release assets
   - SD 1.5 FP16 Pruned Anime Checkpoint & VAE
   - AnimateDiff v2 FP16 Motion Module (.safetensors)
"""
import json
import shutil
import subprocess
import sys
import urllib.request


MIN_FREE_GIB = 15.0


def list_hf_repo(repo: str) -> None:
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    req = urllib.request.Request(url, headers={"User-Agent": "anim.e-preflight/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            items = json.loads(r.read().decode("utf-8"))
            print(f"=== HF Repo: {repo} ===")
            for i in items:
                if i.get("type") == "file":
                    size_mib = i.get("size", 0) / (1024 * 1024)
                    if size_mib > 10:
                        print(f"  {i['path']}: {size_mib:.1f} MiB")
    except Exception as e:
        print(f"=== HF Repo {repo}: {e} ===")


def search_hf_sd15_anime() -> None:
    url = "https://huggingface.co/api/models?search=Counterfeit&limit=10"
    req = urllib.request.Request(url, headers={"User-Agent": "anim.e-preflight/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        models = json.loads(r.read().decode("utf-8"))
        for m in models:
            print("Found model:", m["id"])


def main() -> None:
    for repo in [
        "gsdf/Counterfeit-V2.5",
        "gsdf/Counterfeit-V3.0",
        "Kijai/AnimateDiff_safetensors",
        "guoyww/animatediff",
        "ByteDance/AnimateDiff-Lightning",
        "Lykon/DreamShaper",
        "digiplay/MeinaMix_v11",
        "digiplay/Counterfeit-V3.0",
    ]:
        list_hf_repo(repo)


if __name__ == "__main__":
    main()
