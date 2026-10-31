"""Environment, dependency import, and path configuration tests for Subphase 2.1."""

import shutil
import pytest
from backend.app.core.config import settings


def test_dependency_imports():
    """Verify all pinned dependencies import cleanly without errors."""
    import fastapi
    import uvicorn
    import aiosqlite
    import pydantic
    import pydantic_settings
    import httpx
    import websockets
    import PIL

    assert fastapi.__version__ == "0.115.8"
    assert uvicorn.__version__ == "0.34.0"
    assert aiosqlite.__version__ == "0.21.0"
    assert pydantic.__version__ == "2.10.6"
    assert pydantic_settings.__version__ == "2.7.1"
    assert httpx.__version__ == "0.28.1"
    assert websockets.__version__ == "14.2"
    assert PIL.__version__ == "11.1.0"


def test_no_pytorch_in_backend_environment():
    """Verify PyTorch is NOT installed in the backend venv (preserving disk space)."""
    with pytest.raises(ImportError):
        import torch  # type: ignore  # noqa: F401


def test_directory_paths_and_safeguards():
    """Verify core paths exist and safety settings meet specifications."""
    assert settings.ROOT_DIR.exists()
    assert settings.DATA_DIR.exists()
    assert settings.ENGINE_DIR.exists()
    assert settings.COMFYUI_PORTABLE_DIR.exists()
    assert settings.COMFYUI_PYTHON_EXE.exists()
    assert settings.COMFYUI_MAIN_PY.exists()

    # Safety thresholds
    assert settings.MIN_FREE_DISK_GIB == 15.00
    assert settings.PAGEFILE_EXPANSION_BUFFER_GIB == 9.50
    assert settings.MAX_ACTIVE_GPU_JOBS == 1
    assert settings.TOTAL_VRAM_MIB == 4096.0


def test_phase1_artifacts_intact():
    """Verify all Phase 1 files, benchmarks, and model weights remain intact."""
    manifest_file = settings.ENGINE_DIR / "manifest.json"
    benchmark_report = settings.DOCS_DIR / "hardware_benchmark.md"
    poc_out_dir = settings.DATA_DIR / "outputs" / "poc"

    assert manifest_file.exists(), "engine/manifest.json must be preserved"
    assert benchmark_report.exists(), "docs/hardware_benchmark.md must be preserved"
    assert (poc_out_dir / "rung1_video_384x384_8f.mp4").exists(), "Rung 1 MP4 must be preserved"
    assert (poc_out_dir / "rung2_video_384x384_16f.mp4").exists(), "Rung 2 MP4 must be preserved"
    assert (poc_out_dir / "rung3b_video_512x512_8f.mp4").exists(), "Rung 3b MP4 must be preserved"

    # Model checkpoints
    ckpt = (
        settings.COMFYUI_DIR
        / "models"
        / "checkpoints"
        / "Counterfeit-V3.0_fix_fp16.safetensors"
    )
    vae = (
        settings.COMFYUI_DIR
        / "models"
        / "vae"
        / "vae-ft-mse-840000-ema-pruned.safetensors"
    )
    mm = (
        settings.COMFYUI_DIR
        / "custom_nodes"
        / "ComfyUI-AnimateDiff-Evolved"
        / "models"
        / "mm_sd_v15_v2.ckpt"
    )

    assert ckpt.exists() and ckpt.stat().st_size == 2132651162
    assert vae.exists() and vae.stat().st_size == 334641190
    assert mm.exists() and mm.stat().st_size == 1817888431


def test_free_disk_space_meets_safeguard():
    """Verify host drive currently has >= 15.00 GiB free space."""
    _, _, free = shutil.disk_usage("C:\\")
    free_gib = free / (1024**3)
    assert free_gib >= settings.MIN_FREE_DISK_GIB, (
        f"Free space {free_gib:.2f} GiB violated {settings.MIN_FREE_DISK_GIB} GiB floor"
    )
