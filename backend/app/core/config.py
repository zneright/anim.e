"""Application settings and path configurations for anim.e Studio."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core settings for anim.e Studio local backend."""

    # Branding & App Metadata
    PROJECT_NAME: str = "anim.e Studio"
    PROJECT_TAGLINE: str = "Your imagination, animated."
    VERSION: str = "0.2.0"
    API_V1_STR: str = "/api/v1"

    # Network / Host Binding
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]

    # Filesystem Paths
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = ROOT_DIR / "data"
    DB_PATH: Path = DATA_DIR / "app.db"
    OUTPUTS_DIR: Path = DATA_DIR / "outputs"
    THUMBNAILS_DIR: Path = DATA_DIR / "thumbnails"
    TEMP_DIR: Path = DATA_DIR / "temp"
    LOGS_DIR: Path = DATA_DIR / "logs"
    DOCS_DIR: Path = ROOT_DIR / "docs"

    # Engine (ComfyUI) Paths
    ENGINE_DIR: Path = ROOT_DIR / "engine"
    COMFYUI_PORTABLE_DIR: Path = ENGINE_DIR / "ComfyUI_windows_portable"
    COMFYUI_DIR: Path = COMFYUI_PORTABLE_DIR / "ComfyUI"
    COMFYUI_PYTHON_EXE: Path = COMFYUI_PORTABLE_DIR / "python_embeded" / "python.exe"
    COMFYUI_MAIN_PY: Path = COMFYUI_DIR / "main.py"
    COMFYUI_PID_FILE: Path = DATA_DIR / ".comfyui.pid"

    # Engine Network & Launch Settings
    COMFYUI_HOST: str = "127.0.0.1"
    COMFYUI_PORT: int = 8188
    COMFYUI_BOOT_TIMEOUT_SEC: float = 120.0
    COMFYUI_SHUTDOWN_TIMEOUT_SEC: float = 15.0

    # Resource Protection & Safety Safeguards
    MIN_FREE_DISK_GIB: float = 15.00
    PAGEFILE_EXPANSION_BUFFER_GIB: float = 9.50
    OUTPUT_ESTIMATE_BUFFER_GIB: float = 0.50
    MAX_VRAM_SAFE_CEILING_MIB: float = 3600.0
    TOTAL_VRAM_MIB: float = 4096.0

    # Job Queue & Execution Limits
    MAX_ACTIVE_GPU_JOBS: int = 1
    MAX_QUEUE_SIZE: int = 50
    STEP_WATCHDOG_TIMEOUT_SEC: float = 90.0
    TOTAL_JOB_TIMEOUT_SEC: float = 600.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
