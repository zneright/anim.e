"""ComfyUI process supervisor with strict ownership tracking and health checking."""

import asyncio
import os
from pathlib import Path
import subprocess
import time
from typing import Any
import httpx
from backend.app.core.config import settings


class ComfySupervisor:
    """Manages ComfyUI lifecycle, auto-boot, health checks, and process ownership."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.is_owned: bool = False
        self.base_url: str = f"http://{settings.COMFYUI_HOST}:{settings.COMFYUI_PORT}"

    def detect_installation(self) -> dict[str, Any]:
        """Verify presence of ComfyUI portable, python executable, and custom nodes."""
        exists = (
            settings.COMFYUI_PORTABLE_DIR.exists()
            and settings.COMFYUI_PYTHON_EXE.exists()
            and settings.COMFYUI_MAIN_PY.exists()
        )
        return {
            "installed": exists,
            "portable_dir": str(settings.COMFYUI_PORTABLE_DIR),
            "python_exe": str(settings.COMFYUI_PYTHON_EXE),
            "main_py": str(settings.COMFYUI_MAIN_PY),
        }

    async def check_health(self) -> dict[str, Any] | None:
        """Query ComfyUI health endpoint (/system_stats or /object_info)."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/system_stats")
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass
        return None

    async def is_alive(self) -> bool:
        """Returns True if ComfyUI HTTP server is currently responding."""
        return await self.check_health() is not None

    async def start(self) -> bool:
        """Start ComfyUI if not already running. Tracks process ownership strictly."""
        # 1. Check if an external instance is already running
        if await self.is_alive():
            self.is_owned = False
            return True

        detection = self.detect_installation()
        if not detection["installed"]:
            raise FileNotFoundError(f"ComfyUI installation not found: {detection}")

        # Set up environment with FFmpeg
        env = os.environ.copy()
        # Find ffmpeg binary if available
        ffmpeg_candidates = list(settings.COMFYUI_PORTABLE_DIR.rglob("ffmpeg*.exe"))
        if ffmpeg_candidates:
            ffmpeg_exe = str(ffmpeg_candidates[0])
            env["VHS_FORCE_FFMPEG_PATH"] = ffmpeg_exe
            env["PATH"] = str(Path(ffmpeg_exe).parent) + os.pathsep + env.get("PATH", "")

        cmd = [
            str(settings.COMFYUI_PYTHON_EXE),
            "-s",
            str(settings.COMFYUI_MAIN_PY),
            "--listen",
            settings.COMFYUI_HOST,
            "--port",
            str(settings.COMFYUI_PORT),
            "--lowvram",
            "--reserve-vram",
            "0.5",
            "--disable-auto-launch",
            "--preview-method",
            "none",
        ]

        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(settings.LOGS_DIR / "comfyui_backend.log", "a", encoding="utf-8")

        self.process = subprocess.Popen(
            cmd,
            cwd=str(settings.COMFYUI_PORTABLE_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.is_owned = True

        # Record PID
        settings.COMFYUI_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        settings.COMFYUI_PID_FILE.write_text(str(self.process.pid), encoding="utf-8")

        # Wait for server to boot
        start_time = time.time()
        while time.time() - start_time < settings.COMFYUI_BOOT_TIMEOUT_SEC:
            if self.process.poll() is not None:
                self.is_owned = False
                settings.COMFYUI_PID_FILE.unlink(missing_ok=True)
                raise RuntimeError(
                    f"ComfyUI process exited prematurely with code {self.process.returncode}. "
                    f"Check {settings.LOGS_DIR / 'comfyui_backend.log'}"
                )
            if await self.is_alive():
                return True
            await asyncio.sleep(1.5)

        # Timeout reached
        await self.stop()
        raise TimeoutError(f"Timed out waiting {settings.COMFYUI_BOOT_TIMEOUT_SEC}s for ComfyUI to start.")

    async def stop(self) -> bool:
        """Stop ComfyUI safely. NEVER terminates an externally managed instance."""
        if not self.is_owned:
            return False

        if self.process and self.process.poll() is None:
            pid = self.process.pid
            # Graceful Windows process tree termination
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    timeout=settings.COMFYUI_SHUTDOWN_TIMEOUT_SEC,
                )
            except Exception:
                self.process.terminate()

            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()

        self.process = None
        self.is_owned = False
        settings.COMFYUI_PID_FILE.unlink(missing_ok=True)
        return True


supervisor = ComfySupervisor()
