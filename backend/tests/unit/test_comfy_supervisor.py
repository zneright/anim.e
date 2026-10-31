"""Unit tests for ComfyUI Process Supervisor."""

from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from backend.app.services.comfy_supervisor import ComfySupervisor


@pytest.mark.asyncio
async def test_detect_installation():
    supervisor = ComfySupervisor()
    info = supervisor.detect_installation()
    assert "installed" in info
    assert info["installed"] is True
    assert Path(info["python_exe"]).exists()
    assert Path(info["main_py"]).exists()


@pytest.mark.asyncio
async def test_supervisor_does_not_own_external_instance():
    """If ComfyUI is already running on port 8188, supervisor must NOT claim ownership."""
    supervisor = ComfySupervisor()
    with patch.object(supervisor, "is_alive", new=AsyncMock(return_value=True)):
        started = await supervisor.start()
        assert started is True
        assert supervisor.is_owned is False

        # Attempting stop on an unowned instance must do nothing and return False
        stopped = await supervisor.stop()
        assert stopped is False
        assert supervisor.process is None


@pytest.mark.asyncio
async def test_check_health_parses_response():
    import httpx
    supervisor = ComfySupervisor()
    mock_stats = {"system": {"os": "nt"}, "devices": [{"name": "RTX 3050"}]}

    mock_resp = httpx.Response(
        200,
        json=mock_stats,
        request=httpx.Request("GET", "http://127.0.0.1:8188/system_stats"),
    )

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        health = await supervisor.check_health()
        assert health == mock_stats
        assert await supervisor.is_alive() is True
