"""Unit tests for ComfyUI HTTP Client and Workflow Builder."""

from unittest.mock import AsyncMock, patch
import pytest
from backend.app.services.comfy_client import ComfyClient
from backend.app.workflows.builder import WorkflowBuilder, PRESET_CONFIGS


@pytest.mark.asyncio
async def test_submit_prompt_success():
    import httpx
    client = ComfyClient()
    mock_workflow = {"3": {"class_type": "CLIPTextEncode"}}

    mock_resp = httpx.Response(
        200,
        json={"prompt_id": "test-prompt-uuid-42"},
        request=httpx.Request("POST", "http://127.0.0.1:8188/prompt"),
    )

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        prompt_id = await client.submit_prompt(mock_workflow, client_id="client-1")
        assert prompt_id == "test-prompt-uuid-42"


@pytest.mark.asyncio
async def test_submit_prompt_failure():
    import httpx
    client = ComfyClient()
    mock_resp = httpx.Response(
        400,
        text="Node validation failed",
        request=httpx.Request("POST", "http://127.0.0.1:8188/prompt"),
    )

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        with pytest.raises(RuntimeError, match="ComfyUI prompt submission failed"):
            await client.submit_prompt({}, client_id="client-1")


@pytest.mark.asyncio
async def test_interrupt_and_free_memory():
    import httpx
    client = ComfyClient()
    mock_resp = httpx.Response(
        200,
        json={},
        request=httpx.Request("POST", "http://127.0.0.1:8188/interrupt"),
    )

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        assert await client.interrupt() is True
        assert await client.free_memory() is True


def test_workflow_builder_presets():
    """Verify workflow builder produces valid ComfyUI node graphs matching Phase 1 benchmarks."""
    for preset_name, config in PRESET_CONFIGS.items():
        graph = WorkflowBuilder.build_txt2vid_workflow(
            prompt="anime character rooftop",
            negative_prompt="blurry, lowres",
            width=config["width"],
            height=config["height"],
            frame_count=config["frame_count"],
            fps=config["fps"],
            steps=config["steps"],
            cfg_scale=config["cfg_scale"],
            seed=108080,
            sampler_name=config["sampler_name"],
            scheduler=config["scheduler"],
            output_prefix=f"test_{preset_name}",
        )

        assert "1" in graph  # CheckpointLoaderSimple
        assert "2" in graph  # VAELoader
        assert "3" in graph  # Positive prompt
        assert "4" in graph  # Negative prompt
        assert "5" in graph  # EmptyLatentImage
        assert "6" in graph  # ADE_AnimateDiffLoaderGen1
        assert "7" in graph  # KSampler
        assert "8" in graph  # VAEDecodeTiled
        assert "9" in graph  # VHS_VideoCombine

        assert graph["1"]["inputs"]["ckpt_name"] == "Counterfeit-V3.0_fix_fp16.safetensors"
        assert graph["6"]["inputs"]["model_name"] == "mm_sd_v15_v2.ckpt"
        assert graph["8"]["inputs"]["tile_size"] == 384
        assert graph["9"]["inputs"]["format"] == "video/h264-mp4"
        assert graph["5"]["inputs"]["batch_size"] == config["frame_count"]
