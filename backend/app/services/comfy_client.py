"""Asynchronous HTTP and WebSocket client for the ComfyUI API."""

import asyncio
import json
from typing import Any, Callable, Awaitable
import httpx
import websockets
from backend.app.core.config import settings


class ComfyClient:
    """Client for communicating with ComfyUI over local HTTP and WebSockets."""

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self.host = host or settings.COMFYUI_HOST
        self.port = port or settings.COMFYUI_PORT
        self.base_url = f"http://{self.host}:{self.port}"
        self.ws_url = f"ws://{self.host}:{self.port}/ws"

    async def submit_prompt(self, workflow: dict[str, Any], client_id: str) -> str:
        """Submit a prompt graph to ComfyUI /prompt endpoint and return prompt_id."""
        url = f"{self.base_url}/prompt"
        payload = {"prompt": workflow, "client_id": client_id}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"ComfyUI prompt submission failed ({resp.status_code}): {resp.text}")
            data = resp.json()
            if "prompt_id" not in data:
                raise KeyError(f"Missing prompt_id in ComfyUI response: {data}")
            return data["prompt_id"]

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        """Query /history/{prompt_id} to retrieve execution outputs and node results."""
        url = f"{self.base_url}/history/{prompt_id}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"ComfyUI history query failed ({resp.status_code}): {resp.text}")
            return resp.json()

    async def interrupt(self) -> bool:
        """Post to /interrupt to halt current GPU execution immediately."""
        url = f"{self.base_url}/interrupt"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json={})
            return resp.status_code == 200

    async def free_memory(self, unload_models: bool = True, free_memory: bool = True) -> bool:
        """Post to /free to unload models from VRAM and flush CUDA allocations."""
        url = f"{self.base_url}/free"
        payload = {"unload_models": unload_models, "free_memory": free_memory}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            return resp.status_code == 200

    async def get_object_info(self) -> dict[str, Any]:
        """Retrieve node class specifications from /object_info."""
        url = f"{self.base_url}/object_info"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch object_info ({resp.status_code}): {resp.text}")
            return resp.json()

    async def listen_events(
        self,
        client_id: str,
        on_event: Callable[[str, dict[str, Any]], Awaitable[None]],
        stop_event: Any = None,
    ) -> None:
        """Connect to ComfyUI WebSocket and stream execution events."""
        url = f"{self.ws_url}?clientId={client_id}"
        async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
            while True:
                if stop_event and stop_event.is_set():
                    break
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue

                if isinstance(raw_msg, str):
                    try:
                        parsed = json.loads(raw_msg)
                        msg_type = parsed.get("type", "unknown")
                        msg_data = parsed.get("data", {})
                        await on_event(msg_type, msg_data)
                    except json.JSONDecodeError:
                        pass


comfy_client = ComfyClient()
