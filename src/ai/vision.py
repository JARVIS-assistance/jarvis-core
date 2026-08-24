"""Local vision-model image description.

Standalone from LocalLLMAIClient's streaming/text-chat machinery on purpose:
this is a one-shot multimodal call (image in, text out), not a chat stream,
so it doesn't need session reuse, keep-alive tuning, or fallback endpoints.
Reuses the docker-localhost rewrite helper since Ollama is typically reached
the same way for both.
"""

from __future__ import annotations

import os

import aiohttp
from pydantic import BaseModel, Field

from .client import LocalLLMAIClient

DEFAULT_VISION_ENDPOINT = "http://localhost:11434"
DEFAULT_VISION_MODEL = "qwen2.5vl:3b"
DEFAULT_VISION_TIMEOUT_SECONDS = 20.0
DEFAULT_VISION_PROMPT = (
    "Describe what is currently visible on this screen in concise detail: "
    "the active application/game, on-screen text, UI elements, and anything "
    "state-relevant for deciding the next action. Be factual, not creative."
)


class VisionDescribeError(RuntimeError):
    pass


class VisionDescribeRequest(BaseModel):
    image_base64: str = Field(min_length=1, max_length=8_000_000)
    prompt: str | None = Field(default=None, max_length=2000)


class VisionDescribeResponse(BaseModel):
    description: str
    model: str


async def describe_image(
    image_base64: str,
    *,
    prompt: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
) -> tuple[str, str]:
    """Returns (description, model_used)."""
    if not image_base64.strip():
        raise VisionDescribeError("image_base64 is empty")

    endpoint = LocalLLMAIClient._normalize_localhost_endpoint(
        os.getenv("JARVIS_VISION_MODEL_ENDPOINT", DEFAULT_VISION_ENDPOINT).rstrip("/")
    )
    resolved_model = model or os.getenv("JARVIS_VISION_MODEL", DEFAULT_VISION_MODEL)
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else _float_env("JARVIS_VISION_MODEL_TIMEOUT_SECONDS", DEFAULT_VISION_TIMEOUT_SECONDS)
    )

    payload = {
        "model": resolved_model,
        "messages": [
            {
                "role": "user",
                "content": prompt or DEFAULT_VISION_PROMPT,
                "images": [image_base64],
            }
        ],
        "stream": False,
    }

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.post(f"{endpoint}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise VisionDescribeError(
                        f"vision model HTTP {resp.status}: {text[:300]}"
                    )
                data = await resp.json()
    except aiohttp.ClientError as exc:
        raise VisionDescribeError(f"vision model network error: {exc}") from exc

    message = data.get("message") if isinstance(data, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise VisionDescribeError("vision model returned an empty description")
    return content.strip(), resolved_model


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
