from __future__ import annotations

import asyncio
import contextlib
import io
import os
import platform
import shlex
import shutil
import struct
import subprocess
import tempfile
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from .schemas import TextToSpeechChunk, TextToSpeechPCMRequest, TextToSpeechRequest

DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_MELOTTS_MODEL = "melotts/KR"
DEFAULT_PIPER_MODEL = "piper/default"
DEFAULT_TTS_MODEL_OPTIONS = (
    DEFAULT_TTS_MODEL,
    DEFAULT_MELOTTS_MODEL,
    DEFAULT_PIPER_MODEL,
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
)
DEFAULT_TTS_MODEL_ALIASES = {"", "default", "gpt-4o-mini-tts", "tts-1", "tts-1-hd"}
LOCAL_TTS_MODEL_FALLBACK = "local/system-tts"

MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


@dataclass(frozen=True, slots=True)
class TextToSpeechResult:
    audio: bytes
    media_type: str
    provider: str
    model: str
    voice: str
    response_format: str


class TextToSpeechError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class LocalTextToSpeechEngine:
    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        self.timeout_seconds = timeout_seconds or _tts_timeout_seconds()

    @property
    def model_name(self) -> str:
        if os.getenv("JARVIS_LOCAL_TTS_COMMAND"):
            return os.getenv("JARVIS_LOCAL_TTS_MODEL", DEFAULT_TTS_MODEL)
        return self.runtime_model_name()

    def supports_model_selection(self) -> bool:
        command = os.getenv("JARVIS_LOCAL_TTS_COMMAND", "")
        return bool(command and "{model}" in command)

    def runtime_model_name(self) -> str:
        espeak_ng = shutil.which("espeak-ng")
        if espeak_ng:
            return "local/espeak-ng"
        espeak = shutil.which("espeak")
        if espeak:
            return "local/espeak"
        if (
            platform.system() == "Darwin"
            and shutil.which("say") is not None
            and shutil.which("afconvert") is not None
        ):
            return "local/macos-say"
        return LOCAL_TTS_MODEL_FALLBACK

    def runtime_label(self) -> str:
        labels = {
            "local/espeak-ng": "Local system TTS (espeak-ng)",
            "local/espeak": "Local system TTS (espeak)",
            "local/macos-say": "Local system TTS (macOS say)",
            LOCAL_TTS_MODEL_FALLBACK: "Local system TTS",
        }
        return labels.get(self.runtime_model_name(), self.runtime_model_name())

    def is_available(self) -> bool:
        if os.getenv("JARVIS_LOCAL_TTS_COMMAND"):
            return True
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            return True
        return (
            platform.system() == "Darwin"
            and shutil.which("say") is not None
            and shutil.which("afconvert") is not None
        )

    async def synthesize_wav(
        self,
        *,
        text: str,
        voice: str,
        sample_rate: int,
        model: str | None = None,
    ) -> bytes:
        with tempfile.TemporaryDirectory(prefix="jarvis-tts-") as tmp:
            output = Path(tmp) / "speech.wav"
            await self._render_to_wav(
                text=text,
                voice=voice,
                sample_rate=sample_rate,
                model=model,
                output=output,
            )
            return output.read_bytes()

    async def synthesize_pcm(
        self,
        *,
        text: str,
        voice: str,
        sample_rate: int,
        channels: int,
        sample_width: int,
        model: str | None = None,
    ) -> bytes:
        wav_audio = await self.synthesize_wav(
            text=text,
            voice=voice,
            sample_rate=sample_rate,
            model=model,
        )
        return _wav_to_pcm_s16le(
            wav_audio,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )

    async def _render_to_wav(
        self,
        *,
        text: str,
        voice: str,
        sample_rate: int,
        model: str | None,
        output: Path,
    ) -> None:
        custom_command = os.getenv("JARVIS_LOCAL_TTS_COMMAND")
        if custom_command:
            args = shlex.split(
                custom_command.format(
                    model=shlex.quote(model or self.model_name),
                    output=shlex.quote(str(output)),
                    sample_rate=sample_rate,
                    text=shlex.quote(text),
                    voice=shlex.quote(self._voice_arg(voice)),
                )
            )
            await self._run(args)
            self._ensure_output(output)
            return

        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if espeak:
            args = [espeak, "-w", str(output)]
            voice_arg = self._voice_arg(voice)
            if voice_arg != "default":
                args.extend(["-v", voice_arg])
            args.append(text)
            await self._run(args)
            self._ensure_output(output)
            return

        if platform.system() == "Darwin" and shutil.which("say") and shutil.which("afconvert"):
            await self._render_macos_say(
                text=text,
                voice=voice,
                sample_rate=sample_rate,
                output=output,
            )
            self._ensure_output(output)
            return

        raise TextToSpeechError(
            503,
            "local TTS runtime is unavailable; install espeak-ng/espeak, use macOS say, "
            "or set JARVIS_LOCAL_TTS_COMMAND",
        )

    async def _render_macos_say(
        self,
        *,
        text: str,
        voice: str,
        sample_rate: int,
        output: Path,
    ) -> None:
        aiff_output = output.with_suffix(".aiff")
        voice_arg = self._voice_arg(voice)
        say_args = ["say", "-o", str(aiff_output)]
        if voice_arg != "default":
            say_args.extend(["-v", voice_arg])
        say_args.append(text)

        try:
            await self._run(say_args)
        except TextToSpeechError:
            if voice_arg == "default":
                raise
            fallback_args = ["say", "-o", str(aiff_output), text]
            await self._run(fallback_args)

        await self._run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{sample_rate}", str(aiff_output), str(output)]
        )

    async def _run(self, args: list[str]) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise TextToSpeechError(503, f"local TTS command not found: {args[0]}") from exc

        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            raise TextToSpeechError(504, "local TTS command timed out") from exc

        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise TextToSpeechError(
                503,
                detail or f"local TTS command failed with exit code {process.returncode}",
            )

    @staticmethod
    def _ensure_output(output: Path) -> None:
        if not output.exists() or output.stat().st_size == 0:
            raise TextToSpeechError(503, "local TTS command produced no audio")

    @staticmethod
    def _voice_arg(voice: str) -> str:
        selected = (voice or "").strip()
        configured = os.getenv("JARVIS_LOCAL_TTS_VOICE", "").strip()
        if configured:
            return configured
        if os.getenv("JARVIS_LOCAL_TTS_PASSTHROUGH_VOICE") == "1" and selected:
            return selected
        return "default"


class TextToSpeechService:
    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        openai_endpoint: str | None = None,
        local_engine: LocalTextToSpeechEngine | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.openai_api_key = openai_api_key
        self.openai_endpoint = openai_endpoint
        self.timeout_seconds = timeout_seconds or _tts_timeout_seconds()
        self.local_engine = local_engine or LocalTextToSpeechEngine(
            timeout_seconds=self.timeout_seconds
        )

    async def synthesize(self, request: TextToSpeechRequest) -> TextToSpeechResult:
        if request.provider not in {"openai", "local"}:
            raise TextToSpeechError(400, f"unsupported TTS provider: {request.provider}")
        sample_rate = _local_sample_rate()
        model = self._effective_model(request.model)
        if request.response_format == "pcm":
            audio = await self.local_engine.synthesize_pcm(
                text=request.text,
                voice=request.voice,
                sample_rate=sample_rate,
                channels=1,
                sample_width=2,
                model=model,
            )
            media_type = MEDIA_TYPES["pcm"]
            response_format = "pcm"
        else:
            audio = await self.local_engine.synthesize_wav(
                text=request.text,
                voice=request.voice,
                sample_rate=sample_rate,
                model=model,
            )
            media_type = MEDIA_TYPES["wav"]
            response_format = "wav"
        return TextToSpeechResult(
            audio=audio,
            media_type=media_type,
            provider="local",
            model=model,
            voice=LocalTextToSpeechEngine._voice_arg(request.voice),
            response_format=response_format,
        )

    def list_models(self) -> list[dict[str, object]]:
        if not self._has_model_selectable_runtime():
            model_id = self.local_engine.model_name
            return [
                {
                    "id": model_id,
                    "label": _local_runtime_label(self.local_engine),
                    "provider": "local",
                    "is_default": True,
                }
            ]

        configured = _configured_tts_models()
        default_model = self._effective_pcm_model(None)
        configured_models = configured if configured else list(DEFAULT_TTS_MODEL_OPTIONS)
        model_ids = _dedupe_models([default_model, *configured_models])
        return [
            {
                "id": model_id,
                "label": model_id,
                "provider": _provider_for_model(model_id)
                if self._server_pcm_endpoint() is not None
                else "local-command",
                "is_default": model_id == default_model,
            }
            for model_id in model_ids
        ]

    def server_pcm_headers(self, request: TextToSpeechPCMRequest) -> dict[str, str]:
        provider = (
            _provider_for_model(self._effective_pcm_model(request.model))
            if self._server_pcm_endpoint() is not None
            else "local"
        )
        return {
            "X-TTS-Provider": provider,
            "X-TTS-Model": self._effective_pcm_model(request.model),
            "X-TTS-Voice": request.voice,
            "X-TTS-Format": request.format,
            "X-TTS-Sample-Rate": str(request.sample_rate),
            "X-TTS-Channels": str(request.channels),
            "X-TTS-Sample-Width": str(request.sample_width),
            "X-TTS-Chunk-Count": str(len(request.chunks)),
            "X-AI-Generated-Voice": "true",
        }

    def ensure_server_pcm_available(self) -> None:
        if self._server_pcm_endpoint() is not None:
            return
        if self.local_engine.is_available():
            return
        raise TextToSpeechError(
            503,
            "missing local TTS runtime; set JARVIS_TTS_SERVER_URL, "
            "JARVIS_TTS_SERVER_PCM_ENDPOINT, or JARVIS_LOCAL_TTS_COMMAND",
        )

    async def stream_server_pcm(
        self, request: TextToSpeechPCMRequest
    ) -> AsyncIterator[bytes]:
        endpoint = self._server_pcm_endpoint()
        model = self._effective_pcm_model(request.model)
        if endpoint is None:
            for chunk in request.chunks:
                audio = await self.local_engine.synthesize_pcm(
                    text=chunk.text,
                    voice=request.voice,
                    sample_rate=request.sample_rate,
                    channels=request.channels,
                    sample_width=request.sample_width,
                    model=model,
                )
                if audio:
                    yield audio
            return

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for chunk in request.chunks:
                async for part in self._stream_server_pcm_chunk(
                    session,
                    endpoint,
                    request,
                    chunk,
                    model,
                ):
                    yield part

    def _server_pcm_endpoint(self) -> str | None:
        explicit = os.getenv("JARVIS_TTS_SERVER_PCM_ENDPOINT")
        if explicit:
            return explicit
        base_url = os.getenv("JARVIS_TTS_SERVER_URL", "").strip().rstrip("/")
        return f"{base_url}/tts/pcm" if base_url else None

    async def _stream_server_pcm_chunk(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        request: TextToSpeechPCMRequest,
        chunk: TextToSpeechChunk,
        model: str,
    ) -> AsyncIterator[bytes]:
        payload: dict[str, Any] = {
            "text": chunk.text,
            "voice": request.voice,
            "model": model,
            "sample_rate": request.sample_rate,
            "channels": request.channels,
            "sample_width": request.sample_width,
            "format": request.format,
        }
        if chunk.id:
            payload["chunk_id"] = chunk.id

        async with session.post(endpoint, json=payload) as response:
            if response.status >= 400:
                detail = await response.text()
                raise TextToSpeechError(response.status, detail or response.reason)
            async for part in response.content.iter_chunked(8192):
                if part:
                    yield part

    def _effective_model(self, request_model: str | None) -> str:
        model = _normalize_requested_model(request_model)
        if model:
            if self._local_engine_supports_model_selection():
                return model
            local_model = self.local_engine.model_name
            if model == local_model:
                return model
            raise TextToSpeechError(
                503,
                f"selected TTS model {model!r} requires a model-selectable local TTS "
                "command; set JARVIS_LOCAL_TTS_COMMAND with a {model} placeholder",
            )
        return self.local_engine.model_name

    def _effective_pcm_model(self, request_model: str | None) -> str:
        model = _normalize_requested_model(request_model)
        if model:
            if self._has_model_selectable_runtime():
                return model
            local_model = self.local_engine.model_name
            if model == local_model:
                return model
            raise TextToSpeechError(
                503,
                f"selected TTS model {model!r} requires a model-selectable TTS runtime; "
                "set JARVIS_TTS_SERVER_URL, JARVIS_TTS_SERVER_PCM_ENDPOINT, or "
                "JARVIS_LOCAL_TTS_COMMAND with a {model} placeholder",
            )
        if self._server_pcm_endpoint() is not None:
            return os.getenv("JARVIS_TTS_SERVER_MODEL") or DEFAULT_TTS_MODEL
        return self.local_engine.model_name

    def _has_model_selectable_runtime(self) -> bool:
        return (
            self._server_pcm_endpoint() is not None
            or self._local_engine_supports_model_selection()
        )

    def _local_engine_supports_model_selection(self) -> bool:
        supports = getattr(self.local_engine, "supports_model_selection", None)
        if callable(supports):
            return bool(supports())
        return False


def _local_runtime_label(local_engine: object) -> str:
    label = getattr(local_engine, "runtime_label", None)
    if callable(label):
        return str(label())
    model_name = getattr(local_engine, "model_name", LOCAL_TTS_MODEL_FALLBACK)
    return str(model_name)


def _normalize_requested_model(request_model: str | None) -> str | None:
    model = "" if request_model is None else str(request_model).strip()
    if model.lower() in DEFAULT_TTS_MODEL_ALIASES:
        return None
    return model


def _provider_for_model(model_id: str) -> str:
    normalized = model_id.strip().lower()
    if normalized.startswith(("melotts", "melo/")) or "melotts" in normalized:
        return "melotts"
    if normalized.startswith("piper"):
        return "piper"
    return "qwen"


def _configured_tts_models() -> list[str]:
    raw = os.getenv("JARVIS_TTS_MODELS", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _dedupe_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        if model not in seen:
            seen.add(model)
            result.append(model)
    return result


def _tts_timeout_seconds() -> float:
    raw = os.getenv("JARVIS_TTS_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return 60.0
    with contextlib.suppress(ValueError):
        timeout = float(raw)
        if timeout > 0:
            return timeout
    return 60.0


def _local_sample_rate() -> int:
    raw = os.getenv("JARVIS_LOCAL_TTS_SAMPLE_RATE", "24000")
    try:
        sample_rate = int(raw)
    except ValueError as exc:
        raise TextToSpeechError(503, "invalid JARVIS_LOCAL_TTS_SAMPLE_RATE") from exc
    if not 8000 <= sample_rate <= 48000:
        raise TextToSpeechError(503, "JARVIS_LOCAL_TTS_SAMPLE_RATE must be 8000-48000")
    return sample_rate


def _wav_to_pcm_s16le(
    wav_audio: bytes,
    *,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> bytes:
    if sample_width != 2:
        raise TextToSpeechError(400, "only 16-bit PCM output is supported")

    try:
        with wave.open(io.BytesIO(wav_audio), "rb") as wav_file:
            source_channels = wav_file.getnchannels()
            source_width = wav_file.getsampwidth()
            source_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, EOFError) as exc:
        raise TextToSpeechError(503, "local TTS did not produce readable WAV audio") from exc

    samples = _decode_pcm_to_mono(frames, channels=source_channels, sample_width=source_width)
    if source_rate != sample_rate:
        samples = _resample_linear(samples, source_rate=source_rate, target_rate=sample_rate)
    if channels == 2:
        stereo_samples: list[int] = []
        for sample in samples:
            stereo_samples.extend([sample, sample])
        samples = stereo_samples
    return struct.pack(f"<{len(samples)}h", *samples) if samples else b""


def _decode_pcm_to_mono(data: bytes, *, channels: int, sample_width: int) -> list[int]:
    if channels < 1:
        raise TextToSpeechError(503, "local TTS produced invalid channel count")
    if sample_width == 1:
        raw_samples = [(byte - 128) << 8 for byte in data]
    elif sample_width == 2:
        raw_samples = list(struct.unpack(f"<{len(data) // 2}h", data))
    elif sample_width == 4:
        int32_samples = struct.unpack(f"<{len(data) // 4}i", data)
        raw_samples = [sample >> 16 for sample in int32_samples]
    else:
        raise TextToSpeechError(503, "local TTS produced unsupported PCM sample width")

    if channels == 1:
        return raw_samples

    mono: list[int] = []
    for index in range(0, len(raw_samples), channels):
        frame = raw_samples[index : index + channels]
        if frame:
            mono.append(int(sum(frame) / len(frame)))
    return mono


def _resample_linear(
    samples: list[int],
    *,
    source_rate: int,
    target_rate: int,
) -> list[int]:
    if not samples or source_rate == target_rate:
        return samples
    output_length = max(1, int(len(samples) * target_rate / source_rate))
    ratio = source_rate / target_rate
    output: list[int] = []
    for index in range(output_length):
        source_position = index * ratio
        left = int(source_position)
        right = min(left + 1, len(samples) - 1)
        fraction = source_position - left
        value = samples[left] * (1.0 - fraction) + samples[right] * fraction
        output.append(max(-32768, min(32767, int(value))))
    return output
