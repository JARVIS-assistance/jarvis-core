from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen
from uuid import uuid4

import aiohttp

from .schemas import AIRequest, AIResponse, AIStreamChunk, AITokenStreamEvent

logger = logging.getLogger(__name__)


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid integer env %s=%r", name, raw)
        return None


def _optional_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid float env %s=%r", name, raw)
        return None


def _normalize_ollama_keep_alive(raw_keep_alive: str | None) -> str | None:
    if raw_keep_alive is None:
        return None
    value = raw_keep_alive.strip()
    if not value:
        return None
    if value.startswith("-"):
        return None
    return value


class AIClient(Protocol):
    async def stream_chat(
        self, message: str, route: str, request_id: str
    ) -> AsyncGenerator[AIStreamChunk, None]: ...

    async def respond_once(self, request: AIRequest) -> AIResponse: ...

    async def stream_tokens(self, request: AIRequest) -> AsyncGenerator[str, None]: ...

    async def stream_token_events(
        self, request: AIRequest
    ) -> AsyncGenerator[AITokenStreamEvent, None]: ...

    async def realtime_session_start(self, request: AIRequest) -> str: ...

    async def realtime_session_send(
        self, realtime_session_id: str, request: AIRequest
    ) -> AsyncGenerator[str, None]: ...

    async def realtime_session_close(self, realtime_session_id: str) -> None: ...

    async def cancel_generation(self, realtime_session_id: str) -> None: ...


class StubAIClient:
    async def stream_chat(
        self, message: str, route: str, request_id: str
    ) -> AsyncGenerator[AIStreamChunk, None]:
        yield {"type": "meta", "route": route, "request_id": request_id}
        yield {"type": "token", "content": "[stub] reasoning started"}
        yield {"type": "done", "content": "[stub] response complete"}

    async def respond_once(self, request: AIRequest) -> AIResponse:
        system_prefix = ""
        if request.get("system_prompt"):
            system_prefix = f"[system:{request['system_prompt'][:80]}] "
        return {
            "provider_mode": request["provider_mode"],
            "provider_name": request["provider_name"],
            "model_name": request["model_name"],
            "content": f"[stub:{request['provider_mode']}] {system_prefix}{request['message']}",
        }

    async def stream_tokens(self, request: AIRequest) -> AsyncGenerator[str, None]:
        result = await self.respond_once(request)
        for token in result["content"].split():
            yield token + " "

    async def stream_token_events(
        self, request: AIRequest
    ) -> AsyncGenerator[AITokenStreamEvent, None]:
        async for token in self.stream_tokens(request):
            yield {"type": "token", "content": token}
        yield {"type": "done", "done_reason": "stop"}

    async def realtime_session_start(self, request: AIRequest) -> str:
        return str(uuid4())

    async def realtime_session_send(
        self, realtime_session_id: str, request: AIRequest
    ) -> AsyncGenerator[str, None]:
        async for token in self.stream_tokens(request):
            yield token

    async def realtime_session_close(self, realtime_session_id: str) -> None:
        return None

    async def cancel_generation(self, realtime_session_id: str) -> None:
        return None


class TokenAIClient(StubAIClient):
    @staticmethod
    def _build_messages(request: AIRequest) -> list[dict[str, str]]:
        if request.get("messages"):
            return [
                {"role": message["role"], "content": message["content"]}
                for message in request["messages"]
            ]
        messages: list[dict[str, str]] = []
        if request.get("system_prompt"):
            messages.append({"role": "system", "content": str(request["system_prompt"])})
        messages.append({"role": "user", "content": request["message"]})
        return messages

    @staticmethod
    def _build_prompt_text(request: AIRequest) -> str:
        parts: list[str] = []
        if request.get("system_prompt"):
            parts.append(f"System:\n{request['system_prompt']}")
        parts.append(f"User:\n{request['message']}")
        return "\n\n".join(parts)

    @staticmethod
    def _post_json(
        url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 60
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(url=url, data=body, headers=headers, method="POST")
        with urlopen(req, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    @staticmethod
    def _openai_key(explicit: str | None) -> str | None:
        return explicit or os.getenv("OPENAI_API_KEY")

    @staticmethod
    def _gemini_key(explicit: str | None) -> str | None:
        return explicit or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    async def respond_once(self, request: AIRequest) -> AIResponse:
        provider = request["provider_name"].lower()
        try:
            if provider in {"openai", "chatgpt"}:
                key = self._openai_key(request.get("api_key"))
                if not key:
                    raise ValueError("missing OpenAI API key")
                endpoint = (
                    request.get("endpoint")
                    or "https://api.openai.com/v1/chat/completions"
                )
                data = self._post_json(
                    endpoint,
                    {
                        "model": request["model_name"],
                        "messages": self._build_messages(request),
                    },
                    {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                )
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                if not content:
                    raise ValueError("empty OpenAI response")
                return {
                    "provider_mode": "token",
                    "provider_name": request["provider_name"],
                    "model_name": request["model_name"],
                    "content": str(content),
                }

            if provider == "gemini":
                key = self._gemini_key(request.get("api_key"))
                if not key:
                    raise ValueError("missing Gemini API key")
                base = (
                    request.get("endpoint")
                    or "https://generativelanguage.googleapis.com/v1beta/models"
                )
                endpoint = f"{base}/{request['model_name']}:generateContent?key={key}"
                data = self._post_json(
                    endpoint,
                    {
                        "contents": [
                            {
                                "parts": [
                                    {"text": self._build_prompt_text(request)}
                                ]
                            }
                        ]
                    },
                    {"Content-Type": "application/json"},
                )
                parts = (
                    data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                )
                content = parts[0].get("text") if parts else None
                if not content:
                    raise ValueError("empty Gemini response")
                return {
                    "provider_mode": "token",
                    "provider_name": request["provider_name"],
                    "model_name": request["model_name"],
                    "content": str(content),
                }

            raise ValueError(f"unsupported token provider: {request['provider_name']}")
        except (
            HTTPError,
            URLError,
            TimeoutError,
            KeyError,
            IndexError,
            ValueError,
        ) as exc:
            return {
                "provider_mode": "token",
                "provider_name": request["provider_name"],
                "model_name": request["model_name"],
                "content": f"[provider-error] {exc}",
            }


class LocalLLMAIClient(StubAIClient):
    OLLAMA_PROVIDER_NAMES = {
        "ollama",
        "ollama_chat",
        "ollama-generate",
        "ollma",
        "ollmahost",
    }

    def __init__(self) -> None:
        self._stream_session: aiohttp.ClientSession | None = None
        self._stream_session_timeout_key: (
            tuple[float | None, float | None, float | None] | None
        ) = None

    @staticmethod
    def _timeout_key(
        timeout: aiohttp.ClientTimeout,
    ) -> tuple[float | None, float | None, float | None]:
        return timeout.total, timeout.connect, timeout.sock_read

    async def _session_for_timeout(
        self,
        timeout: aiohttp.ClientTimeout,
    ) -> aiohttp.ClientSession:
        key = self._timeout_key(timeout)
        session = self._stream_session
        if (
            session is not None
            and not getattr(session, "closed", False)
            and self._stream_session_timeout_key == key
        ):
            return session
        if session is not None and not getattr(session, "closed", False):
            await session.close()
        self._stream_session = aiohttp.ClientSession(timeout=timeout)
        self._stream_session_timeout_key = key
        return self._stream_session

    async def close(self) -> None:
        session = self._stream_session
        if session is not None and not getattr(session, "closed", False):
            await session.close()
        self._stream_session = None
        self._stream_session_timeout_key = None

    @staticmethod
    def _build_messages(request: AIRequest) -> list[dict[str, str]]:
        if request.get("messages"):
            return [
                {"role": message["role"], "content": message["content"]}
                for message in request["messages"]
            ]
        messages: list[dict[str, str]] = []
        if request.get("system_prompt"):
            messages.append({"role": "system", "content": str(request["system_prompt"])})
        messages.append({"role": "user", "content": request["message"]})
        return messages

    @staticmethod
    def _build_prompt_text(request: AIRequest) -> str:
        parts: list[str] = []
        system_prompt = request.get("system_prompt")
        if system_prompt:
            parts.append(f"System:\n{system_prompt}")
        parts.append(f"User:\n{request['message']}")
        return "\n\n".join(parts)

    @staticmethod
    def _is_ollama_provider(provider_name: str) -> bool:
        normalized = provider_name.strip().lower().replace(" ", "_")
        return normalized in LocalLLMAIClient.OLLAMA_PROVIDER_NAMES

    @staticmethod
    def _normalize_localhost_endpoint(raw_endpoint: str) -> str:
        """Make host-local model APIs reachable from Dockerized JARVIS services."""
        if os.getenv("JARVIS_REWRITE_DOCKER_LOCALHOST", "1").lower() in {
            "0",
            "false",
            "no",
        }:
            return raw_endpoint
        if not os.path.exists("/.dockerenv"):
            return raw_endpoint

        parsed = urlparse(raw_endpoint)
        if parsed.hostname not in {"localhost", "127.0.0.1"}:
            return raw_endpoint

        netloc = "host.docker.internal"
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        if parsed.username or parsed.password:
            auth = parsed.username or ""
            if parsed.password:
                auth = f"{auth}:{parsed.password}"
            netloc = f"{auth}@{netloc}"
        return urlunparse(parsed._replace(netloc=netloc))

    @staticmethod
    def _resolve_ollama_endpoint(
        raw_endpoint: str | None,
        provider_name: str | None = None,
    ) -> tuple[str, str]:
        raw = LocalLLMAIClient._normalize_localhost_endpoint(
            (raw_endpoint or "http://localhost:11434").rstrip("/")
        )
        path = urlparse(raw).path.rstrip("/")
        if path.endswith("/api/chat") or path.endswith("/chat"):
            return raw, "chat"
        if path.endswith("/api/generate") or path.endswith("/generate"):
            return raw, "generate"
        if (
            provider_name
            and provider_name.strip().lower().replace(" ", "_") == "ollama_chat"
        ):
            return f"{raw}/api/chat", "chat"
        return f"{raw}/api/generate", "generate"

    @staticmethod
    def _ollama_fallback_endpoint(endpoint: str, endpoint_type: str) -> str | None:
        parsed = urlparse(endpoint)
        path = parsed.path.rstrip("/")
        if endpoint_type == "chat" and path.endswith("/api/chat"):
            return urlunparse(
                parsed._replace(path=path.removesuffix("/api/chat") + "/chat")
            )
        if endpoint_type == "generate" and path.endswith("/api/generate"):
            return urlunparse(
                parsed._replace(path=path.removesuffix("/api/generate") + "/generate")
            )
        return None

    @staticmethod
    def _ollama_options_for_request(request: AIRequest) -> dict[str, Any]:
        route = str(request.get("route") or "realtime").upper()
        route_prefix = f"JARVIS_OLLAMA_{route}_"
        generic_prefix = "JARVIS_OLLAMA_"
        option_envs: tuple[tuple[str, str, str], ...] = (
            ("num_predict", "NUM_PREDICT", "int"),
            ("num_ctx", "NUM_CTX", "int"),
            ("num_thread", "NUM_THREAD", "int"),
            ("num_gpu", "NUM_GPU", "int"),
            ("temperature", "TEMPERATURE", "float"),
            ("top_p", "TOP_P", "float"),
            ("top_k", "TOP_K", "int"),
            ("repeat_penalty", "REPEAT_PENALTY", "float"),
        )
        options: dict[str, Any] = {}
        for option_name, env_suffix, kind in option_envs:
            route_env = f"{route_prefix}{env_suffix}"
            generic_env = f"{generic_prefix}{env_suffix}"
            value = (
                _optional_int_env(route_env)
                if kind == "int"
                else _optional_float_env(route_env)
            )
            if value is None:
                value = (
                    _optional_int_env(generic_env)
                    if kind == "int"
                    else _optional_float_env(generic_env)
                )
            if value is not None:
                options[option_name] = value
        return options

    @staticmethod
    def _apply_ollama_runtime_options(
        payload: dict[str, Any],
        request: AIRequest,
    ) -> None:
        options = LocalLLMAIClient._ollama_options_for_request(request)
        if options:
            payload["options"] = options
        route = str(request.get("route") or "realtime").upper()
        keep_alive = (
            os.getenv(f"JARVIS_OLLAMA_{route}_KEEP_ALIVE")
            or os.getenv("JARVIS_OLLAMA_KEEP_ALIVE")
        )
        if keep_alive:
            normalized_keep_alive = _normalize_ollama_keep_alive(keep_alive)
            if normalized_keep_alive:
                payload["keep_alive"] = normalized_keep_alive
        think = os.getenv(
            f"JARVIS_OLLAMA_{route}_THINK",
            os.getenv("JARVIS_OLLAMA_THINK", "false" if route == "REALTIME" else ""),
        )
        if think.strip():
            payload["think"] = think.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _ollama_timeout_for_request(request: AIRequest) -> aiohttp.ClientTimeout:
        route = str(request.get("route") or "realtime").upper()
        total = (
            _optional_float_env(f"JARVIS_OLLAMA_{route}_STREAM_TOTAL_TIMEOUT")
            or _optional_float_env("JARVIS_OLLAMA_STREAM_TOTAL_TIMEOUT")
            or (18.0 if route == "REALTIME" else 120.0)
        )
        connect = (
            _optional_float_env(f"JARVIS_OLLAMA_{route}_STREAM_CONNECT_TIMEOUT")
            or _optional_float_env("JARVIS_OLLAMA_STREAM_CONNECT_TIMEOUT")
            or 5.0
        )
        sock_read = (
            _optional_float_env(f"JARVIS_OLLAMA_{route}_STREAM_READ_TIMEOUT")
            or _optional_float_env("JARVIS_OLLAMA_STREAM_READ_TIMEOUT")
            or (4.0 if route == "REALTIME" else None)
        )
        min_sock_read = (
            _optional_float_env(f"JARVIS_OLLAMA_{route}_STREAM_MIN_READ_TIMEOUT")
            or _optional_float_env("JARVIS_OLLAMA_STREAM_MIN_READ_TIMEOUT")
            or (25.0 if route == "REALTIME" else None)
        )
        if min_sock_read is not None:
            sock_read = max(sock_read or min_sock_read, min_sock_read)
        if sock_read is not None:
            total = max(total, connect + sock_read + 5.0)
        return aiohttp.ClientTimeout(
            total=max(total, 1.0),
            connect=max(connect, 0.5),
            sock_read=max(sock_read, 0.5) if sock_read is not None else None,
        )

    @staticmethod
    def _ollama_cold_timeout_for_request(request: AIRequest) -> aiohttp.ClientTimeout:
        route = str(request.get("route") or "realtime").upper()
        base = LocalLLMAIClient._ollama_timeout_for_request(request)
        cold_sock_read = (
            _optional_float_env(f"JARVIS_OLLAMA_{route}_STREAM_COLD_READ_TIMEOUT")
            or _optional_float_env("JARVIS_OLLAMA_STREAM_COLD_READ_TIMEOUT")
            or (45.0 if route == "REALTIME" else base.sock_read)
        )
        sock_read = max(
            base.sock_read or 0.0,
            cold_sock_read or 0.0,
            0.5,
        )
        connect = max(base.connect or 0.5, 0.5)
        cold_total = (
            _optional_float_env(f"JARVIS_OLLAMA_{route}_STREAM_COLD_TOTAL_TIMEOUT")
            or _optional_float_env("JARVIS_OLLAMA_STREAM_COLD_TOTAL_TIMEOUT")
            or (connect + sock_read + 15.0)
        )
        total = max(base.total or 0.0, cold_total, connect + sock_read + 5.0)
        return aiohttp.ClientTimeout(
            total=total,
            connect=connect,
            sock_read=sock_read,
        )

    @staticmethod
    def _post_json(
        url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 120
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers.setdefault("User-Agent", "JARVIS/1.0")
        logger.info("[AI-REQUEST] POST %s", url)
        logger.info("[AI-REQUEST] headers=%s", headers)
        logger.info("[AI-REQUEST] payload=%s", json.dumps(payload, ensure_ascii=False))
        req = Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=timeout) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
                logger.info("[AI-RESPONSE] status=%s body=%s", response.status, raw[:500])
                return json.loads(raw) if raw else {}
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            logger.error(
                "[AI-ERROR] %s %s | status=%s body=%s",
                exc, url, exc.code, error_body[:500],
            )
            raise

    async def _respond_ollama(self, request: AIRequest) -> AIResponse:
        endpoint, endpoint_type = self._resolve_ollama_endpoint(
            request.get("endpoint"),
            request["provider_name"],
        )
        if endpoint_type == "chat":
            payload = {
                "model": request["model_name"],
                "messages": self._build_messages(request),
                "stream": False,
            }
        else:
            payload = {
                "model": request["model_name"],
                "prompt": self._build_prompt_text(request),
                "stream": False,
            }
        self._apply_ollama_runtime_options(payload, request)
        try:
            data = self._post_json(
                endpoint,
                payload,
                {"Content-Type": "application/json"},
            )
        except HTTPError as exc:
            fallback = self._ollama_fallback_endpoint(endpoint, endpoint_type)
            if exc.code != 404 or fallback is None:
                raise
            data = self._post_json(
                fallback,
                payload,
                {"Content-Type": "application/json"},
            )
        if endpoint_type == "chat":
            content = data.get("message", {}).get("content")
        else:
            content = data.get("response")
        if not content:
            raise ValueError("empty Ollama response")
        return {
            "provider_mode": "local",
            "provider_name": request["provider_name"],
            "model_name": request["model_name"],
            "content": str(content),
        }

    def _resolve_openai_compat_endpoint(self, raw_endpoint: str) -> str:
        """Resolve full URL for OpenAI-compatible endpoint."""
        path = urlparse(raw_endpoint).path
        if path and path != "/" and "/completions" in path:
            return raw_endpoint
        return f"{raw_endpoint}/v1/chat/completions"

    async def _respond_openai_compat(self, request: AIRequest) -> AIResponse:
        raw_endpoint = (request.get("endpoint") or "http://localhost:8080").rstrip("/")
        path = urlparse(raw_endpoint).path
        if path and path != "/" and "/completions" in path:
            # Full URL with path provided (e.g. http://host/engines/v1/chat/completions)
            endpoint = raw_endpoint
        else:
            # Base URL only — append default OpenAI-compatible path
            endpoint = f"{raw_endpoint}/v1/chat/completions"
        model = request["model_name"]
        data = self._post_json(
            endpoint,
            {
                "model": model,
                "messages": self._build_messages(request),
                "stream": False,
            },
            {"Content-Type": "application/json"},
        )
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            raise ValueError("empty response from local provider")
        return {
            "provider_mode": "local",
            "provider_name": request["provider_name"],
            "model_name": model,
            "content": str(content),
        }

    # ── SSE 비동기 스트리밍 ──────────────────────────────

    async def stream_tokens(self, request: AIRequest) -> AsyncGenerator[str, None]:
        """aiohttp로 OpenAI-compatible SSE 엔드포인트에서 토큰을 실시간 스트리밍."""
        async for event in self.stream_token_events(request):
            if event.get("type") == "token":
                content = event.get("content")
                if content:
                    yield content

    async def stream_token_events(
        self, request: AIRequest
    ) -> AsyncGenerator[AITokenStreamEvent, None]:
        """Stream token events, preserving provider completion metadata."""
        provider = request["provider_name"].lower()
        if self._is_ollama_provider(provider):
            async for event in self._stream_ollama_events(request):
                yield event
            return

        raw_endpoint = (request.get("endpoint") or "http://localhost:8080").rstrip("/")
        endpoint = self._resolve_openai_compat_endpoint(raw_endpoint)
        model = request["model_name"]
        payload = {
            "model": model,
            "messages": self._build_messages(request),
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "JARVIS/1.0",
        }
        logger.info("[AI-SSE-REQUEST] POST %s", endpoint)

        timeout = self._ollama_timeout_for_request(request)
        try:
            session = await self._session_for_timeout(timeout)
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("[AI-SSE-ERROR] status=%s body=%s", resp.status, body[:300])
                    yield {"type": "token", "content": f"[provider-error] HTTP {resp.status}"}
                    yield {"type": "done", "done_reason": "error"}
                    return

                count = 0
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            count += 1
                            yield {"type": "token", "content": content}
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
                logger.info("[AI-SSE-RESPONSE] streamed %d tokens", count)
                yield {"type": "done", "done_reason": "stop"}
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("[AI-SSE-ERROR] %s", exc)
            yield {"type": "token", "content": f"[provider-error] {exc}"}
            yield {"type": "done", "done_reason": "error"}

    # ── respond_once / realtime (unchanged pattern) ─────────────────

    async def _stream_ollama_tokens(
        self,
        request: AIRequest,
    ) -> AsyncGenerator[str, None]:
        async for event in self._stream_ollama_events(request):
            if event.get("type") == "token":
                content = event.get("content")
                if content:
                    yield content

    async def _stream_ollama_events(
        self,
        request: AIRequest,
    ) -> AsyncGenerator[AITokenStreamEvent, None]:
        timeout = self._ollama_timeout_for_request(request)
        yielded_any = False
        try:
            async for event in self._stream_ollama_events_once(request, timeout=timeout):
                if event.get("type") == "token":
                    yielded_any = True
                yield event
            return
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if yielded_any:
                logger.error("[AI-OLLAMA-STREAM-ERROR] %s", exc)
                yield {"type": "token", "content": f"[provider-error] {exc}"}
                yield {"type": "done", "done_reason": "error"}
                return

            cold_timeout = self._ollama_cold_timeout_for_request(request)
            logger.warning(
                "[AI-OLLAMA-STREAM-RETRY] no tokens before timeout; "
                "retrying with cold-start timeout total=%s read=%s error=%s",
                cold_timeout.total,
                cold_timeout.sock_read,
                exc,
            )
            try:
                async for event in self._stream_ollama_events_once(
                    request,
                    timeout=cold_timeout,
                ):
                    yield event
                return
            except (aiohttp.ClientError, asyncio.TimeoutError) as retry_exc:
                logger.error("[AI-OLLAMA-STREAM-ERROR] %s", retry_exc)
                yield {"type": "token", "content": f"[provider-error] {retry_exc}"}
                yield {"type": "done", "done_reason": "error"}

    async def _stream_ollama_events_once(
        self,
        request: AIRequest,
        *,
        timeout: aiohttp.ClientTimeout,
    ) -> AsyncGenerator[AITokenStreamEvent, None]:
        endpoint, endpoint_type = self._resolve_ollama_endpoint(
            request.get("endpoint"),
            request["provider_name"],
        )
        if endpoint_type == "chat":
            payload = {
                "model": request["model_name"],
                "messages": self._build_messages(request),
                "stream": True,
            }
        else:
            payload = {
                "model": request["model_name"],
                "prompt": self._build_prompt_text(request),
                "stream": True,
            }
        self._apply_ollama_runtime_options(payload, request)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
            "User-Agent": "JARVIS/1.0",
        }
        endpoints = [endpoint]
        fallback = self._ollama_fallback_endpoint(endpoint, endpoint_type)
        if fallback is not None:
            endpoints.append(fallback)

        session = await self._session_for_timeout(timeout)
        for candidate_endpoint in endpoints:
            logger.info("[AI-OLLAMA-STREAM-REQUEST] POST %s", candidate_endpoint)
            async with session.post(
                candidate_endpoint, json=payload, headers=headers
            ) as resp:
                if resp.status == 404 and candidate_endpoint != endpoints[-1]:
                    body = await resp.text()
                    logger.info(
                        "[AI-OLLAMA-STREAM-FALLBACK] status=404 endpoint=%s body=%s",
                        candidate_endpoint,
                        body[:300],
                    )
                    continue

                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "[AI-OLLAMA-STREAM-ERROR] status=%s body=%s",
                        resp.status,
                        body[:300],
                    )
                    yield {"type": "token", "content": f"[provider-error] HTTP {resp.status}"}
                    yield {"type": "done", "done_reason": "error"}
                    return

                count = 0
                done_reason = "stop"
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if endpoint_type == "chat":
                        token = chunk.get("message", {}).get("content")
                    else:
                        token = chunk.get("response")
                    if isinstance(token, str) and token:
                        count += 1
                        yield {"type": "token", "content": token}
                    if chunk.get("done") is True:
                        raw_done_reason = chunk.get("done_reason") or chunk.get("finish_reason")
                        if isinstance(raw_done_reason, str) and raw_done_reason:
                            done_reason = raw_done_reason
                        break
                logger.info("[AI-OLLAMA-STREAM-RESPONSE] streamed %d tokens", count)
                yield {"type": "done", "done_reason": done_reason}
                return

    async def respond_once(self, request: AIRequest) -> AIResponse:
        provider = request["provider_name"].lower()
        try:
            if self._is_ollama_provider(provider):
                return await self._respond_ollama(request)
            return await self._respond_openai_compat(request)
        except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as exc:
            return {
                "provider_mode": "local",
                "provider_name": request["provider_name"],
                "model_name": request["model_name"],
                "content": f"[provider-error] {exc}",
            }
