import asyncio
import json
import os

from ai.client import LocalLLMAIClient


def _clear_ollama_env(monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("JARVIS_OLLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("JARVIS_REWRITE_DOCKER_LOCALHOST", "0")


class _FakeContent:
    def __init__(self, lines: list[dict[str, object]]) -> None:
        self._lines = [json.dumps(line).encode("utf-8") + b"\n" for line in lines]

    def __aiter__(self):
        self._iter = iter(self._lines)
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeResponse:
    def __init__(
        self,
        lines: list[dict[str, object]] | None = None,
        *,
        status: int = 200,
        body: str = "",
    ) -> None:
        self.status = status
        self._body = body
        self.content = _FakeContent(
            lines
            or [
                {"response": "first ", "done": False},
                {"response": "second", "done": False},
                {"done": True},
            ]
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def text(self) -> str:
        return self._body


class _FakeSession:
    requests: list[dict[str, object]] = []
    timeouts: list[object] = []
    response_lines: list[dict[str, object]] | None = None
    statuses: list[int] = []

    def __init__(self, *, timeout) -> None:
        self.timeout = timeout
        self.closed = False
        self.timeouts.append(timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, endpoint, *, json, headers):
        self.requests.append({"endpoint": endpoint, "json": json, "headers": headers})
        status = self.statuses.pop(0) if self.statuses else 200
        return _FakeResponse(
            self.response_lines,
            status=status,
            body='{"error":"not found"}',
        )

    async def close(self) -> None:
        self.closed = True


class _NoRespondOnceLocalClient(LocalLLMAIClient):
    async def respond_once(self, request):  # pragma: no cover - should never be called
        raise AssertionError("ollama streaming must not fall back to respond_once")


def test_ollama_stream_tokens_uses_native_streaming(monkeypatch) -> None:
    _clear_ollama_env(monkeypatch)
    _FakeSession.requests.clear()
    _FakeSession.timeouts.clear()
    _FakeSession.response_lines = None
    monkeypatch.setattr("ai.client.aiohttp.ClientSession", _FakeSession)

    request = {
        "message": "hello",
        "route": "realtime",
        "request_id": "r-ollama",
        "provider_mode": "local",
        "provider_name": "ollama",
        "model_name": "llama3.2",
        "api_key": None,
        "endpoint": "http://ollama:11434",
        "system_prompt": "Be fast.",
        "messages": [],
    }

    async def collect() -> list[str]:
        return [token async for token in _NoRespondOnceLocalClient().stream_tokens(request)]

    assert asyncio.run(collect()) == ["first ", "second"]
    assert _FakeSession.requests == [
        {
            "endpoint": "http://ollama:11434/api/generate",
            "json": {
                "model": "llama3.2",
                "prompt": "System:\nBe fast.\n\nUser:\nhello",
                "stream": True,
                "think": False,
            },
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson",
                "User-Agent": "JARVIS/1.0",
            },
        }
    ]


def test_ollama_stream_tokens_uses_configured_chat_endpoint(monkeypatch) -> None:
    _clear_ollama_env(monkeypatch)
    _FakeSession.requests.clear()
    _FakeSession.timeouts.clear()
    _FakeSession.response_lines = [
        {"message": {"content": "first "}, "done": False},
        {"message": {"content": "second"}, "done": False},
        {"done": True},
    ]
    monkeypatch.setattr("ai.client.aiohttp.ClientSession", _FakeSession)

    request = {
        "message": "hello",
        "route": "realtime",
        "request_id": "r-ollama",
        "provider_mode": "local",
        "provider_name": "ollama",
        "model_name": "gemma4:e2b",
        "api_key": None,
        "endpoint": "https://ollma.breakpack.cc/chat",
        "system_prompt": "Be fast.",
        "messages": [],
    }

    async def collect() -> list[str]:
        return [token async for token in _NoRespondOnceLocalClient().stream_tokens(request)]

    assert asyncio.run(collect()) == ["first ", "second"]
    assert _FakeSession.requests == [
        {
            "endpoint": "https://ollma.breakpack.cc/chat",
            "json": {
                "model": "gemma4:e2b",
                "messages": [
                    {"role": "system", "content": "Be fast."},
                    {"role": "user", "content": "hello"},
                ],
                "stream": True,
                "think": False,
            },
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson",
                "User-Agent": "JARVIS/1.0",
            },
        }
    ]


def test_ollama_chat_stream_tokens_falls_back_to_wrapper_chat_endpoint(monkeypatch) -> None:
    _clear_ollama_env(monkeypatch)
    _FakeSession.requests.clear()
    _FakeSession.timeouts.clear()
    _FakeSession.statuses = [404, 200]
    _FakeSession.response_lines = [
        {"message": {"content": "wrapped "}, "done": False},
        {"message": {"content": "stream"}, "done": False},
        {"done": True},
    ]
    monkeypatch.setattr("ai.client.aiohttp.ClientSession", _FakeSession)

    request = {
        "message": "hello",
        "route": "realtime",
        "request_id": "r-ollama",
        "provider_mode": "local",
        "provider_name": "ollama_chat",
        "model_name": "qwen2.5:1.5b",
        "api_key": None,
        "endpoint": "http://localhost:3030",
        "system_prompt": "Be fast.",
        "messages": [],
    }

    async def collect() -> list[str]:
        return [token async for token in _NoRespondOnceLocalClient().stream_tokens(request)]

    assert asyncio.run(collect()) == ["wrapped ", "stream"]
    assert [item["endpoint"] for item in _FakeSession.requests] == [
        "http://localhost:3030/api/chat",
        "http://localhost:3030/chat",
    ]


class _TimeoutContent:
    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        raise asyncio.TimeoutError("Timeout on reading data from socket")


class _TimeoutThenSuccessSession(_FakeSession):
    attempts = 0

    def post(self, endpoint, *, json, headers):
        self.requests.append({"endpoint": endpoint, "json": json, "headers": headers})
        type(self).attempts += 1
        if type(self).attempts == 1:
            response = _FakeResponse([])
            response.content = _TimeoutContent()
            return response
        return _FakeResponse(
            [
                {"response": "retried ", "done": False},
                {"response": "ok", "done": False},
                {"done": True},
            ]
        )


def test_ollama_stream_tokens_retries_timeout_before_first_token(monkeypatch) -> None:
    _clear_ollama_env(monkeypatch)
    _TimeoutThenSuccessSession.requests.clear()
    _TimeoutThenSuccessSession.timeouts.clear()
    _TimeoutThenSuccessSession.response_lines = None
    _TimeoutThenSuccessSession.statuses = []
    _TimeoutThenSuccessSession.attempts = 0
    monkeypatch.setenv("JARVIS_OLLAMA_REALTIME_STREAM_READ_TIMEOUT", "12")
    monkeypatch.setenv("JARVIS_OLLAMA_REALTIME_STREAM_TOTAL_TIMEOUT", "18")
    monkeypatch.setattr("ai.client.aiohttp.ClientSession", _TimeoutThenSuccessSession)

    request = {
        "message": "hello",
        "route": "realtime",
        "request_id": "r-ollama",
        "provider_mode": "local",
        "provider_name": "ollama",
        "model_name": "gemma4:e2b",
        "api_key": None,
        "endpoint": "http://ollama:11434",
        "system_prompt": "Be fast.",
        "messages": [],
    }

    async def collect() -> list[str]:
        return [token async for token in _NoRespondOnceLocalClient().stream_tokens(request)]

    assert asyncio.run(collect()) == ["retried ", "ok"]
    assert len(_TimeoutThenSuccessSession.requests) == 2
    assert _TimeoutThenSuccessSession.timeouts[0].sock_read == 25.0
    assert _TimeoutThenSuccessSession.timeouts[0].total == 35.0
    assert _TimeoutThenSuccessSession.timeouts[1].sock_read == 45.0


def test_ollama_stream_token_events_preserves_done_reason(monkeypatch) -> None:
    _clear_ollama_env(monkeypatch)
    _FakeSession.requests.clear()
    _FakeSession.timeouts.clear()
    _FakeSession.response_lines = [
        {"response": "short ", "done": False},
        {"response": "answer", "done": False},
        {"done": True, "done_reason": "length"},
    ]
    monkeypatch.setattr("ai.client.aiohttp.ClientSession", _FakeSession)

    request = {
        "message": "hello",
        "route": "realtime",
        "request_id": "r-ollama",
        "provider_mode": "local",
        "provider_name": "ollama",
        "model_name": "gemma4:e2b",
        "api_key": None,
        "endpoint": "http://ollama:11434",
        "system_prompt": "Be fast.",
        "messages": [],
    }

    async def collect() -> list[dict[str, str]]:
        return [
            event
            async for event in _NoRespondOnceLocalClient().stream_token_events(request)
        ]

    assert asyncio.run(collect()) == [
        {"type": "token", "content": "short "},
        {"type": "token", "content": "answer"},
        {"type": "done", "done_reason": "length"},
    ]
