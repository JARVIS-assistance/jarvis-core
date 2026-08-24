import json
import logging
from tempfile import NamedTemporaryFile

from fastapi.testclient import TestClient
from jarvis_contracts import JarvisCoreEndpoints

from ai import AIService
from app import create_app
from application.audio.schemas import TextToSpeechPCMRequest
from application.audio.service import TextToSpeechResult, TextToSpeechService
from application.chat.service import _build_alternating_messages, _build_messages_for_model
from core.config.prompt_loader import load_prompt
from core.db.db_connection import connect
from core.db.db_operations import add_message, get_or_create_session_for_user


class RecordingAIClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.force_provider_error = False

    async def respond_once(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        if self.force_provider_error:
            content = "[provider-error] HTTP Error 500: Internal Server Error"
        else:
            content = (
                '{"goal":"메모리 확인","constraints":[],"steps":['
                '{"id":"s1","title":"확인","description":"메모리를 반영한다"}]}'
                if "planning engine" in str(request.get("system_prompt", ""))
                else "메모리를 반영한 응답"
            )
        return {
            "provider_mode": request["provider_mode"],
            "provider_name": request["provider_name"],
            "model_name": request["model_name"],
            "content": content,
        }

    async def stream_chat(self, message: str, route: str, request_id: str):
        yield {"type": "done", "content": ""}

    async def stream_tokens(self, request: dict[str, object]):
        self.requests.append(request)
        yield "ok"

    async def stream_token_events(self, request: dict[str, object]):
        async for token in self.stream_tokens(request):
            yield {"type": "token", "content": token}
        yield {"type": "done", "done_reason": "stop"}

    async def realtime_session_start(self, request: dict[str, object]) -> str:
        return "rt-test"

    async def realtime_session_send(
        self,
        realtime_session_id: str,
        request: dict[str, object],
    ):
        yield "ok"

    async def realtime_session_close(self, realtime_session_id: str) -> None:
        return None

    async def cancel_generation(self, realtime_session_id: str) -> None:
        return None


class LengthLimitedAIClient(RecordingAIClient):
    async def stream_token_events(self, request: dict[str, object]):
        self.requests.append(request)
        yield {"type": "token", "content": "짧게 "}
        yield {"type": "token", "content": "답변"}
        yield {"type": "done", "done_reason": "length"}


class RecordingTextToSpeechService:
    def __init__(self) -> None:
        self.requests: list[object] = []
        self.pcm_requests: list[object] = []

    async def synthesize(self, request: object) -> TextToSpeechResult:
        self.requests.append(request)
        return TextToSpeechResult(
            audio=b"audio-bytes",
            media_type="audio/mpeg",
            provider="openai",
            model="gpt-4o-mini-tts",
            voice="marin",
            response_format="mp3",
        )

    def ensure_server_pcm_available(self) -> None:
        return None

    def server_pcm_headers(self, request: object) -> dict[str, str]:
        return {
            "X-TTS-Provider": "server",
            "X-TTS-Voice": getattr(request, "voice", "default"),
            "X-TTS-Format": getattr(request, "format", "pcm_s16le"),
            "X-TTS-Sample-Rate": str(getattr(request, "sample_rate", 24000)),
            "X-TTS-Channels": str(getattr(request, "channels", 1)),
            "X-TTS-Sample-Width": str(getattr(request, "sample_width", 2)),
            "X-TTS-Chunk-Count": str(len(getattr(request, "chunks", []))),
            "X-AI-Generated-Voice": "true",
        }

    async def stream_server_pcm(self, request: object):
        self.pcm_requests.append(request)
        yield b"pcm-1"
        yield b"pcm-2"


class RecordingLocalTTSEngine:
    def __init__(
        self,
        *,
        model_name: str = "test-local-tts",
        supports_model_selection: bool = False,
    ) -> None:
        self._model_name = model_name
        self._supports_model_selection = supports_model_selection
        self.wav_requests: list[dict[str, object]] = []
        self.pcm_requests: list[dict[str, object]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def supports_model_selection(self) -> bool:
        return self._supports_model_selection

    def runtime_label(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return True

    async def synthesize_wav(
        self,
        *,
        text: str,
        voice: str,
        sample_rate: int,
        model: str | None = None,
    ) -> bytes:
        self.wav_requests.append(
            {"text": text, "voice": voice, "sample_rate": sample_rate, "model": model}
        )
        return b"local-wav"

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
        self.pcm_requests.append(
            {
                "text": text,
                "voice": voice,
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width,
                "model": model,
            }
        )
        return b"local-pcm"


def test_health() -> None:
    client = TestClient(create_app())
    response = client.get(JarvisCoreEndpoints.HEALTH.path)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["mode"] == "library-first"
    assert payload["capabilities"] == ["realtime", "deep"]


def test_internal_conversation_realtime_endpoint() -> None:
    client = TestClient(create_app())
    response = client.post(
        JarvisCoreEndpoints.INTERNAL_CONVERSATION_RESPOND.path,
        json={"mode": "realtime", "message": "배포 상태 알려줘"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "realtime"
    assert "실시간 응답" in payload["content"]


def test_internal_conversation_deep_endpoint() -> None:
    client = TestClient(create_app())
    response = client.post(
        JarvisCoreEndpoints.INTERNAL_CONVERSATION_RESPOND.path,
        json={"mode": "deep", "message": "Traceback: bad state"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "deep"
    assert "Deep thinking result" in payload["content"]


def test_internal_chat_request_accepts_deep_override() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/internal/chat/request",
        json={
            "message": "깊게 분석해줘",
            "route_override": "deep",
        },
        headers={"x-user-id": "u1", "x-user-email": "u1@example.com", "x-request-id": "r1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "deep"


def test_internal_audio_speech_returns_generated_audio() -> None:
    tts_service = RecordingTextToSpeechService()
    client = TestClient(create_app(tts_service=tts_service))
    response = client.post(
        JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH.path,
        json={
            "text": "안녕하세요",
            "voice": "marin",
            "response_format": "mp3",
        },
        headers={"x-user-id": "u1", "x-request-id": "r-tts"},
    )

    assert response.status_code == 200
    assert response.content == b"audio-bytes"
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["x-tts-provider"] == "openai"
    assert response.headers["x-tts-model"] == "gpt-4o-mini-tts"
    assert response.headers["x-ai-generated-voice"] == "true"
    assert tts_service.requests


def test_internal_audio_speech_uses_local_tts_without_cloud_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    local_engine = RecordingLocalTTSEngine()
    tts_service = TextToSpeechService(local_engine=local_engine)
    client = TestClient(create_app(tts_service=tts_service))
    response = client.post(
        JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH.path,
        json={
            "text": "안녕하세요",
            "voice": "marin",
            "response_format": "mp3",
        },
        headers={"x-user-id": "u1", "x-request-id": "r-tts-local"},
    )

    assert response.status_code == 200
    assert response.content == b"local-wav"
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-tts-provider"] == "local"
    assert response.headers["x-tts-model"] == "test-local-tts"
    assert response.headers["x-tts-voice"] == "default"
    assert response.headers["x-tts-format"] == "wav"
    assert local_engine.wav_requests == [
        {
            "text": "안녕하세요",
            "voice": "marin",
            "sample_rate": 24000,
            "model": "test-local-tts",
        }
    ]


def test_internal_audio_speech_pcm_streams_generated_chunks() -> None:
    tts_service = RecordingTextToSpeechService()
    client = TestClient(create_app(tts_service=tts_service))
    response = client.post(
        JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH_PCM.path,
        json={
            "chunks": [
                {"id": "c1", "text": "안녕"},
                {"id": "c2", "text": "하세요"},
            ],
            "voice": "main",
            "sample_rate": 24000,
        },
        headers={"x-user-id": "u1", "x-request-id": "r-tts-pcm"},
    )

    assert response.status_code == 200
    assert response.content == b"pcm-1pcm-2"
    assert response.headers["content-type"] == "audio/pcm"
    assert response.headers["x-tts-provider"] == "server"
    assert response.headers["x-tts-format"] == "pcm_s16le"
    assert response.headers["x-tts-sample-rate"] == "24000"
    assert response.headers["x-tts-chunk-count"] == "2"
    assert tts_service.pcm_requests


def test_internal_audio_speech_pcm_accepts_frontend_default_aliases() -> None:
    tts_service = RecordingTextToSpeechService()
    client = TestClient(create_app(tts_service=tts_service))
    response = client.post(
        JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH_PCM.path,
        json={
            "chunks": [{"id": "c-1770000000000", "text": "안녕하세요."}],
            "voice": "marin",
            "model": "gpt-4o-mini-tts",
            "sample_rate": 24000,
            "channels": 1,
            "sample_width": 2,
            "format": "pcm_s16le",
        },
        headers={"x-user-id": "u1", "x-request-id": "r-tts-pcm"},
    )

    assert response.status_code == 200
    assert response.content == b"pcm-1pcm-2"
    request = tts_service.pcm_requests[-1]
    assert getattr(request, "voice") == "default"
    assert getattr(request, "model") is None


def test_internal_audio_speech_pcm_uses_default_qwen_tts_for_frontend_model_alias(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JARVIS_TTS_SERVER_URL", "http://tts-runtime.local")
    monkeypatch.delenv("JARVIS_TTS_SERVER_PCM_ENDPOINT", raising=False)
    monkeypatch.delenv("JARVIS_LOCAL_TTS_MODEL", raising=False)
    body = TextToSpeechPCMRequest(
        chunks=[{"id": "c-1770000000000", "text": "안녕하세요."}],
        voice="default",
        model="gpt-4o-mini-tts",
        sample_rate=24000,
        channels=1,
        sample_width=2,
        format="pcm_s16le",
    )

    headers = TextToSpeechService().server_pcm_headers(body)

    assert body.model is None
    assert headers["X-TTS-Model"] == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def test_internal_audio_speech_pcm_preserves_user_selected_model(monkeypatch) -> None:
    monkeypatch.delenv("JARVIS_TTS_SERVER_URL", raising=False)
    monkeypatch.delenv("JARVIS_TTS_SERVER_PCM_ENDPOINT", raising=False)
    local_engine = RecordingLocalTTSEngine(supports_model_selection=True)
    tts_service = TextToSpeechService(local_engine=local_engine)
    client = TestClient(create_app(tts_service=tts_service))
    response = client.post(
        JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH_PCM.path,
        json={
            "chunks": [{"id": "c-1770000000000", "text": "안녕하세요."}],
            "voice": "default",
            "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "sample_rate": 24000,
            "channels": 1,
            "sample_width": 2,
            "format": "pcm_s16le",
        },
        headers={"x-user-id": "u1", "x-request-id": "r-tts-selected-model"},
    )

    assert response.status_code == 200
    assert response.headers["x-tts-model"] == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert local_engine.pcm_requests[-1]["model"] == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def test_internal_audio_speech_models_returns_selectable_qwen_tts_models(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JARVIS_TTS_SERVER_URL", "http://tts-runtime.local")
    client = TestClient(create_app())
    response = client.get(
        JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH_MODELS.path,
        headers={"x-user-id": "u1", "x-request-id": "r-tts-models"},
    )

    assert response.status_code == 200
    payload = response.json()
    model_ids = [item["id"] for item in payload["models"]]
    assert model_ids[0] == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" in model_ids


def test_internal_audio_speech_models_returns_local_only_without_selectable_runtime(
    monkeypatch,
) -> None:
    monkeypatch.delenv("JARVIS_TTS_SERVER_URL", raising=False)
    monkeypatch.delenv("JARVIS_TTS_SERVER_PCM_ENDPOINT", raising=False)
    local_engine = RecordingLocalTTSEngine(model_name="local/test-tts")
    tts_service = TextToSpeechService(local_engine=local_engine)
    client = TestClient(create_app(tts_service=tts_service))
    response = client.get(
        JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH_MODELS.path,
        headers={"x-user-id": "u1", "x-request-id": "r-tts-models-local"},
    )

    assert response.status_code == 200
    assert response.json()["models"] == [
        {
            "id": "local/test-tts",
            "label": "local/test-tts",
            "provider": "local",
            "is_default": True,
        }
    ]


def test_internal_audio_speech_pcm_rejects_qwen_model_without_selectable_runtime(
    monkeypatch,
) -> None:
    monkeypatch.delenv("JARVIS_TTS_SERVER_URL", raising=False)
    monkeypatch.delenv("JARVIS_TTS_SERVER_PCM_ENDPOINT", raising=False)
    local_engine = RecordingLocalTTSEngine(model_name="local/test-tts")
    tts_service = TextToSpeechService(local_engine=local_engine)
    client = TestClient(create_app(tts_service=tts_service))
    response = client.post(
        JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH_PCM.path,
        json={
            "chunks": [{"id": "c-1770000000000", "text": "안녕하세요."}],
            "voice": "default",
            "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "sample_rate": 24000,
            "channels": 1,
            "sample_width": 2,
            "format": "pcm_s16le",
        },
        headers={"x-user-id": "u1", "x-request-id": "r-tts-no-runtime"},
    )

    assert response.status_code == 503
    assert "requires a model-selectable TTS runtime" in response.json()["detail"]
    assert local_engine.pcm_requests == []


def test_internal_audio_speech_pcm_falls_back_to_local_tts(
    monkeypatch,
) -> None:
    monkeypatch.delenv("JARVIS_TTS_SERVER_URL", raising=False)
    monkeypatch.delenv("JARVIS_TTS_SERVER_PCM_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    local_engine = RecordingLocalTTSEngine()
    tts_service = TextToSpeechService(local_engine=local_engine)
    client = TestClient(create_app(tts_service=tts_service))
    response = client.post(
        JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH_PCM.path,
        json={
            "chunks": [{"id": "c-1770000000000", "text": "안녕하세요."}],
            "voice": "default",
            "model": "gpt-4o-mini-tts",
            "sample_rate": 24000,
            "channels": 1,
            "sample_width": 2,
            "format": "pcm_s16le",
        },
        headers={"x-user-id": "u1", "x-request-id": "r-tts-pcm"},
    )

    assert response.status_code == 200
    assert response.content == b"local-pcm"
    assert response.headers["content-type"] == "audio/pcm"
    assert response.headers["x-tts-provider"] == "local"
    assert response.headers["x-tts-model"] == "test-local-tts"
    assert local_engine.pcm_requests == [
        {
            "text": "안녕하세요.",
            "voice": "default",
            "sample_rate": 24000,
            "channels": 1,
            "sample_width": 2,
            "model": "test-local-tts",
        }
    ]


def test_default_prompt_loader_reads_workbench_prompt() -> None:
    prompt = load_prompt("base_system")

    assert prompt is not None
    assert "진행하겠습니다!" in prompt


def test_runtime_profile_preserves_application_routing_metadata() -> None:
    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name))
        with TestClient(app) as client:
            response = client.put(
                JarvisCoreEndpoints.INTERNAL_CLIENT_RUNTIME_PROFILE.path,
                json={
                    "platform": "macos",
                    "applications": [
                        {
                            "name": "Weather",
                            "display_name": "Weather",
                            "aliases": ["Weather", "weather", "날씨"],
                            "bundle_id": "com.apple.weather",
                            "executable": "Weather",
                            "kind": "macos_app",
                            "capabilities": ["weather", "forecast", "날씨", "예보"],
                            "categories": ["weather"],
                            "keywords": ["오늘 날씨", "지역 날씨"],
                        }
                    ],
                },
                headers={"x-user-id": "u-runtime"},
            )

            assert response.status_code == 200
            app_profile = response.json()["applications"][0]
            assert app_profile["capabilities"] == ["weather", "forecast", "날씨", "예보"]
            assert app_profile["categories"] == ["weather"]
            assert app_profile["keywords"] == ["오늘 날씨", "지역 날씨"]


def test_runtime_profile_preserves_terminal_policy_metadata() -> None:
    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name))
        with TestClient(app) as client:
            response = client.put(
                JarvisCoreEndpoints.INTERNAL_CLIENT_RUNTIME_PROFILE.path,
                json={
                    "platform": "macos",
                    "capabilities": ["terminal.run"],
                    "terminal": {
                        "enabled": True,
                        "shell": "zsh",
                        "cwd": "/Users/chawonje/Desktop/Workspace/project/JARVIS",
                        "allowed_commands": ["echo", "pwd", "ls", "git status"],
                        "allowed_cwds": [
                            "/Users/chawonje/Desktop/Workspace/project/JARVIS"
                        ],
                        "timeout_seconds": 20,
                    },
                },
                headers={"x-user-id": "u-runtime-terminal"},
            )

            assert response.status_code == 200
            terminal = response.json()["terminal"]
            assert terminal["enabled"] is True
            assert terminal["allowed_commands"] == ["echo", "pwd", "ls", "git status"]
            assert terminal["allowed_cwds"] == [
                "/Users/chawonje/Desktop/Workspace/project/JARVIS"
            ]


def test_internal_todo_crud_keeps_calendar_fields_as_references() -> None:
    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name))
        headers = {"x-user-id": "u-todo", "x-user-email": "todo@example.com"}
        with TestClient(app) as client:
            created_response = client.post(
                JarvisCoreEndpoints.INTERNAL_TODOS.path,
                json={
                    "title": "캘린더 연동 준비",
                    "description": "나중에 일정과 연결할 할 일",
                    "priority": 2,
                    "due_at": "2026-05-20T09:00:00+09:00",
                    "timezone": "Asia/Seoul",
                    "calendar_provider": "apple",
                    "calendar_id": "work",
                    "calendar_event_id": "evt-1",
                    "metadata": {"source": "test"},
                },
                headers=headers,
            )

            assert created_response.status_code == 200
            created = created_response.json()
            assert created["status"] == "open"
            assert created["calendar_sync_status"] == "linked"
            assert created["calendar_event_id"] == "evt-1"
            assert created["metadata"] == {"source": "test"}

            invalid_response = client.post(
                JarvisCoreEndpoints.INTERNAL_TODOS.path,
                json={"title": "시간 형식 오류", "due_at": "not-a-date"},
                headers=headers,
            )
            assert invalid_response.status_code == 422

            list_response = client.get(
                JarvisCoreEndpoints.INTERNAL_TODOS_LIST.path,
                headers=headers,
            )
            assert list_response.status_code == 200
            assert [item["id"] for item in list_response.json()["items"]] == [
                created["id"]
            ]

            update_response = client.patch(
                JarvisCoreEndpoints.INTERNAL_TODO_UPDATE.path.format(
                    todo_id=created["id"]
                ),
                json={"status": "completed"},
                headers=headers,
            )
            assert update_response.status_code == 200
            updated = update_response.json()
            assert updated["status"] == "completed"
            assert updated["completed_at"] is not None
            assert updated["calendar_event_id"] == "evt-1"

            delete_response = client.delete(
                JarvisCoreEndpoints.INTERNAL_TODO_DELETE.path.format(
                    todo_id=created["id"]
                ),
                headers=headers,
            )
            assert delete_response.status_code == 200
            assert delete_response.json() == {"id": created["id"], "deleted": True}

            empty_response = client.get(
                JarvisCoreEndpoints.INTERNAL_TODOS_LIST.path,
                headers=headers,
            )
            assert empty_response.status_code == 200
            assert empty_response.json()["items"] == []


def test_realtime_request_uses_workbench_base_prompt(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_OLLAMA_REALTIME_COMPACT_PROMPT", "0")
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            response = client.post(
                "/internal/chat/request",
                json={"message": "브라우저 열어줘", "route_override": "realtime"},
                headers={
                    "x-user-id": "u-prompt",
                    "x-user-email": "u-prompt@example.com",
                    "x-request-id": "r-prompt",
                },
            )

    assert response.status_code == 200
    system_prompt = str(ai_client.requests[-1]["system_prompt"])
    assert "진행하겠습니다!" in system_prompt
    assert "Sound like a real person in a messenger chat" in system_prompt
    assert "answer directly in-character first" in system_prompt


def test_ollama_realtime_stream_uses_workbench_realtime_prompt(monkeypatch) -> None:
    def fake_prompt_load(key: str) -> str | None:
        assert key == "realtime_system"
        return "Workbench realtime prompt. 짧게 답해."

    monkeypatch.setattr("application.chat.service._load_prompt", fake_prompt_load)
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/internal/chat/stream",
                json={"message": "안녕?", "route_override": "realtime"},
                headers={
                    "x-user-id": "u-compact-prompt",
                    "x-user-email": "u-compact-prompt@example.com",
                    "x-request-id": "r-compact-prompt",
                },
            ) as response:
                body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "assistant_delta" in body
    system_prompt = str(ai_client.requests[-1]["system_prompt"])
    assert system_prompt.startswith("Workbench realtime prompt. 짧게 답해.")
    assert "실시간 규칙" in system_prompt
    assert "Current-turn priority" not in system_prompt


def test_ollama_realtime_prompt_reflects_yaml_updates(monkeypatch, tmp_path) -> None:
    prompts_path = tmp_path / "prompts.yaml"
    monkeypatch.setenv("JARVIS_PROMPTS_YAML", str(prompts_path))
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            for content in ("첫 번째 realtime prompt", "두 번째 realtime prompt"):
                prompts_path.write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "prompts": {
                                "realtime_system": {
                                    "name": "Realtime",
                                    "description": "test",
                                    "content": content,
                                }
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                with client.stream(
                    "POST",
                    "/internal/chat/stream",
                    json={"message": "안녕?", "route_override": "realtime"},
                    headers={
                        "x-user-id": "u-live-prompt",
                        "x-user-email": "u-live-prompt@example.com",
                        "x-request-id": f"r-live-prompt-{len(ai_client.requests)}",
                    },
                ) as response:
                    body = "".join(response.iter_text())
                assert response.status_code == 200
                assert "assistant_delta" in body
                system_prompt = str(ai_client.requests[-1]["system_prompt"])
                assert system_prompt.startswith(content)
                assert "실시간 규칙" in system_prompt


def test_ollama_realtime_prompt_preserves_long_yaml_content(monkeypatch, tmp_path) -> None:
    long_prompt = "긴 realtime prompt\n" + ("보존해야 하는 시스템 규칙입니다.\n" * 2500)
    prompts_path = tmp_path / "prompts.yaml"
    prompts_path.write_text(
        json.dumps(
            {
                "version": 1,
                "prompts": {
                    "realtime_system": {
                        "name": "Realtime",
                        "description": "long prompt test",
                        "content": long_prompt,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("JARVIS_PROMPTS_YAML", str(prompts_path))
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/internal/chat/stream",
                json={"message": "안녕?", "route_override": "realtime"},
                headers={
                    "x-user-id": "u-long-prompt",
                    "x-user-email": "u-long-prompt@example.com",
                    "x-request-id": "r-long-prompt",
                },
            ) as response:
                body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "assistant_delta" in body
    system_prompt = str(ai_client.requests[-1]["system_prompt"])
    assert system_prompt.startswith(long_prompt.strip())
    assert "실시간 규칙" in system_prompt


def test_realtime_stream_marks_length_limited_response() -> None:
    ai_client = LengthLimitedAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/internal/chat/stream",
                json={"message": "자세히 설명해줘", "route_override": "realtime"},
                headers={
                    "x-user-id": "u-length-limit",
                    "x-user-email": "u-length-limit@example.com",
                    "x-request-id": "r-length-limit",
                },
            ) as response:
                body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "assistant_truncated" in body
    assert '"done_reason": "length"' in body
    assert '"truncated": true' in body


def test_realtime_stream_marks_latest_user_message_as_current_turn() -> None:
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            for index, message in enumerate(
                (
                    "점심 메뉴 추천해줘",
                    "sublimetext켜서 안녕하세요 작성해볼래?",
                )
            ):
                with client.stream(
                    "POST",
                    "/internal/chat/stream",
                    json={"message": message, "route_override": "realtime"},
                    headers={
                        "x-user-id": "u-latest-turn",
                        "x-user-email": "u-latest-turn@example.com",
                        "x-request-id": f"r-latest-{index}",
                    },
                ) as response:
                    body = "".join(response.iter_text())
                assert response.status_code == 200
                assert "assistant_delta" in body

    messages = ai_client.requests[-1]["messages"]
    latest = messages[-1]
    assert latest["role"] == "user"
    assert latest["content"].startswith("Latest user message.")
    assert "sublimetext켜서 안녕하세요 작성해볼래?" in latest["content"]
    assert "점심 메뉴 추천" not in latest["content"]
    assert "Current-turn priority" in str(ai_client.requests[-1]["system_prompt"])


def test_latest_user_message_does_not_create_consecutive_user_turns() -> None:
    messages = _build_alternating_messages(
        system_prompt="system",
        context_messages=[
            {"role": "user", "content": "브라우저 열어서 네이버 웹툰 페이지 열어줘"},
            {"role": "user", "content": "안녕?"},
        ],
        user_message="안녕?",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[-1]["content"].startswith("Latest user message.")
    assert "브라우저 열어서 네이버 웹툰 페이지 열어줘" not in messages[-1]["content"]


def test_ollama_realtime_uses_compact_current_turn_messages() -> None:
    system_prompt, messages = _build_messages_for_model(
        system_prompt="long workbench system prompt",
        context_messages=[
            {"role": "user", "content": "이전 질문"},
            {"role": "assistant", "content": "이전 답변"},
        ],
        user_message="안녕?",
        route="realtime",
        selected_model={"provider_name": "ollama", "model_name": "gemma4:e2b"},
    )

    assert system_prompt is not None
    assert len(system_prompt) < 200
    assert "JARVIS" in system_prompt
    assert messages == [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Latest user message. Respond to this message, not to the previous "
                "topic, unless this message explicitly refers back to it:\n안녕?"
            ),
        },
    ]


def test_ollama_realtime_followup_keeps_recent_context() -> None:
    _system_prompt, messages = _build_messages_for_model(
        system_prompt="long workbench system prompt",
        context_messages=[
            {"role": "user", "content": "내 강아지 이름은 콩이야"},
            {"role": "assistant", "content": "기억해둘게."},
        ],
        user_message="그거 다시 말해줘",
        route="realtime",
        selected_model={"provider_name": "ollama", "model_name": "gemma4:e2b"},
    )

    contents = "\n".join(str(message["content"]) for message in messages)
    assert "내 강아지 이름은 콩이야" in contents
    assert messages[-1]["content"].startswith("Latest user message.")
    assert "그거 다시 말해줘" in messages[-1]["content"]


def test_realtime_stream_uses_existing_user_persona_without_chat_selection(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JARVIS_OLLAMA_REALTIME_COMPACT_PROMPT", "0")
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        db = connect(db_file.name)
        app = create_app(db=db, ai_service=ai_service)
        with TestClient(app) as client:
            persona_response = client.post(
                "/internal/chat/persona",
                json={
                    "name": "Casual Friend",
                    "description": "친근한 친구 톤",
                    "prompt_template": "친구처럼 짧고 자연스럽게 말해.",
                    "tone": "casual",
                    "alias": "friend",
                },
                headers={"x-user-id": "u-persona-fallback"},
            )
            assert persona_response.status_code == 200
            db.conn.execute(
                """
                UPDATE chats
                SET last_selected_user_persona_id = NULL
                WHERE user_id = ?
                """,
                ("u-persona-fallback",),
            )
            db.conn.commit()

            with client.stream(
                "POST",
                "/internal/chat/stream",
                json={"message": "안녕?", "route_override": "realtime"},
                headers={
                    "x-user-id": "u-persona-fallback",
                    "x-user-email": "u-persona-fallback@example.com",
                    "x-request-id": "r-persona-fallback",
                },
            ) as response:
                body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "assistant_delta" in body
    system_prompt = str(ai_client.requests[-1]["system_prompt"])
    assert "Realtime persona override" in system_prompt
    assert "친구처럼 짧고 자연스럽게 말해." in system_prompt
    assert "Persona tone: casual" in system_prompt


def test_ollama_realtime_compact_prompt_includes_selected_persona(monkeypatch) -> None:
    monkeypatch.delenv("JARVIS_OLLAMA_REALTIME_COMPACT_PROMPT", raising=False)
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            persona_response = client.post(
                "/internal/chat/persona",
                json={
                    "name": "Casual Friend",
                    "description": "친근한 친구 톤",
                    "prompt_template": "친구처럼 짧고 자연스럽게 말해.",
                    "tone": "casual",
                    "alias": "friend",
                },
                headers={"x-user-id": "u-compact-persona"},
            )
            assert persona_response.status_code == 200
            select_response = client.post(
                "/internal/chat/persona/select",
                json={"user_persona_id": persona_response.json()["user_persona_id"]},
                headers={"x-user-id": "u-compact-persona"},
            )
            assert select_response.status_code == 200
            assert select_response.json()["is_selected"] is True

            with client.stream(
                "POST",
                "/internal/chat/stream",
                json={"message": "안녕?", "route_override": "realtime"},
                headers={
                    "x-user-id": "u-compact-persona",
                    "x-user-email": "u-compact-persona@example.com",
                    "x-request-id": "r-compact-persona",
                },
            ) as response:
                body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "assistant_delta" in body
    system_prompt = str(ai_client.requests[-1]["system_prompt"])
    assert "Realtime persona override" in system_prompt
    assert "친구처럼 짧고 자연스럽게 말해." in system_prompt
    assert "Persona tone: casual" in system_prompt
    assert ai_client.requests[-1]["messages"] == [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "안녕?"},
    ]


def test_deep_chat_does_not_apply_user_persona(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_OLLAMA_REALTIME_COMPACT_PROMPT", "0")
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            persona_response = client.post(
                "/internal/chat/persona",
                json={
                    "name": "Only Realtime",
                    "description": "deep에는 적용하지 않는다",
                    "prompt_template": "반드시 장난스럽게 말해.",
                    "tone": "playful",
                    "alias": "realtime-only",
                },
                headers={"x-user-id": "u-deep-no-persona"},
            )
            assert persona_response.status_code == 200
            select_response = client.post(
                "/internal/chat/persona/select",
                json={"user_persona_id": persona_response.json()["user_persona_id"]},
                headers={"x-user-id": "u-deep-no-persona"},
            )
            assert select_response.status_code == 200

            response = client.post(
                "/internal/chat/request",
                json={"message": "분석해줘", "route_override": "deep"},
                headers={
                    "x-user-id": "u-deep-no-persona",
                    "x-user-email": "u-deep-no-persona@example.com",
                    "x-request-id": "r-deep-no-persona",
                },
            )

    assert response.status_code == 200
    system_prompt = str(ai_client.requests[-1]["system_prompt"])
    assert "반드시 장난스럽게 말해." not in system_prompt
    assert "Persona tone: playful" not in system_prompt
    assert "Realtime persona override" not in system_prompt


def test_realtime_stream_prefers_selected_realtime_model_over_default(caplog) -> None:
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        caplog.set_level(logging.INFO, logger="jarvis_core.chat")
        with TestClient(app) as client:
            default_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "default-model",
                    "is_default": True,
                    "supports_stream": True,
                    "supports_realtime": True,
                },
                headers={"x-user-id": "u-model"},
            )
            assert default_response.status_code == 200

            realtime_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "selected-realtime-model",
                    "is_default": False,
                    "supports_stream": True,
                    "supports_realtime": True,
                },
                headers={"x-user-id": "u-model"},
            )
            assert realtime_response.status_code == 200
            realtime_id = realtime_response.json()["id"]

            selection_response = client.post(
                "/internal/chat/model-selection",
                json={"realtime_model_config_id": realtime_id},
                headers={"x-user-id": "u-model"},
            )
            assert selection_response.status_code == 200

            with client.stream(
                "POST",
                "/internal/chat/stream",
                json={"message": "바로 답해줘", "route_override": "realtime"},
                headers={
                    "x-user-id": "u-model",
                    "x-user-email": "u-model@example.com",
                    "x-request-id": "r-model",
                },
            ) as response:
                body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "selected-realtime-model" in body
    assert ai_client.requests[-1]["model_name"] == "selected-realtime-model"
    assert "model request request_id=r-model" in caplog.text
    assert "model response request_id=r-model" in caplog.text
    assert "model=selected-realtime-model" in caplog.text


def test_default_model_update_preserves_realtime_selection() -> None:
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            default_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "default-non-realtime",
                    "is_default": True,
                    "supports_stream": False,
                    "supports_realtime": False,
                },
                headers={"x-user-id": "u-preserve-selection"},
            )
            assert default_response.status_code == 200
            default_id = default_response.json()["id"]

            realtime_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "selected-realtime",
                    "is_default": False,
                    "supports_stream": True,
                    "supports_realtime": True,
                },
                headers={"x-user-id": "u-preserve-selection"},
            )
            assert realtime_response.status_code == 200
            realtime_id = realtime_response.json()["id"]

            selection_response = client.post(
                "/internal/chat/model-selection",
                json={"realtime_model_config_id": realtime_id},
                headers={"x-user-id": "u-preserve-selection"},
            )
            assert selection_response.status_code == 200

            updated_default = client.put(
                f"/internal/chat/model-config/{default_id}",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "default-non-realtime",
                    "is_default": True,
                    "supports_stream": False,
                    "supports_realtime": False,
                },
                headers={"x-user-id": "u-preserve-selection"},
            )
            assert updated_default.status_code == 200

            current_selection = client.get(
                "/internal/chat/model-selection",
                headers={"x-user-id": "u-preserve-selection"},
            )

    assert current_selection.status_code == 200
    assert current_selection.json()["realtime_model_config_id"] == realtime_id
    assert current_selection.json()["deep_model_config_id"] == default_id


def test_model_selection_partial_update_preserves_other_slot() -> None:
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            realtime_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "realtime-model",
                    "supports_stream": True,
                    "supports_realtime": True,
                },
                headers={"x-user-id": "u-partial-selection"},
            )
            deep_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "deep-model",
                    "supports_stream": True,
                    "supports_realtime": False,
                },
                headers={"x-user-id": "u-partial-selection"},
            )
            other_realtime_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "other-realtime-model",
                    "supports_stream": True,
                    "supports_realtime": True,
                },
                headers={"x-user-id": "u-partial-selection"},
            )
            realtime_id = realtime_response.json()["id"]
            deep_id = deep_response.json()["id"]
            other_realtime_id = other_realtime_response.json()["id"]

            selection_response = client.post(
                "/internal/chat/model-selection",
                json={
                    "realtime_model_config_id": realtime_id,
                    "deep_model_config_id": deep_id,
                },
                headers={"x-user-id": "u-partial-selection"},
            )
            assert selection_response.status_code == 200

            partial_response = client.post(
                "/internal/chat/model-selection",
                json={"realtime_model_config_id": other_realtime_id},
                headers={"x-user-id": "u-partial-selection"},
            )

    assert partial_response.status_code == 200
    assert partial_response.json()["realtime_model_config_id"] == other_realtime_id
    assert partial_response.json()["deep_model_config_id"] == deep_id


def test_realtime_selection_rejects_non_realtime_model() -> None:
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            model_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "default-non-realtime",
                    "is_default": True,
                    "supports_stream": False,
                    "supports_realtime": False,
                },
                headers={"x-user-id": "u-reject-selection"},
            )
            assert model_response.status_code == 200

            selection_response = client.post(
                "/internal/chat/model-selection",
                json={"realtime_model_config_id": model_response.json()["id"]},
                headers={"x-user-id": "u-reject-selection"},
            )

    assert selection_response.status_code == 400
    assert selection_response.json()["detail"] == "invalid realtime_model_config_id"


def test_delete_model_config_removes_user_model() -> None:
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            created = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "delete-me",
                    "is_default": False,
                    "supports_stream": True,
                    "supports_realtime": False,
                },
                headers={"x-user-id": "u-delete-model"},
            )
            assert created.status_code == 200
            model_id = created.json()["id"]

            deleted = client.delete(
                f"/internal/chat/model-config/{model_id}",
                headers={"x-user-id": "u-delete-model"},
            )
            listed = client.get(
                "/internal/chat/model-config",
                headers={"x-user-id": "u-delete-model"},
            )

    assert deleted.status_code == 200
    assert deleted.json() == {"id": model_id, "deleted": True}
    assert listed.status_code == 200
    assert listed.json() == []


def test_realtime_stream_prefers_realtime_capable_model_without_selection() -> None:
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            default_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "default-non-realtime-model",
                    "is_default": True,
                    "supports_stream": True,
                    "supports_realtime": False,
                },
                headers={"x-user-id": "u-realtime-auto"},
            )
            assert default_response.status_code == 200

            realtime_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "realtime-enabled-model",
                    "is_default": False,
                    "supports_stream": True,
                    "supports_realtime": True,
                },
                headers={"x-user-id": "u-realtime-auto"},
            )
            assert realtime_response.status_code == 200

            with client.stream(
                "POST",
                "/internal/chat/stream",
                json={"message": "바로 답해줘", "route_override": "realtime"},
                headers={
                    "x-user-id": "u-realtime-auto",
                    "x-user-email": "u-realtime-auto@example.com",
                    "x-request-id": "r-realtime-auto",
                },
            ) as response:
                body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "realtime-enabled-model" in body
    assert ai_client.requests[-1]["model_name"] == "realtime-enabled-model"


def test_vision_describe_calls_local_model_and_returns_description(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_describe_image(image_base64, *, prompt=None, model=None, timeout_seconds=None):
        calls.append({"image_base64": image_base64, "prompt": prompt})
        return "A blue sky with a game HUD showing 100 HP.", "qwen2.5vl:3b"

    monkeypatch.setattr("app.describe_image", fake_describe_image)

    with NamedTemporaryFile(suffix=".db") as db_file:
        with TestClient(create_app(db=connect(db_file.name))) as client:
            response = client.post(
                "/internal/vision/describe",
                json={"image_base64": "ZmFrZS1qcGVn"},
                headers={"x-user-id": "u-vision"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["description"] == "A blue sky with a game HUD showing 100 HP."
    assert body["model"] == "qwen2.5vl:3b"
    assert calls[0]["image_base64"] == "ZmFrZS1qcGVn"


def test_vision_describe_maps_model_error_to_502(monkeypatch) -> None:
    from ai.vision import VisionDescribeError

    async def failing_describe_image(image_base64, *, prompt=None, model=None, timeout_seconds=None):
        raise VisionDescribeError("vision model network error: connection refused")

    monkeypatch.setattr("app.describe_image", failing_describe_image)

    with NamedTemporaryFile(suffix=".db") as db_file:
        with TestClient(create_app(db=connect(db_file.name))) as client:
            response = client.post(
                "/internal/vision/describe",
                json={"image_base64": "ZmFrZS1qcGVn"},
                headers={"x-user-id": "u-vision"},
            )

    assert response.status_code == 502
    assert "connection refused" in response.json()["detail"]


def test_internal_persona_and_memory_endpoints() -> None:
    with TestClient(create_app()) as client:
        persona_response = client.post(
            "/internal/chat/persona",
            json={
                "name": "Architect",
                "description": "Focus on design",
                "prompt_template": "You are Jarvis acting as a careful architect.",
                "tone": "analytical",
                "alias": "architect",
            },
            headers={"x-user-id": "u1"},
        )
        assert persona_response.status_code == 200
        persona = persona_response.json()
        assert persona["name"] == "Architect"

        memory_response = client.post(
            "/internal/chat/memory",
            json={
                "type": "preference",
                "content": "Prefer concise Korean answers.",
                "importance": 5,
            },
            headers={"x-user-id": "u1"},
        )
        assert memory_response.status_code == 200
        memory = memory_response.json()
        assert memory["type"] == "preference"

        list_response = client.get("/internal/chat/persona", headers={"x-user-id": "u1"})
        assert list_response.status_code == 200
        assert len(list_response.json()) >= 1

        memory_list_response = client.get("/internal/chat/memory", headers={"x-user-id": "u1"})
        assert memory_list_response.status_code == 200
        assert len(memory_list_response.json()) >= 1


def test_deepthink_uses_memory_and_saves_user_message() -> None:
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            memory_response = client.post(
                "/internal/chat/memory",
                json={
                    "type": "preference",
                    "content": "항상 한국어로 짧게 답하기",
                    "importance": 5,
                },
                headers={"x-user-id": "u-memory"},
            )
            assert memory_response.status_code == 200

            plan_response = client.post(
                JarvisCoreEndpoints.INTERNAL_DEEPTHINK_PLAN.path,
                json={"request_id": "r-memory", "message": "내 선호 기억하고 있어?"},
                headers={"x-user-id": "u-memory"},
            )
            assert plan_response.status_code == 200

            assert ai_client.requests
            system_prompt = str(ai_client.requests[0]["system_prompt"])
            assert "Relevant memory" in system_prompt
            assert "항상 한국어로 짧게 답하기" in system_prompt

            db = client.app.state.db
            cursor = db.conn.execute(
                "SELECT role, content FROM messages WHERE role = ? AND content = ?",
                ("user", "내 선호 기억하고 있어?"),
            )
            assert cursor.fetchone() is not None


def test_deepthink_prefers_selected_deep_model_over_default() -> None:
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            default_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "default-model",
                    "is_default": True,
                    "supports_stream": True,
                    "supports_realtime": True,
                },
                headers={"x-user-id": "u-deep-selection"},
            )
            assert default_response.status_code == 200

            deep_response = client.post(
                "/internal/chat/model-config",
                json={
                    "provider_mode": "local",
                    "provider_name": "ollama",
                    "model_name": "selected-deep-model",
                    "is_default": False,
                    "supports_stream": True,
                    "supports_realtime": False,
                },
                headers={"x-user-id": "u-deep-selection"},
            )
            assert deep_response.status_code == 200
            deep_id = deep_response.json()["id"]

            selection_response = client.post(
                "/internal/chat/model-selection",
                json={"deep_model_config_id": deep_id},
                headers={"x-user-id": "u-deep-selection"},
            )
            assert selection_response.status_code == 200

            plan_response = client.post(
                JarvisCoreEndpoints.INTERNAL_DEEPTHINK_PLAN.path,
                json={"request_id": "r-deep-model", "message": "깊게 분석해줘"},
                headers={"x-user-id": "u-deep-selection"},
            )

    assert plan_response.status_code == 200
    assert ai_client.requests
    assert ai_client.requests[-1]["model_name"] == "selected-deep-model"


def test_deepthink_without_user_model_uses_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_FALLBACK_PROVIDER_NAME", "ollama")
    monkeypatch.setenv("JARVIS_FALLBACK_MODEL_NAME", "gemma4:e4b")
    monkeypatch.setenv(
        "JARVIS_FALLBACK_MODEL_ENDPOINT",
        "http://host.docker.internal:3030/chat",
    )
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            response = client.post(
                JarvisCoreEndpoints.INTERNAL_DEEPTHINK_PLAN.path,
                json={"request_id": "r-fallback", "message": "라우팅 설계해줘"},
                headers={"x-user-id": "u-no-model"},
            )

    assert response.status_code == 200
    assert ai_client.requests
    request = ai_client.requests[0]
    assert request["provider_name"] == "ollama"
    assert request["model_name"] == "gemma4:e4b"
    assert request["endpoint"] == "http://host.docker.internal:3030/chat"


def test_deep_chat_context_filters_provider_errors_and_internal_steps() -> None:
    ai_client = RecordingAIClient()
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        db = connect(db_file.name)
        app = create_app(db=db, ai_service=ai_service)
        with TestClient(app) as client:
            session = get_or_create_session_for_user(
                db,
                user_id="u-filter",
                email="filter@example.com",
            )
            add_message(
                db, session["id"], "assistant", "[provider-error] localhost:8080"
            )
            add_message(
                db,
                session["id"],
                "user",
                "[현재 단계: bad]\n원래 요청: 앱 실행\n이 단계의 목표: execute",
            )
            add_message(db, session["id"], "assistant", "정상 이전 응답")
            response = client.post(
                "/internal/chat/request",
                json={
                    "message": (
                        "앱 실행, 브라우저 검색, 터미널 실행 액션 라우팅 "
                        "우선순위를 설계해줘"
                    ),
                    "route_override": "deep",
                },
                headers={"x-user-id": "u-filter", "x-user-email": "filter@example.com"},
            )

    assert response.status_code == 200
    assert ai_client.requests
    request = ai_client.requests[-1]
    messages_text = "\n".join(
        str(message.get("content", "")) for message in request["messages"]
    )
    assert "provider-error" not in messages_text
    assert "[현재 단계:" not in messages_text
    assert "원래 요청:" not in messages_text
    assert "HIGH PRIORITY" in messages_text
    assert "not asking you to run an app" in messages_text
    assert "분석/설계 요청입니다" in messages_text


def test_deepthink_scroll_fallback_creates_browser_action() -> None:
    ai_client = RecordingAIClient()
    ai_client.force_provider_error = True
    ai_service = AIService(
        default_client=ai_client,
        local_client=ai_client,
        token_client=ai_client,
    )

    with NamedTemporaryFile(suffix=".db") as db_file:
        app = create_app(db=connect(db_file.name), ai_service=ai_service)
        with TestClient(app) as client:
            response = client.post(
                JarvisCoreEndpoints.INTERNAL_DEEPTHINK_EXECUTE.path,
                json={
                    "request_id": "r-scroll",
                    "message": "스크롤 내려줘",
                    "plan_steps": [
                        {
                            "id": "s1",
                            "title": "Initiate Scroll Down",
                            "description": "Execute scroll down in browser",
                        }
                    ],
                },
                headers={"x-user-id": "u-scroll"},
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["actions"][0]["type"] == "browser_control"
    assert payload["actions"][0]["command"] == "scroll"
    assert payload["actions"][0]["target"] == "active_tab"
    assert payload["actions"][0]["args"]["direction"] == "down"
