from __future__ import annotations

import logging

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from jarvis_contracts import (
    ClientAction,
    DeepThinkPlanRequest,
    DeepThinkPlanResponse,
    DeepThinkRequest,
    DeepThinkResponse,
    DeepThinkStepPayload,
    DeepThinkStepResult,
    InternalConversationRequest,
    InternalConversationResponse,
    JarvisCoreEndpoints,
    TodoCreateRequest,
    TodoListResponse,
    TodoResponse,
    TodoUpdateRequest,
)

from ai import AIService
from ai.client import StubAIClient
from application.audio.schemas import TextToSpeechPCMRequest, TextToSpeechRequest
from application.audio.service import TextToSpeechError, TextToSpeechService
from application.chat.schemas import (
    ChatOnceRequest,
    ChatOnceResponse,
    MemoryCreateRequest,
    MemoryResponse,
    ModelConfigResponse,
    ModelConfigUpsertRequest,
    ModelSelectionResponse,
    ModelSelectionUpsertRequest,
    PersonaResponse,
    PersonaSelectionRequest,
    PersonaUpsertRequest,
    RuntimeProfileResponse,
    RuntimeProfileUpsertRequest,
)
from application.chat.service import ChatService
from application.deepthink import DeepThinkService
from application.deepthink.schemas import (
    DeepThinkInternalRequest,
    DeepThinkPlanInternalRequest,
    DeepThinkStepInput,
)
from core.db.db_connection import DBClient, connect
from core.db.db_operations import (
    create_todo_item,
    delete_todo_item,
    ensure_user_exists,
    get_runtime_profile,
    get_todo_item,
    list_todo_items,
    set_runtime_profile,
    update_todo_item,
)
from core.db.db_schema import init_db
from jarvis_core import available_modes, run_deep_thinking, run_realtime_conversation
from middleware import RequestIDMiddleware

logging.basicConfig(level=logging.INFO)


def _get_chat_service(app: FastAPI) -> ChatService:
    service = getattr(app.state, "chat_service", None)
    if isinstance(service, ChatService):
        return service
    if not hasattr(app.state, "db") or app.state.db is None:
        app.state.db = connect()
        init_db(app.state.db)
    if not hasattr(app.state, "ai_service") or app.state.ai_service is None:
        app.state.ai_service = AIService(default_client=StubAIClient())
    service = ChatService(db=app.state.db, ai_service=app.state.ai_service)
    app.state.chat_service = service
    return service


def _get_deepthink_service(app: FastAPI) -> DeepThinkService:
    return DeepThinkService(db=app.state.db, ai_service=app.state.ai_service)


def _get_tts_service(app: FastAPI):
    service = getattr(app.state, "tts_service", None)
    if service is not None:
        return service
    service = TextToSpeechService()
    app.state.tts_service = service
    return service


def create_app(
    db: DBClient | None = None,
    ai_service: AIService | None = None,
    tts_service: TextToSpeechService | None = None,
) -> FastAPI:
    app = FastAPI(title="jarvis-core", version="0.3.0")
    app.add_middleware(RequestIDMiddleware)
    if ai_service is not None:
        app.state.ai_service = ai_service
    if tts_service is not None:
        app.state.tts_service = tts_service

    @app.on_event("startup")
    def startup() -> None:
        if not hasattr(app.state, "db") or app.state.db is None:
            app.state.db = db or connect()
            init_db(app.state.db)
        if not hasattr(app.state, "ai_service") or app.state.ai_service is None:
            app.state.ai_service = ai_service or AIService(default_client=StubAIClient())
        if not hasattr(app.state, "tts_service") or app.state.tts_service is None:
            app.state.tts_service = tts_service or TextToSpeechService()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        ai = getattr(app.state, "ai_service", None)
        close = getattr(ai, "close", None)
        if callable(close):
            await close()
        db_client: DBClient | None = getattr(app.state, "db", None)
        if db_client is not None:
            db_client.conn.close()

    # ── health ──────────────────────────────────────────────

    @app.get(JarvisCoreEndpoints.HEALTH.path)
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "jarvis-core",
            "mode": "library-first",
            "capabilities": list(available_modes()),
        }

    # ── internal: conversation (기존) ───────────────────────

    @app.post(
        JarvisCoreEndpoints.INTERNAL_CONVERSATION_RESPOND.path,
        response_model=InternalConversationResponse,
    )
    def respond(body: InternalConversationRequest) -> InternalConversationResponse:
        result = (
            run_deep_thinking(body.message)
            if body.mode == "deep"
            else run_realtime_conversation(body.message)
        )
        return InternalConversationResponse(
            mode=result.mode,
            summary=result.summary,
            content=result.content,
            next_actions=result.next_actions,
        )

    # ── internal: chat ──────────────────────────────────────

    @app.post("/internal/chat/request", response_model=ChatOnceResponse)
    async def chat_request(
        body: ChatOnceRequest,
        x_user_id: str = Header(...),
        x_user_email: str = Header(default=""),
        x_request_id: str = Header(default=""),
    ) -> ChatOnceResponse:
        service = _get_chat_service(app)
        result = await service.request_once(
            body=body,
            request_id=x_request_id,
            user_id=x_user_id,
            email=x_user_email,
        )
        from jarvis_contracts import ErrorResponse

        if isinstance(result, ErrorResponse):
            return ChatOnceResponse(
                request_id=result.request_id or x_request_id,
                route="blocked",
                provider_mode="local",
                provider_name="safety",
                model_name="none",
                content=result.message or "blocked",
            )
        return ChatOnceResponse(**result)

    @app.post("/internal/chat/stream")
    async def chat_stream(
        body: ChatOnceRequest,
        x_user_id: str = Header(...),
        x_user_email: str = Header(default=""),
        x_request_id: str = Header(default=""),
    ) -> StreamingResponse:
        service = _get_chat_service(app)
        return StreamingResponse(
            service.run_realtime_sse(
                message=body.message,
                task_type=body.task_type,
                confirm=body.confirm,
                route_override=body.route_override,
                request_id=x_request_id,
                user_id=x_user_id,
                email=x_user_email,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── internal: model config ──────────────────────────────

    @app.post("/internal/chat/model-config", response_model=ModelConfigResponse)
    def create_model_config(
        body: ModelConfigUpsertRequest,
        x_user_id: str = Header(...),
    ) -> ModelConfigResponse:
        service = _get_chat_service(app)
        result = service.create_model_config(user_id=x_user_id, body=body)
        return ModelConfigResponse(**result)

    @app.get("/internal/chat/model-config", response_model=list[ModelConfigResponse])
    def list_model_configs(
        x_user_id: str = Header(...),
    ) -> list[ModelConfigResponse]:
        service = _get_chat_service(app)
        result = service.list_model_configs(user_id=x_user_id)
        return [ModelConfigResponse(**item) for item in result]

    @app.put("/internal/chat/model-config/{model_config_id}", response_model=ModelConfigResponse)
    def update_model_config(
        model_config_id: str,
        body: ModelConfigUpsertRequest,
        x_user_id: str = Header(...),
    ) -> ModelConfigResponse:
        service = _get_chat_service(app)
        result = service.update_model_config(
            user_id=x_user_id,
            model_config_id=model_config_id,
            body=body,
        )
        return ModelConfigResponse(**result)

    @app.delete("/internal/chat/model-config/{model_config_id}")
    def delete_model_config(
        model_config_id: str,
        x_user_id: str = Header(...),
    ) -> dict[str, bool | str]:
        service = _get_chat_service(app)
        return service.delete_model_config(
            user_id=x_user_id,
            model_config_id=model_config_id,
        )

    @app.post("/internal/chat/model-selection", response_model=ModelSelectionResponse)
    def set_model_selection(
        body: ModelSelectionUpsertRequest,
        x_user_id: str = Header(...),
    ) -> ModelSelectionResponse:
        service = _get_chat_service(app)
        result = service.set_model_selection(user_id=x_user_id, body=body)
        return ModelSelectionResponse(**result)

    @app.get("/internal/chat/model-selection", response_model=ModelSelectionResponse)
    def get_model_selection(
        x_user_id: str = Header(...),
    ) -> ModelSelectionResponse:
        service = _get_chat_service(app)
        result = service.get_model_selection(user_id=x_user_id)
        return ModelSelectionResponse(**result)

    @app.post("/internal/chat/persona", response_model=PersonaResponse)
    def create_persona(
        body: PersonaUpsertRequest,
        x_user_id: str = Header(...),
    ) -> PersonaResponse:
        service = _get_chat_service(app)
        result = service.create_persona(user_id=x_user_id, body=body)
        return PersonaResponse(**result)

    @app.get("/internal/chat/persona", response_model=list[PersonaResponse])
    def list_personas(
        x_user_id: str = Header(...),
    ) -> list[PersonaResponse]:
        service = _get_chat_service(app)
        result = service.list_personas(user_id=x_user_id)
        return [PersonaResponse(**item) for item in result]

    @app.put("/internal/chat/persona/{user_persona_id}", response_model=PersonaResponse)
    def update_persona(
        user_persona_id: str,
        body: PersonaUpsertRequest,
        x_user_id: str = Header(...),
    ) -> PersonaResponse:
        service = _get_chat_service(app)
        result = service.update_persona(
            user_id=x_user_id,
            user_persona_id=user_persona_id,
            body=body,
        )
        return PersonaResponse(**result)

    @app.post("/internal/chat/persona/select", response_model=PersonaResponse)
    def select_persona(
        body: PersonaSelectionRequest,
        x_user_id: str = Header(...),
    ) -> PersonaResponse:
        service = _get_chat_service(app)
        result = service.select_persona(user_id=x_user_id, body=body)
        return PersonaResponse(**result)

    @app.post("/internal/chat/memory", response_model=MemoryResponse)
    def create_memory(
        body: MemoryCreateRequest,
        x_user_id: str = Header(...),
    ) -> MemoryResponse:
        service = _get_chat_service(app)
        result = service.create_memory(user_id=x_user_id, body=body)
        return MemoryResponse(**result)

    @app.get("/internal/chat/memory", response_model=list[MemoryResponse])
    def list_memory(
        x_user_id: str = Header(...),
        chat_id: str | None = None,
    ) -> list[MemoryResponse]:
        service = _get_chat_service(app)
        result = service.list_memory(user_id=x_user_id, chat_id=chat_id)
        return [MemoryResponse(**item) for item in result]

    # ── internal: audio ─────────────────────────────────────

    @app.post(JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH.path)
    async def synthesize_speech(
        body: TextToSpeechRequest,
        x_user_id: str = Header(...),
        x_request_id: str = Header(default=""),
    ) -> Response:
        _ = x_user_id, x_request_id
        service = _get_tts_service(app)
        try:
            result = await service.synthesize(body)
        except TextToSpeechError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        return Response(
            content=result.audio,
            media_type=result.media_type,
            headers={
                "X-TTS-Provider": result.provider,
                "X-TTS-Model": result.model,
                "X-TTS-Voice": result.voice,
                "X-TTS-Format": result.response_format,
                "X-AI-Generated-Voice": "true",
            },
        )

    @app.post(JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH_PCM.path)
    async def synthesize_speech_pcm(
        body: TextToSpeechPCMRequest,
        x_user_id: str = Header(...),
        x_request_id: str = Header(default=""),
    ) -> StreamingResponse:
        _ = x_user_id, x_request_id
        service = _get_tts_service(app)
        try:
            if hasattr(service, "ensure_server_pcm_available"):
                service.ensure_server_pcm_available()
            headers = service.server_pcm_headers(body)
            stream = service.stream_server_pcm(body)
            first_chunk = await anext(stream)
        except StopAsyncIteration as exc:
            raise HTTPException(status_code=503, detail="tts produced no pcm audio") from exc
        except TextToSpeechError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        async def stream_with_first_chunk():
            yield first_chunk
            async for chunk in stream:
                yield chunk

        return StreamingResponse(
            stream_with_first_chunk(),
            media_type="audio/pcm",
            headers=headers,
        )

    @app.get(JarvisCoreEndpoints.INTERNAL_AUDIO_SPEECH_MODELS.path)
    def list_speech_models(
        x_user_id: str = Header(...),
        x_request_id: str = Header(default=""),
    ) -> dict[str, object]:
        _ = x_user_id, x_request_id
        service = _get_tts_service(app)
        return {"models": service.list_models()}

    # ── internal: client runtime profile ──────────────────────

    @app.put(
        JarvisCoreEndpoints.INTERNAL_CLIENT_RUNTIME_PROFILE.path,
        response_model=RuntimeProfileResponse,
    )
    def upsert_runtime_profile(
        body: RuntimeProfileUpsertRequest,
        x_user_id: str = Header(...),
    ) -> RuntimeProfileResponse:
        result = set_runtime_profile(
            app.state.db,
            user_id=x_user_id,
            profile=body.model_dump(),
        )
        return RuntimeProfileResponse(**result)

    @app.get(
        JarvisCoreEndpoints.INTERNAL_CLIENT_RUNTIME_PROFILE_GET.path,
        response_model=RuntimeProfileResponse,
    )
    def read_runtime_profile(
        x_user_id: str = Header(...),
    ) -> RuntimeProfileResponse:
        result = get_runtime_profile(app.state.db, user_id=x_user_id)
        return RuntimeProfileResponse(**result)

    # ── internal: todos ─────────────────────────────────────

    @app.post(
        JarvisCoreEndpoints.INTERNAL_TODOS.path,
        response_model=TodoResponse,
    )
    def create_todo(
        body: TodoCreateRequest,
        x_user_id: str = Header(...),
        x_user_email: str = Header(default=""),
    ) -> TodoResponse:
        ensure_user_exists(
            app.state.db,
            user_id=x_user_id,
            email=x_user_email or f"{x_user_id}@local.jarvis",
        )
        result = create_todo_item(
            app.state.db,
            user_id=x_user_id,
            payload=body.model_dump(mode="json"),
        )
        return TodoResponse(**result)

    @app.get(
        JarvisCoreEndpoints.INTERNAL_TODOS_LIST.path,
        response_model=TodoListResponse,
    )
    def list_todos(
        x_user_id: str = Header(...),
        status: str | None = None,
        include_deleted: bool = False,
        limit: int = 50,
    ) -> TodoListResponse:
        items = list_todo_items(
            app.state.db,
            user_id=x_user_id,
            status=status,
            include_deleted=include_deleted,
            limit=limit,
        )
        return TodoListResponse(items=[TodoResponse(**item) for item in items])

    @app.get(
        JarvisCoreEndpoints.INTERNAL_TODO_DETAIL.path,
        response_model=TodoResponse,
    )
    def read_todo(todo_id: str, x_user_id: str = Header(...)) -> TodoResponse:
        result = get_todo_item(app.state.db, user_id=x_user_id, todo_id=todo_id)
        if result is None:
            raise HTTPException(status_code=404, detail="todo not found")
        return TodoResponse(**result)

    @app.patch(
        JarvisCoreEndpoints.INTERNAL_TODO_UPDATE.path,
        response_model=TodoResponse,
    )
    def patch_todo(
        todo_id: str,
        body: TodoUpdateRequest,
        x_user_id: str = Header(...),
    ) -> TodoResponse:
        result = update_todo_item(
            app.state.db,
            user_id=x_user_id,
            todo_id=todo_id,
            updates=body.model_dump(exclude_unset=True, mode="json"),
        )
        if result is None:
            raise HTTPException(status_code=404, detail="todo not found")
        return TodoResponse(**result)

    @app.delete(JarvisCoreEndpoints.INTERNAL_TODO_DELETE.path)
    def remove_todo(todo_id: str, x_user_id: str = Header(...)) -> dict[str, object]:
        deleted = delete_todo_item(app.state.db, user_id=x_user_id, todo_id=todo_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="todo not found")
        return {"id": todo_id, "deleted": True}

    # ── internal: deepthink ────────────────────────────────

    @app.post(
        JarvisCoreEndpoints.INTERNAL_DEEPTHINK_PLAN.path,
        response_model=DeepThinkPlanResponse,
    )
    async def deepthink_plan(
        body: DeepThinkPlanRequest,
        x_user_id: str = Header(...),
        x_request_id: str = Header(default=""),
    ) -> DeepThinkPlanResponse:
        service = _get_deepthink_service(app)
        internal_req = DeepThinkPlanInternalRequest(
            request_id=body.request_id,
            message=body.message,
        )
        result = await service.plan(internal_req, user_id=x_user_id)
        return DeepThinkPlanResponse(
            request_id=result.request_id,
            goal=result.goal,
            steps=[
                DeepThinkStepPayload(
                    id=s.id,
                    title=s.title,
                    description=s.description,
                )
                for s in result.steps
            ],
            constraints=result.constraints,
        )

    @app.post(
        JarvisCoreEndpoints.INTERNAL_DEEPTHINK_EXECUTE.path,
        response_model=DeepThinkResponse,
    )
    async def deepthink_execute(
        body: DeepThinkRequest,
        x_user_id: str = Header(...),
        x_request_id: str = Header(default=""),
    ) -> DeepThinkResponse:
        service = _get_deepthink_service(app)
        internal_req = DeepThinkInternalRequest(
            request_id=body.request_id,
            message=body.message,
            plan_steps=[
                DeepThinkStepInput(
                    id=step.id,
                    title=step.title,
                    description=step.description,
                )
                for step in body.plan_steps
            ],
            execution_context=body.execution_context,
        )
        result = await service.execute(internal_req, user_id=x_user_id)

        def _to_client_action(a) -> ClientAction:
            return ClientAction(
                type=a.type,
                command=a.command,
                target=a.target,
                payload=a.payload,
                args=a.args,
                description=a.description,
                requires_confirm=a.requires_confirm,
                step_id=a.step_id,
            )

        return DeepThinkResponse(
            request_id=result.request_id,
            steps=[
                DeepThinkStepResult(
                    step_id=s.step_id,
                    title=s.title,
                    status=s.status,
                    content=s.content,
                    actions=[_to_client_action(a) for a in s.actions],
                )
                for s in result.steps
            ],
            summary=result.summary,
            content=result.content,
            actions=[_to_client_action(a) for a in result.actions],
        )

    return app


app = create_app()
