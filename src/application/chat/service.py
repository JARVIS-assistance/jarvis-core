from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import AsyncGenerator
from uuid import uuid4

logger = logging.getLogger("jarvis_core.chat")

from fastapi import HTTPException, status
from jarvis_contracts import ErrorResponse
from starlette.websockets import WebSocket

from ai import AIService

# ── 베이스 시스템 프롬프트 ─────────────────────────────────
# 페르소나보다 먼저 적용되는 JARVIS 핵심 정체성·규칙.
# 모든 대화(realtime, deep)에서 항상 포함된다.
# Workbench UI에서 수정 가능 (prompts.yaml → base_system 키).
from core.config.prompt_loader import load_prompt as _load_prompt
from core.db.db_connection import DBClient
from core.db.db_operations import (
    add_message,
    create_memory_item,
    create_user_model_config,
    create_user_persona,
    delete_user_model_config,
    ensure_default_persona_for_user,
    ensure_user_settings,
    get_active_model_for_user,
    get_latest_chat_summary,
    get_model_config_by_id_for_user,
    get_or_create_session_for_user,
    get_selected_persona_for_user,
    get_user_ai_selection,
    list_memory_items,
    list_recent_messages,
    list_user_model_configs,
    list_user_personas,
    select_user_persona,
    set_user_ai_selection,
    update_user_model_config,
    update_user_persona,
)
from core.db.db_operations.model_config import get_active_realtime_model_for_user
from router import choose_route
from safety import safety_gate

from .schemas import (
    ChatOnceRequest,
    MemoryCreateRequest,
    ModelConfigUpsertRequest,
    ModelSelectionUpsertRequest,
    PersonaSelectionRequest,
    PersonaUpsertRequest,
)

_BASE_SYSTEM_PROMPT_FALLBACK = """\
You are JARVIS — an intelligent AI assistant system.

## Core rules
1. Always respond in the same language the user uses. If the user writes in Korean, respond in Korean. If English, respond in English.
2. Be accurate. If you are unsure, say so honestly rather than guessing.
3. Be concise but thorough. Avoid unnecessary filler, but include all relevant details.
4. When the user asks you to do something (open apps, search, create files, etc.), respond exactly: "진행하겠습니다!".
5. Respect privacy. Never ask for sensitive personal information unless the user offers it.
6. If a request is ambiguous, ask one clarifying question before proceeding.
7. Do not emit ```actions blocks or JSON action blocks in normal chat responses. Client actions are created by the Controller action registry path.
8. Never invent action types. App launch is `type=app_control`, `command=open`, `target=<app name>`; never use `launch_app`.

## Capabilities
- General conversation and Q&A
- Web search for real-time information (weather, news, prices, etc.)
- Deep analysis and multi-step reasoning
- Remote PC control: terminal commands, app launch, file operations, mouse/keyboard, screenshots
- Planning and step-by-step task execution

## Response style
- Use markdown formatting when it improves readability (lists, bold, code blocks).
- For simple questions, answer directly without unnecessary structure.
- For complex tasks, break down into clear steps.
- If the message is an action request that the Controller should execute (for example opening a browser/app, searching, clicking, typing, running shell, checking the screen, or controlling the computer), respond exactly: "진행하겠습니다!"
- Do not explain that you cannot operate the screen for action requests. The Controller action intent lane runs in parallel and will attempt the action.
"""

_CONTEXT_EXCLUDE_MARKERS = (
    "[provider-error]",
    "provider-error",
    "[현재 단계:",
    "원래 요청:",
    "이 단계의 목표:",
    "client action result timed out",
)


def _get_base_system_prompt() -> str:
    """prompts.yaml에서 base_system을 읽고, 없으면 fallback을 반환한다."""
    loaded = _load_prompt("base_system")
    return loaded if loaded else _BASE_SYSTEM_PROMPT_FALLBACK


def _log_model_request(
    *,
    request_id: str,
    user_id: str,
    route: str,
    purpose: str,
    model: dict[str, object],
    stream: str,
) -> None:
    logger.info(
        "[chat] model request request_id=%s user=%s route=%s purpose=%s "
        "stream=%s provider=%s/%s model=%s config_id=%s",
        request_id,
        user_id,
        route,
        purpose,
        stream,
        model.get("provider_mode"),
        model.get("provider_name"),
        model.get("model_name"),
        model.get("id"),
    )


def _log_model_response(
    *,
    request_id: str,
    user_id: str,
    route: str,
    provider_mode: object,
    provider_name: object,
    model_name: object,
    stream: str,
    output_chars: int,
) -> None:
    logger.info(
        "[chat] model response request_id=%s user=%s route=%s stream=%s "
        "provider=%s/%s model=%s output_chars=%d",
        request_id,
        user_id,
        route,
        stream,
        provider_mode,
        provider_name,
        model_name,
        output_chars,
    )


def _build_alternating_messages(
    *,
    system_prompt: str | None,
    context_messages: list[dict[str, str]],
    user_message: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for message in context_messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not content:
            continue
        if len(messages) == 1 and messages[0]["role"] == "system" and role == "assistant":
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += f"\n\n{content}"
        else:
            messages.append({"role": role, "content": content})

    # Some local chat templates, including small Ollama models, can emit an empty
    # response when the message list contains consecutive user turns.
    while messages and messages[-1]["role"] == "user":
        messages.pop()

    messages.append({"role": "user", "content": _latest_user_message_content(user_message)})
    return messages


def _latest_user_message_content(user_message: str) -> str:
    return (
        "Latest user message. Respond to this message, not to the previous topic, "
        "unless this message explicitly refers back to it:\n"
        f"{user_message}"
    )


_OLLAMA_PROVIDER_NAMES = {
    "ollama",
    "ollama_chat",
    "ollama-generate",
    "ollma",
    "ollmahost",
}


_REALTIME_COMPACT_SYSTEM_PROMPT = (
    "너는 JARVIS. 한국어로 짧게 답해. "
    "컴퓨터 조작 요청은 '진행하겠습니다!'만 출력."
)

_REALTIME_RESPONSE_GUARD = (
    "실시간 규칙: 한국어 1-2문장, 80자 안팎. 긴 분석/계획/비교는 짧게 deep 전환만 말해."
)

_LENGTH_DONE_REASONS = {
    "length",
    "max_tokens",
    "max_token",
    "num_predict",
    "limit",
}


def _get_realtime_system_prompt() -> str:
    loaded = _load_prompt("realtime_system")
    base = loaded if loaded and loaded.strip() else _REALTIME_COMPACT_SYSTEM_PROMPT
    if _REALTIME_RESPONSE_GUARD in base:
        return base
    return f"{base.strip()}\n\n{_REALTIME_RESPONSE_GUARD}"


def _is_length_done_reason(done_reason: str | None) -> bool:
    if not done_reason:
        return False
    normalized = done_reason.strip().casefold()
    return normalized in _LENGTH_DONE_REASONS or any(
        marker in normalized for marker in _LENGTH_DONE_REASONS
    )


def _is_ollama_provider_name(provider_name: object) -> bool:
    normalized = str(provider_name or "").strip().lower().replace(" ", "_")
    return normalized in _OLLAMA_PROVIDER_NAMES


def _use_compact_realtime_messages(route: str, provider_name: object) -> bool:
    if route != "realtime" or not _is_ollama_provider_name(provider_name):
        return False
    raw = os.getenv("JARVIS_OLLAMA_REALTIME_COMPACT_PROMPT", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def _realtime_context_reference_requested(message: str) -> bool:
    folded = message.casefold()
    compact = re.sub(r"\s+", "", folded)
    context_terms = (
        "이전",
        "직전",
        "방금",
        "마지막",
        "아까",
        "앞에서",
        "위에서",
        "그거",
        "그걸",
        "그것",
        "그 질문",
        "그 답",
        "그 내용",
        "거기",
        "이어서",
        "계속",
        "더 설명",
        "자세히",
        "다시 말",
        "다시 알려",
        "previous",
        "last",
        "earlier",
        "above",
        "that",
        "it",
        "continue",
        "elaborate",
        "more detail",
    )
    compact_patterns = (
        "내가뭐라",
        "내가뭐라고",
        "뭐라고했",
        "뭐라했",
        "이어서",
        "계속해",
        "더알려",
        "더설명",
        "자세히설명",
    )
    return any(term in folded for term in context_terms) or any(
        pattern in compact for pattern in compact_patterns
    )


def _is_default_persona(persona: dict[str, object]) -> bool:
    return (
        str(persona.get("alias") or "").strip().lower() == "default"
        and str(persona.get("name") or "").strip() == "Default Persona"
    )


def _realtime_persona_prompt_parts(persona: dict[str, object] | None) -> list[str]:
    if not persona or _is_default_persona(persona):
        return []

    prompt_template = persona.get("prompt_template")
    parts: list[str] = []
    if isinstance(prompt_template, str) and prompt_template.strip():
        parts.append(
            "## Realtime persona override\n"
            "Follow this persona for voice, tone, wording, and conversational behavior. "
            "When base style and this persona differ, use this persona unless it conflicts "
            "with safety or client action protocol.\n"
            f"{prompt_template.strip()}"
        )

    tone = persona.get("tone")
    if isinstance(tone, str) and tone.strip():
        parts.append(f"Persona tone: {tone.strip()}")
    return parts


def _build_messages_for_model(
    *,
    system_prompt: str | None,
    context_messages: list[dict[str, str]],
    user_message: str,
    route: str,
    selected_model: dict[str, object],
) -> tuple[str | None, list[dict[str, str]]]:
    effective_user_message = user_message
    if route == "deep" and _looks_like_action_routing_analysis(user_message):
        effective_user_message = (
            "분석/설계 요청입니다. 앱 실행, 브라우저 검색, 터미널 실행을 실제로 "
            "수행하지 말고, 충돌하지 않는 라우팅 우선순위와 판단 기준을 답하세요.\n\n"
            f"{user_message}"
        )
    if _use_compact_realtime_messages(route, selected_model.get("provider_name")):
        realtime_prompt = (
            system_prompt
            if system_prompt and not context_messages
            else _get_realtime_system_prompt()
        )
        if context_messages and _realtime_context_reference_requested(
            effective_user_message
        ):
            return realtime_prompt, _build_alternating_messages(
                system_prompt=realtime_prompt,
                context_messages=context_messages,
                user_message=effective_user_message,
            )
        return realtime_prompt, [
            {"role": "system", "content": realtime_prompt},
            {
                "role": "user",
                "content": _latest_user_message_content(effective_user_message),
            },
        ]
    return system_prompt, _build_alternating_messages(
        system_prompt=system_prompt,
        context_messages=context_messages,
        user_message=effective_user_message,
    )


def _recent_context_limit(route: str) -> int:
    if route == "deep":
        return _int_env("JARVIS_DEEP_CONTEXT_RECENT_LIMIT", 12)
    return _int_env("JARVIS_REALTIME_CONTEXT_RECENT_LIMIT", 4)


def _context_content_allowed(content: str) -> bool:
    folded = content.casefold()
    return not any(marker.casefold() in folded for marker in _CONTEXT_EXCLUDE_MARKERS)


def _filter_context_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        message
        for message in messages
        if _context_content_allowed(str(message.get("content", "")))
    ]


def _looks_like_action_routing_analysis(message: str) -> bool:
    folded = message.casefold()
    design_terms = (
        "설계",
        "분석",
        "우선순위",
        "충돌",
        "비교",
        "라우팅",
        "design",
        "analyze",
        "priority",
        "conflict",
        "routing",
    )
    action_terms = (
        "액션",
        "앱 실행",
        "브라우저 검색",
        "터미널 실행",
        "action",
        "app",
        "browser",
        "terminal",
    )
    return any(term in folded for term in design_terms) and any(
        term in folded for term in action_terms
    )


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _realtime_context_cache_ttl_seconds() -> int:
    return _int_env("JARVIS_REALTIME_CONTEXT_CACHE_TTL_SECONDS", 20)


def _rollback_failed_transaction(db: DBClient) -> None:
    rollback = getattr(db.conn, "rollback", None)
    if not callable(rollback):
        return
    try:
        rollback()
    except Exception:
        logger.exception("[chat] failed to rollback db transaction")


class ChatService:
    def __init__(self, db: DBClient, ai_service: AIService) -> None:
        self.db = db
        self.ai_service = ai_service
        self._realtime_model_cache: dict[
            str, tuple[float, dict[str, str | bool | None]]
        ] = {}
        self._realtime_persona_parts_cache: dict[str, tuple[float, list[str]]] = {}

    def _cached_until(self) -> float:
        ttl = _realtime_context_cache_ttl_seconds()
        if ttl <= 0:
            return 0.0
        return time.monotonic() + ttl

    def _is_cache_entry_valid(self, expires_at: float) -> bool:
        return expires_at > time.monotonic()

    def _invalidate_realtime_cache(self, user_id: str) -> None:
        self._realtime_model_cache.pop(user_id, None)
        self._realtime_persona_parts_cache.pop(user_id, None)

    def _remember_model_config(
        self,
        *,
        user_id: str,
        purpose: str,
        model: dict[str, str | bool | None],
    ) -> dict[str, str | bool | None]:
        if purpose == "realtime":
            expires_at = self._cached_until()
            if expires_at:
                self._realtime_model_cache[user_id] = (expires_at, dict(model))
        return model

    def _cached_realtime_persona_parts(self, user_id: str) -> list[str]:
        cached = self._realtime_persona_parts_cache.get(user_id)
        if cached is not None:
            expires_at, parts = cached
            if self._is_cache_entry_valid(expires_at):
                return list(parts)

        persona = get_selected_persona_for_user(self.db, user_id=user_id)
        if persona is None:
            persona = ensure_default_persona_for_user(self.db, user_id=user_id)
        parts = _realtime_persona_prompt_parts(persona)
        expires_at = self._cached_until()
        if expires_at:
            self._realtime_persona_parts_cache[user_id] = (expires_at, list(parts))
        return parts

    def _compact_prompt_context_for_model(
        self,
        *,
        route: str,
        selected_model: dict[str, object],
        user_id: str | None = None,
    ) -> tuple[str | None, list[dict[str, str]]] | None:
        if not _use_compact_realtime_messages(
            route,
            selected_model.get("provider_name"),
        ):
            return None
        system_parts = [_get_realtime_system_prompt()]
        if user_id:
            try:
                system_parts.extend(self._cached_realtime_persona_parts(user_id))
            except Exception:
                logger.exception(
                    "[chat] failed to load compact realtime persona user=%s",
                    user_id,
                )
        realtime_prompt = "\n\n".join(system_parts)
        return realtime_prompt, [{"role": "system", "content": realtime_prompt}]

    def _fallback_model_config(self) -> dict[str, str | bool | None]:
        return {
            "id": "default-local",
            "provider_mode": "local",
            "provider_name": os.getenv("JARVIS_FALLBACK_PROVIDER_NAME", "ollama"),
            "model_name": os.getenv("JARVIS_FALLBACK_MODEL_NAME", "qwen2.5:1.5b"),
            "api_key": None,
            "endpoint": os.getenv(
                "JARVIS_FALLBACK_MODEL_ENDPOINT",
                "http://host.docker.internal:3030/chat",
            ),
            "is_active": True,
            "is_default": True,
            "supports_stream": True,
            "supports_realtime": True,
            "transport": "http_sse",
            "input_modalities": "text",
            "output_modalities": "text",
        }

    def _select_model_config(
        self, user_id: str, purpose: str
    ) -> dict[str, str | bool | None]:
        if purpose == "realtime":
            cached = self._realtime_model_cache.get(user_id)
            if cached is not None:
                expires_at, model = cached
                if self._is_cache_entry_valid(expires_at):
                    return dict(model)

        selected_id: str | None = None
        selection = get_user_ai_selection(self.db, user_id=user_id)
        if selection is not None:
            if purpose == "deep":
                selected_id = selection.get("deep_model_config_id")
            else:
                selected_id = selection.get("realtime_model_config_id")

        if selected_id:
            selected_model = get_model_config_by_id_for_user(
                self.db, user_id=user_id, model_config_id=selected_id
            )
            if selected_model is not None and bool(
                selected_model.get("is_active", True)
            ):
                if purpose != "realtime" or bool(
                    selected_model.get("supports_realtime", False)
                ):
                    logger.info(
                        "[chat] model selected purpose=%s provider=%s/%s model=%s config_id=%s user=%s",
                        purpose,
                        selected_model["provider_mode"],
                        selected_model["provider_name"],
                        selected_model["model_name"],
                        selected_model["id"],
                        user_id,
                    )
                    return self._remember_model_config(
                        user_id=user_id,
                        purpose=purpose,
                        model=selected_model,
                    )
                logger.info(
                    "[chat] selected realtime model ignored because supports_realtime=false "
                    "model=%s config_id=%s user=%s",
                    selected_model["model_name"],
                    selected_model["id"],
                    user_id,
                )

        if purpose == "realtime":
            realtime_config = get_active_realtime_model_for_user(
                self.db, user_id=user_id
            )
            if realtime_config is not None:
                logger.info(
                    "[chat] model selected via realtime support "
                    "provider=%s/%s model=%s config_id=%s user=%s",
                    realtime_config["provider_mode"],
                    realtime_config["provider_name"],
                    realtime_config["model_name"],
                    realtime_config["id"],
                    user_id,
                )
                return self._remember_model_config(
                    user_id=user_id,
                    purpose=purpose,
                    model=realtime_config,
                )

        default_config = get_active_model_for_user(self.db, user_id=user_id)
        if default_config is not None and bool(default_config.get("is_default", False)):
            logger.info(
                "[chat] model selected via default purpose=%s provider=%s/%s model=%s config_id=%s user=%s",
                purpose,
                default_config["provider_mode"],
                default_config["provider_name"],
                default_config["model_name"],
                default_config["id"],
                user_id,
            )
            return self._remember_model_config(
                user_id=user_id,
                purpose=purpose,
                model=default_config,
            )

        result = default_config or {**self._fallback_model_config()}
        logger.info(
            "[chat] model fallback purpose=%s provider=%s/%s model=%s user=%s",
            purpose,
            result.get("provider_mode"),
            result.get("provider_name"),
            result.get("model_name"),
            user_id,
        )
        return self._remember_model_config(
            user_id=user_id,
            purpose=purpose,
            model=result,
        )

    def _resolve_route(self, message: str, task_type: str, route_override: str | None) -> str:
        if route_override in {"realtime", "deep"}:
            return route_override
        return choose_route(message, task_type)

    def _build_prompt_context(
        self, *, user_id: str, chat_id: str, route: str, current_message: str = ""
    ) -> tuple[str | None, list[dict[str, str]]]:
        settings = ensure_user_settings(self.db, user_id=user_id)
        persona = None
        if route == "realtime":
            persona = get_selected_persona_for_user(self.db, user_id=user_id)
            if persona is None:
                persona = ensure_default_persona_for_user(self.db, user_id=user_id)

        memories = list_memory_items(self.db, user_id=user_id, chat_id=chat_id, limit=5)
        summary = get_latest_chat_summary(self.db, chat_id=chat_id)
        recent_messages = list_recent_messages(
            self.db,
            chat_id,
            limit=_recent_context_limit(route),
        )
        metadata = settings.get("metadata", {}) if isinstance(settings, dict) else {}
        persona_hint = metadata.get("persona_hint") if isinstance(metadata, dict) else None
        custom_instructions = metadata.get("custom_instructions") if isinstance(metadata, dict) else None
        memory_lines = [f"- ({item['type']}/{item['importance']}) {item['content']}" for item in memories]
        route_line = (
            "Use deeper analysis, surface assumptions, and be explicit about tradeoffs. "
            "If the latest message asks to design, analyze, compare, or prioritize "
            "action routing or capabilities, treat it as an analysis request, not an "
            "execution request; do not say '진행하겠습니다!' and do not claim actions ran."
            if route == "deep"
            else "Respond with low latency and concise, directly useful guidance."
        )
        system_parts = [
            _get_base_system_prompt(),
            f"Route mode: {route}",
            f"User locale: {settings['locale']}",
            f"User timezone: {settings['timezone']}",
            f"Preferred response style: {settings['response_style']}",
            route_line,
            "Current-turn priority: answer or execute the latest user message. "
            "Do not continue a previous topic unless the latest message clearly refers to it.",
        ]
        if route == "realtime":
            system_parts.extend(_realtime_persona_prompt_parts(persona))
        if persona_hint:
            system_parts.append(f"Persona hint from user settings: {persona_hint}")
        if custom_instructions:
            system_parts.append(f"Custom user instructions: {custom_instructions}")
        if memory_lines:
            system_parts.append("Relevant memory:\n" + "\n".join(memory_lines))
        if route == "deep" and _looks_like_action_routing_analysis(current_message):
            system_parts.append(
                "HIGH PRIORITY: The latest user message is asking for a design or "
                "analysis of action routing/capability priority. It is not asking "
                "you to run an app, browser search, or terminal command. Answer with "
                "the requested routing design. Do not say '진행하겠습니다!' and do "
                "not claim any action was performed."
            )
        if (
            summary is not None
            and summary.get("summary_text")
            and _context_content_allowed(str(summary["summary_text"]))
        ):
            system_parts.append(summary["summary_text"])
        system_prompt = "\n\n".join(system_parts)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(_filter_context_messages(recent_messages))
        return system_prompt, messages

    async def request_once(
        self, body: ChatOnceRequest, request_id: str, user_id: str, email: str
    ) -> dict[str, str] | ErrorResponse:
        allowed, reason = safety_gate(body.message, body.confirm)
        if not allowed:
            return ErrorResponse(
                error_code="SAFETY_BLOCKED",
                message=reason or "blocked",
                request_id=request_id,
            )

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing user id in auth token",
            )

        session = get_or_create_session_for_user(
            self.db, user_id=user_id, email=email or "unknown@local.jarvis"
        )
        session_id = session["id"]
        route = self._resolve_route(body.message, body.task_type, body.route_override)
        purpose = "deep" if route == "deep" else "realtime"
        selected = self._select_model_config(user_id=user_id, purpose=purpose)
        compact_context = (
            None
            if _realtime_context_reference_requested(body.message)
            else self._compact_prompt_context_for_model(
                route=route,
                selected_model=selected,
                user_id=user_id,
            )
        )
        if compact_context is None:
            system_prompt, prompt_messages = self._build_prompt_context(
                user_id=user_id,
                chat_id=session_id,
                route=route,
                current_message=body.message,
            )
        else:
            system_prompt, prompt_messages = compact_context
        add_message(self.db, session_id, "user", body.message)

        effective_system_prompt, effective_messages = _build_messages_for_model(
            system_prompt=system_prompt,
            context_messages=prompt_messages[1:],
            user_message=body.message,
            route=route,
            selected_model=selected,
        )
        _log_model_request(
            request_id=request_id,
            user_id=user_id,
            route=route,
            purpose=purpose,
            model=selected,
            stream="once",
        )
        ai_result = await self.ai_service.respond_once(
            {
                "message": body.message,
                "route": route,
                "request_id": request_id,
                "provider_mode": selected["provider_mode"],
                "provider_name": selected["provider_name"],
                "model_name": selected["model_name"],
                "api_key": selected.get("api_key"),
                "endpoint": selected.get("endpoint"),
                "system_prompt": effective_system_prompt,
                "messages": effective_messages,
            }
        )
        _log_model_response(
            request_id=request_id,
            user_id=user_id,
            route=route,
            provider_mode=ai_result["provider_mode"],
            provider_name=ai_result["provider_name"],
            model_name=ai_result["model_name"],
            stream="once",
            output_chars=len(str(ai_result["content"])),
        )
        add_message(self.db, session_id, "assistant", ai_result["content"])
        return {
            "request_id": request_id,
            "route": route,
            "provider_mode": ai_result["provider_mode"],
            "provider_name": ai_result["provider_name"],
            "model_name": ai_result["model_name"],
            "content": ai_result["content"],
        }

    # ── realtime 함수에서 쓸 함수들 ───────────────────────────────────────────

    @staticmethod
    def _parse_ws_event(raw: str) -> dict | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    @staticmethod
    async def _cancel_task(task: asyncio.Task[None] | None) -> None:
        """Cancel an asyncio task and suppress CancelledError."""
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _interrupt_generation(
        self,
        websocket: WebSocket,
        generation: asyncio.Task[None] | None,
        rt_session_id: str | None,
        request_payload: dict[str, object] | None,
        reason: str,
    ) -> None:
        """Cancel the active generation task and notify the client."""
        await self._cancel_task(generation)
        if rt_session_id and request_payload:
            await self.ai_service.cancel_generation(request_payload, rt_session_id)
        payload: dict[str, str] = {"type": "interrupted", "content": reason}
        if request_payload:
            rid = request_payload.get("request_id")
            if rid:
                payload["request_id"] = str(rid)
        await websocket.send_json(payload)

    async def _stream_generation(
        self,
        websocket: WebSocket,
        request_payload: dict[str, object],
        rt_session_id: str,
        request_id: str,
        chat_session_id: str,
    ) -> None:
        """Stream tokens from the AI service to the WebSocket client."""
        chunks: list[str] = []
        done_reason = "stop"
        try:
            async for event in self.ai_service.stream_token_events(request_payload):
                event_type = event.get("type")
                if event_type == "done":
                    done_reason = str(event.get("done_reason") or done_reason)
                    continue
                token = str(event.get("content") or "")
                if not token:
                    continue
                chunks.append(token)
                await websocket.send_json(
                    {
                        "type": "assistant_delta",
                        "request_id": request_id,
                        "content": token,
                    }
                )
        except asyncio.CancelledError:
            return
        except Exception:
            await websocket.send_json(
                {
                    "type": "error",
                    "request_id": request_id,
                    "content": "generation failed",
                }
            )
            return

        full = "".join(chunks).strip()
        if full:
            add_message(self.db, chat_session_id, "assistant", full)
        truncated = _is_length_done_reason(done_reason)
        _log_model_response(
            request_id=request_id,
            user_id=str(request_payload.get("user_id", "")),
            route=str(request_payload.get("route", "")),
            provider_mode=request_payload.get("provider_mode"),
            provider_name=request_payload.get("provider_name"),
            model_name=request_payload.get("model_name"),
            stream="websocket",
            output_chars=len(full),
        )
        if truncated:
            await websocket.send_json(
                {
                    "type": "assistant_truncated",
                    "request_id": request_id,
                    "done_reason": done_reason,
                    "content": "realtime response hit provider length limit",
                }
            )
        await websocket.send_json(
            {
                "type": "assistant_done",
                "request_id": request_id,
                "content": full,
                "done_reason": done_reason,
                "truncated": truncated,
            }
        )

    # ── run_realtime ────────────────────────────────────────────

    async def run_realtime(
        self, websocket: WebSocket, user_id: str, email: str
    ) -> None:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing user id in auth token",
            )

        await websocket.accept()
        await websocket.send_json({"type": "ready"})

        session = get_or_create_session_for_user(
            self.db, user_id=user_id, email=email or "unknown@local.jarvis"
        )
        chat_session_id = session["id"]
        active_generation: asyncio.Task[None] | None = None
        active_rt_session_id: str | None = None
        active_request: dict[str, object] | None = None
        realtime_session_request: dict[str, object] | None = None

        try:
            while True:
                raw = await websocket.receive_text()
                event = self._parse_ws_event(raw)
                if event is None:
                    await websocket.send_json(
                        {"type": "error", "content": "invalid json"}
                    )
                    continue

                event_type = str(event.get("type", ""))
                if event_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                if event_type == "interrupt":
                    await self._interrupt_generation(
                        websocket,
                        active_generation,
                        active_rt_session_id,
                        active_request,
                        reason="user_interrupt",
                    )
                    active_generation = None
                    continue

                if event_type != "user_message":
                    await websocket.send_json(
                        {"type": "error", "content": "unsupported event type"}
                    )
                    continue

                # Barge-in: new message interrupts in-flight generation
                if active_generation is not None and not active_generation.done():
                    await self._interrupt_generation(
                        websocket,
                        active_generation,
                        active_rt_session_id,
                        active_request,
                        reason="barge_in",
                    )
                    active_generation = None

                content = str(event.get("content", "")).strip()
                task_type = str(event.get("task_type", "general"))
                confirm = bool(event.get("confirm", False))
                if not content:
                    await websocket.send_json(
                        {"type": "error", "content": "empty content"}
                    )
                    continue

                allowed, reason = safety_gate(content, confirm)
                if not allowed:
                    await websocket.send_json(
                        {"type": "error", "content": reason or "blocked"}
                    )
                    continue

                selected = self._select_model_config(
                    user_id=user_id, purpose="realtime"
                )
                if not bool(selected.get("supports_realtime", False)):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "content": "selected model does not support realtime",
                        }
                    )
                    continue

                route = self._resolve_route(
                    content, task_type, event.get("route_override")
                )
                request_id = str(uuid4())
                compact_context = (
                    None
                    if _realtime_context_reference_requested(content)
                    else self._compact_prompt_context_for_model(
                        route=route,
                        selected_model=selected,
                        user_id=user_id,
                    )
                )
                if compact_context is None:
                    system_prompt, prompt_messages = self._build_prompt_context(
                        user_id=user_id,
                        chat_id=chat_session_id,
                        route=route,
                        current_message=content,
                    )
                else:
                    system_prompt, prompt_messages = compact_context
                add_message(self.db, chat_session_id, "user", content)
                effective_system_prompt, effective_messages = _build_messages_for_model(
                    system_prompt=system_prompt,
                    context_messages=prompt_messages[1:],
                    user_message=content,
                    route=route,
                    selected_model=selected,
                )
                request_payload: dict[str, object] = {
                    "message": content,
                    "route": route,
                    "request_id": request_id,
                    "provider_mode": str(selected["provider_mode"]),
                    "provider_name": str(selected["provider_name"]),
                    "model_name": str(selected["model_name"]),
                    "api_key": selected.get("api_key"),
                    "endpoint": selected.get("endpoint"),
                    "system_prompt": effective_system_prompt,
                    "messages": effective_messages,
                }
                request_payload["user_id"] = user_id
                _log_model_request(
                    request_id=request_id,
                    user_id=user_id,
                    route=route,
                    purpose="realtime",
                    model=selected,
                    stream="websocket",
                )
                active_request = request_payload

                if active_rt_session_id is None:
                    active_rt_session_id = await self.ai_service.realtime_session_start(
                        request_payload
                    )
                    realtime_session_request = request_payload

                await websocket.send_json(
                    {
                        "type": "meta",
                        "request_id": request_id,
                        "route": route,
                        "provider_mode": selected["provider_mode"],
                        "provider_name": selected["provider_name"],
                        "model_name": selected["model_name"],
                        "session_id": active_rt_session_id,
                    }
                )

                active_generation = asyncio.create_task(
                    self._stream_generation(
                        websocket,
                        request_payload,
                        active_rt_session_id or "",
                        request_id,
                        chat_session_id,
                    )
                )
        finally:
            await self._cancel_task(active_generation)
            if active_rt_session_id and realtime_session_request:
                await self.ai_service.realtime_session_close(
                    realtime_session_request, active_rt_session_id
                )

    async def run_realtime_sse(
        self,
        message: str,
        task_type: str,
        confirm: bool,
        route_override: str | None,
        request_id: str,
        user_id: str,
        email: str,
    ) -> AsyncGenerator[str, None]:
        """SSE 스트리밍: AI 제공자에서 토큰을 받아 SSE 이벤트로 yield."""
        logger.info(
            "[chat] stream start request_id=%s user=%s message=%s",
            request_id,
            user_id,
            message[:200],
        )
        setup_started = time.monotonic()
        if not user_id:
            yield f"event: error\ndata: {json.dumps({'content': 'missing user id'})}\n\n"
            return

        allowed, reason = safety_gate(message, confirm)
        if not allowed:
            yield f"event: error\ndata: {json.dumps({'content': reason or 'blocked'})}\n\n"
            return

        try:
            session = get_or_create_session_for_user(
                self.db, user_id=user_id, email=email or "unknown@local.jarvis"
            )
            session_id = session["id"]
            route = self._resolve_route(message, task_type, route_override)
            logger.info(
                "[chat] stream context start request_id=%s route=%s session=%s",
                request_id,
                route,
                session_id,
            )
            purpose = "deep" if route == "deep" else "realtime"
            selected = self._select_model_config(user_id=user_id, purpose=purpose)
            compact_context = (
                None
                if _realtime_context_reference_requested(message)
                else self._compact_prompt_context_for_model(
                    route=route,
                    selected_model=selected,
                    user_id=user_id,
                )
            )
            if compact_context is None:
                system_prompt, prompt_messages = self._build_prompt_context(
                    user_id=user_id,
                    chat_id=session_id,
                    route=route,
                    current_message=message,
                )
            else:
                system_prompt, prompt_messages = compact_context
            logger.info("[chat] stream context ready request_id=%s", request_id)
            add_message(self.db, session_id, "user", message)
        except Exception as exc:
            _rollback_failed_transaction(self.db)
            logger.exception("[chat] stream setup failed request_id=%s", request_id)
            yield f"event: error\ndata: {json.dumps({'request_id': request_id, 'content': f'stream setup failed: {exc}'})}\n\n"
            return

        if not bool(selected.get("supports_stream", False)):
            yield f"event: error\ndata: {json.dumps({'content': 'selected model does not support streaming'})}\n\n"
            return

        effective_system_prompt, effective_messages = _build_messages_for_model(
            system_prompt=system_prompt,
            context_messages=prompt_messages[1:],
            user_message=message,
            route=route,
            selected_model=selected,
        )
        request_payload = {
            "message": message,
            "route": route,
            "request_id": request_id,
            "user_id": user_id,
            "provider_mode": str(selected["provider_mode"]),
            "provider_name": str(selected["provider_name"]),
            "model_name": str(selected["model_name"]),
            "api_key": selected.get("api_key"),
            "endpoint": selected.get("endpoint"),
            "system_prompt": effective_system_prompt,
            "messages": effective_messages,
        }
        _log_model_request(
            request_id=request_id,
            user_id=user_id,
            route=route,
            purpose=purpose,
            model=selected,
            stream="sse",
        )

        # meta event
        meta = {
            "type": "meta",
            "request_id": request_id,
            "route": route,
            "provider_mode": selected["provider_mode"],
            "provider_name": selected["provider_name"],
            "model_name": selected["model_name"],
            "setup_ms": int((time.monotonic() - setup_started) * 1000),
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # stream tokens
        chunks: list[str] = []
        done_reason = "stop"
        try:
            async for event in self.ai_service.stream_token_events(request_payload):
                event_type = event.get("type")
                if event_type == "done":
                    done_reason = str(event.get("done_reason") or done_reason)
                    continue
                token = str(event.get("content") or "")
                if not token:
                    continue
                chunks.append(token)
                yield f"event: assistant_delta\ndata: {json.dumps({'request_id': request_id, 'content': token})}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'request_id': request_id, 'content': str(exc)})}\n\n"
            return

        full = "".join(chunks).strip()
        if full:
            add_message(self.db, session_id, "assistant", full)
        truncated = _is_length_done_reason(done_reason)
        _log_model_response(
            request_id=request_id,
            user_id=user_id,
            route=route,
            provider_mode=selected["provider_mode"],
            provider_name=selected["provider_name"],
            model_name=selected["model_name"],
            stream="sse",
            output_chars=len(full),
        )
        if truncated:
            yield (
                "event: assistant_truncated\n"
                f"data: {json.dumps({'request_id': request_id, 'done_reason': done_reason, 'content': 'realtime response hit provider length limit'})}\n\n"
            )

        yield (
            "event: assistant_done\n"
            f"data: {json.dumps({'request_id': request_id, 'content': full, 'done_reason': done_reason, 'truncated': truncated})}\n\n"
        )

    async def close_realtime_session(
        self, request: dict[str, object], realtime_session_id: str
    ) -> None:
        await self.ai_service.realtime_session_close(request, realtime_session_id)

    def create_model_config(
        self, user_id: str, body: ModelConfigUpsertRequest
    ) -> dict[str, str | bool | None]:
        result = create_user_model_config(
            self.db,
            user_id=user_id,
            provider_mode=body.provider_mode,
            provider_name=body.provider_name,
            model_name=body.model_name,
            api_key=body.api_key,
            endpoint=body.endpoint,
            is_default=body.is_default,
            supports_stream=body.supports_stream,
            supports_realtime=body.supports_realtime,
            transport=body.transport,
            input_modalities=body.input_modalities,
            output_modalities=body.output_modalities,
        )
        if body.is_default:
            self._set_default_model_selection(user_id=user_id, default_model=result)
        self._invalidate_realtime_cache(user_id)
        return result

    def list_model_configs(self, user_id: str) -> list[dict[str, str | bool | None]]:
        return list_user_model_configs(self.db, user_id=user_id)

    def update_model_config(
        self,
        user_id: str,
        model_config_id: str,
        body: ModelConfigUpsertRequest,
    ) -> dict[str, str | bool | None]:
        result = update_user_model_config(
            self.db,
            user_id=user_id,
            model_config_id=model_config_id,
            provider_mode=body.provider_mode,
            provider_name=body.provider_name,
            model_name=body.model_name,
            api_key=body.api_key,
            endpoint=body.endpoint,
            is_default=body.is_default,
            supports_stream=body.supports_stream,
            supports_realtime=body.supports_realtime,
            transport=body.transport,
            input_modalities=body.input_modalities,
            output_modalities=body.output_modalities,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="model config not found")
        if body.is_default:
            self._set_default_model_selection(user_id=user_id, default_model=result)
        self._invalidate_realtime_cache(user_id)
        return result

    def _set_default_model_selection(
        self,
        *,
        user_id: str,
        default_model: dict[str, str | bool | None],
    ) -> dict[str, str | None]:
        selection = get_user_ai_selection(self.db, user_id=user_id) or {}
        realtime_model_config_id = selection.get("realtime_model_config_id")
        if realtime_model_config_id:
            realtime = get_model_config_by_id_for_user(
                self.db,
                user_id=user_id,
                model_config_id=realtime_model_config_id,
            )
            if (
                realtime is None
                or not bool(realtime.get("is_active", True))
                or not bool(realtime.get("supports_realtime", False))
            ):
                realtime_model_config_id = None

        if realtime_model_config_id is None and bool(
            default_model.get("supports_realtime", False)
        ):
            realtime_model_config_id = str(default_model["id"])

        return set_user_ai_selection(
            self.db,
            user_id=user_id,
            realtime_model_config_id=realtime_model_config_id,
            deep_model_config_id=str(default_model["id"]),
        )

    def delete_model_config(self, user_id: str, model_config_id: str) -> dict[str, bool | str]:
        deleted = delete_user_model_config(
            self.db,
            user_id=user_id,
            model_config_id=model_config_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="model config not found")
        self._invalidate_realtime_cache(user_id)
        return {"id": model_config_id, "deleted": True}

    def set_model_selection(
        self, user_id: str, body: ModelSelectionUpsertRequest
    ) -> dict[str, str | None]:
        existing = get_user_ai_selection(self.db, user_id=user_id) or {}
        fields_set = getattr(body, "model_fields_set", set())
        realtime_model_config_id = (
            body.realtime_model_config_id
            if "realtime_model_config_id" in fields_set
            else existing.get("realtime_model_config_id")
        )
        deep_model_config_id = (
            body.deep_model_config_id
            if "deep_model_config_id" in fields_set
            else existing.get("deep_model_config_id")
        )

        if realtime_model_config_id is not None:
            realtime = get_model_config_by_id_for_user(
                self.db, user_id=user_id, model_config_id=realtime_model_config_id
            )
            if realtime is None or not bool(realtime.get("supports_realtime", False)):
                raise HTTPException(
                    status_code=400, detail="invalid realtime_model_config_id"
                )
        if deep_model_config_id is not None:
            deep = get_model_config_by_id_for_user(
                self.db, user_id=user_id, model_config_id=deep_model_config_id
            )
            if deep is None:
                raise HTTPException(
                    status_code=400, detail="invalid deep_model_config_id"
                )

        result = set_user_ai_selection(
            self.db,
            user_id=user_id,
            realtime_model_config_id=realtime_model_config_id,
            deep_model_config_id=deep_model_config_id,
        )
        self._invalidate_realtime_cache(user_id)
        return result

    def get_model_selection(self, user_id: str) -> dict[str, str | None]:
        selection = get_user_ai_selection(self.db, user_id=user_id)
        if selection is not None:
            return selection
        return {
            "realtime_model_config_id": None,
            "deep_model_config_id": None,
        }

    def list_personas(self, user_id: str) -> list[dict[str, object]]:
        ensure_default_persona_for_user(self.db, user_id=user_id)
        return list_user_personas(self.db, user_id=user_id)

    def create_persona(
        self, user_id: str, body: PersonaUpsertRequest
    ) -> dict[str, object]:
        result = create_user_persona(
            self.db,
            user_id=user_id,
            name=body.name,
            description=body.description,
            prompt_template=body.prompt_template,
            tone=body.tone,
            alias=body.alias,
        )
        self._invalidate_realtime_cache(user_id)
        return result

    def update_persona(
        self, user_id: str, user_persona_id: str, body: PersonaUpsertRequest
    ) -> dict[str, object]:
        result = update_user_persona(
            self.db,
            user_id=user_id,
            user_persona_id=user_persona_id,
            name=body.name,
            description=body.description,
            prompt_template=body.prompt_template,
            tone=body.tone,
            alias=body.alias,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="persona not found")
        self._invalidate_realtime_cache(user_id)
        return result

    def select_persona(
        self, user_id: str, body: PersonaSelectionRequest
    ) -> dict[str, object]:
        result = select_user_persona(
            self.db, user_id=user_id, user_persona_id=body.user_persona_id
        )
        if result is None:
            raise HTTPException(status_code=404, detail="persona not found")
        self._invalidate_realtime_cache(user_id)
        return result

    def list_memory(self, user_id: str, chat_id: str | None = None) -> list[dict[str, object]]:
        return list_memory_items(self.db, user_id=user_id, chat_id=chat_id, limit=50)

    def create_memory(
        self, user_id: str, body: MemoryCreateRequest
    ) -> dict[str, object]:
        return create_memory_item(
            self.db,
            user_id=user_id,
            chat_id=body.chat_id,
            memory_type=body.type,
            content=body.content,
            importance=body.importance,
            source_message_id=body.source_message_id,
            expires_at=body.expires_at,
        )
