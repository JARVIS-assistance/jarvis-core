from __future__ import annotations

from typing import Literal, TypedDict


class AIMessage(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class AIStreamChunk(TypedDict, total=False):
    type: Literal["meta", "token", "done"]
    route: str
    request_id: str
    content: str


class AITokenStreamEvent(TypedDict, total=False):
    type: Literal["token", "done"]
    content: str
    done_reason: str


class AIRequest(TypedDict):
    message: str
    route: str
    request_id: str
    provider_mode: Literal["token", "local"]
    provider_name: str
    model_name: str
    api_key: str | None
    endpoint: str | None
    system_prompt: str | None
    messages: list[AIMessage]


class AIResponse(TypedDict):
    provider_mode: Literal["token", "local"]
    provider_name: str
    model_name: str
    content: str


class RealtimeInboundEvent(TypedDict, total=False):
    type: Literal["user_message", "interrupt", "ping"]
    content: str
    task_type: Literal["general", "analysis", "execution"]
    confirm: bool


class RealtimeOutboundEvent(TypedDict, total=False):
    type: Literal[
        "ready",
        "assistant_delta",
        "assistant_done",
        "interrupted",
        "error",
        "meta",
        "pong",
    ]
    request_id: str
    content: str
    route: str
    provider_mode: Literal["token", "local"]
    provider_name: str
    model_name: str
    session_id: str
