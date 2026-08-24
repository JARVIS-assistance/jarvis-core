from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from jarvis_contracts import normalize_action_payload

from ai import AIService
from core.config.prompt_loader import load_prompt as _load_prompt
from core.db.db_connection import DBClient
from core.db.db_operations import (
    add_message,
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
)

from .action_executor import ActionExecutor, ExecutionResult
from .schemas import (
    ClientActionInternal,
    DeepThinkInternalRequest,
    DeepThinkInternalResponse,
    DeepThinkPlanInternalRequest,
    DeepThinkPlanInternalResponse,
    DeepThinkStepInput,
    DeepThinkStepOutput,
)

logger = logging.getLogger("jarvis_core.deepthink")

# ── AI system prompts ──────────────────────────────────────
# Workbench UI에서 수정 가능 (prompts.yaml).
# yaml에 값이 있으면 우선 사용, 없으면 아래 fallback 사용.

_PLANNING_SYSTEM_PROMPT_FALLBACK = """\
You are JARVIS deep-thinking planning engine.
You control a remote user PC and can perform both logical tasks and physical control.

Given a user request, produce a structured execution plan as JSON.

You MUST respond with ONLY a JSON object (no markdown fences, no extra text) in this format:
{
  "goal": "one sentence describing the objective",
  "constraints": ["constraint 1", "constraint 2"],
  "steps": [
    {"id": "s1", "title": "short title", "description": "what to do in this step"},
    {"id": "s2", "title": "short title", "description": "what to do in this step"}
  ]
}

Available capabilities for planning:
- Logical: terminal commands, file create/read/write, app launch, web search, URL open, clipboard, notify
- Physical: mouse click/drag, keyboard typing, hotkey combos, screenshot capture
- Search: web search to retrieve real-time information (weather, news, stock prices, general knowledge, etc.)

Planning rules:
- 1~6 steps, each step must be concrete and actionable
- Steps should be ordered by dependency
- Do not collapse a multi-operation request into one broad step. Use the fewest
  steps that preserve every distinct operation the user asked for.
- If a request opens/focuses an app and asks JARVIS to write/create/compose
  content there, keep that as separate ordered steps: open/focus the app,
  compose the final content, then type or paste the final content.
- Use web_search only when the user explicitly asks to search/browse/open a
  web page, or when the answer depends on real-time/local facts such as weather,
  news, prices, stock quotes, exchange rates, operating hours, or current
  availability.
- Recommendation, advice, brainstorming, menu suggestions, and casual follow-ups
  are answer-generation requests. Do not plan web_search or browser actions for
  them unless the user explicitly asks to search or open the browser.
- For explicit search/live-information requests:
  1. web_search step to retrieve the information
  2. notify step to present the summarized result to the user
  Keep it simple — usually 2 steps is enough.
- For physical control tasks (GUI interaction), plan a screenshot step FIRST to understand the screen layout, then action steps based on coordinates
- For logical tasks (file ops, terminal), no screenshot needed — execute directly
- Include constraints the user mentioned or that are implied
- Write in the same language as the user's request
"""

_EXECUTION_SYSTEM_PROMPT_FALLBACK = """\
You are JARVIS deep-thinking execution engine controlling a remote user PC.
You receive a plan step and must produce the concrete actions the client should perform.

Include actions in a JSON array fenced with ```actions ... ```.

## Action types

### Logical actions (no screen needed)
| type | command | target | payload | args |
|------|---------|--------|---------|------|
| terminal | shell command | null | null | {cwd, env, timeout} |
| app_control | app name to launch | null | null | {} |
| file_write | null | file path | full file content | {} |
| file_read | null | file path | null | {} |
| open_url | null | URL or file path | null | {} |
| browser_control | scroll/back/forward/reload/extract_dom/click_element/type_element/select_result | active_tab | optional text | browser args |
| web_search | search query | null | null | {max_results: 3} |
| clipboard | null | null | text to copy | {} |
| notify | null | null | null | {} |

### Physical control actions (screen interaction)
| type | command | target | payload | args |
|------|---------|--------|---------|------|
| screenshot | null | null | null | {region: [x,y,w,h] or null} |
| screen_stream | start / stop / describe | null | null | start:{fps?,quality?,max_width?}; describe:{prompt?} |
| mouse_click | null | null | null | {x, y, button: "left", clicks: 1} |
| mouse_drag | null | null | null | {start_x, start_y, end_x, end_y, button?} |
| mouse_move | null / position | null | null | {x, y, duration_seconds?} |
| mouse_scroll | null | null | null | {amount, direction, x?, y?} |
| keyboard_type | null | null | text to type | {enter: true} |
| hotkey | key combo e.g. "ctrl,c" | null | null | {duration_seconds? — hold the key, e.g. for movement} |

## Rules
1. **Logical tasks** (terminal, file, app, search): execute directly, no screenshot needed.
2. **Web page tasks**: if clicking or typing inside the current page is needed, emit `browser_control` command `extract_dom` first. Use `click_element` or `type_element` only after DOM results provide an `ai_id`.
3. **Physical tasks** (GUI click, type in app): a plain `screenshot` action returns a raw image you (a text-only model) cannot see — its result is not usable for coordinate analysis. To actually understand what is on screen, emit `screen_stream` command `start` once (if not already streaming), then `screen_stream` command `describe` — that result is a text description you CAN read and reason about. For a task that requires reacting to a changing screen over multiple turns (games, multi-step UI flows you can't fully plan up front), you may be invoked again for the same step after your actions run: emit only the `describe` + the next single action for now, don't try to solve the whole task in one response. When the goal is fully achieved, emit a `notify` action summarizing the result instead of further actions — that signals you are done.
4. **Search tasks**: emit `web_search` only when the current step explicitly requires web search or live facts. For recommendation, advice, brainstorming, menu suggestions, and casual follow-ups, answer directly without actions unless the user explicitly asks to search/browse/open a page. The server will execute the search and inject results into the next step's context automatically.
5. `requires_confirm`: true for destructive operations (delete, overwrite, install). false for reads, screenshots, notifications, searches.
6. `description`: human-readable explanation in the user's language.
7. Your analysis text should come BEFORE the ```actions``` block.
8. Respond in the same language as the user's request.
9. Use only canonical action types from the Client Action Registry. For app launch use `app_control` with `command=open` and `target=<app name>`. Never use `launch_app`.
10. If the user refers to a result/link/button on the current browser page, emit `browser_control/extract_dom` first. Do not emit `web_search` or open a new search page for current-page selection.

## Examples

Terminal command:
```actions
[{"type": "terminal", "command": "pip install requests", "target": null, "payload": null, "args": {"cwd": "/project"}, "description": "requests 패키지 설치", "requires_confirm": true}]
```

App launch + keyboard typing:
```actions
[
  {"type": "app_control", "command": "notepad", "target": null, "payload": null, "args": {}, "description": "메모장 실행", "requires_confirm": false},
  {"type": "keyboard_type", "command": null, "target": null, "payload": "Hello World", "args": {"enter": false}, "description": "메모장에 텍스트 입력", "requires_confirm": false}
]
```

Screen analysis needed:
```actions
[{"type": "screenshot", "command": null, "target": null, "payload": null, "args": {}, "description": "현재 화면 캡처하여 좌표 파악", "requires_confirm": false}]
```

Click at coordinate:
```actions
[{"type": "mouse_click", "command": null, "target": null, "payload": null, "args": {"x": 500, "y": 300, "button": "left", "clicks": 1}, "description": "확인 버튼 클릭", "requires_confirm": false}]
```

Web search for information:
```actions
[{"type": "web_search", "command": "서울 날씨 오늘", "target": null, "payload": null, "args": {"max_results": 3}, "description": "서울 오늘 날씨 검색", "requires_confirm": false}]
```

Browser scroll:
```actions
[{"type": "browser_control", "command": "scroll", "target": "active_tab", "payload": null, "args": {"direction": "down", "amount": "page"}, "description": "브라우저 스크롤", "requires_confirm": false}]
```

Select search result:
```actions
[{"type": "browser_control", "command": "select_result", "target": "active_tab", "payload": null, "args": {"index": 2}, "description": "검색 결과에서 두 번째 항목 선택", "requires_confirm": false}]
```

Extract DOM to open a link:
```actions
[{"type": "browser_control", "command": "extract_dom", "target": "active_tab", "payload": null, "args": {"purpose": "resolve_open_request", "query": "초간단 마카롱 만들기", "include_links": true, "include_elements": true, "max_links": 120}, "description": "현재 페이지에서 초간단 마카롱 만들기 링크 후보 추출", "requires_confirm": false}]
```

Type into a page field after DOM target is known:
```actions
[{"type": "browser_control", "command": "type_element", "target": "active_tab", "payload": "마카롱", "args": {"ai_id": 3, "enter": false}, "description": "검색 입력란에 마카롱 입력", "requires_confirm": false}]
```

Notify user with search results:
```actions
[{"type": "notify", "command": null, "target": null, "payload": "서울 현재 기온 18°C, 맑음, 미세먼지 보통", "args": {}, "description": "검색 결과를 사용자에게 알림", "requires_confirm": false}]
```

Understand the current screen before acting (vision-grounded loop):
```actions
[{"type": "screen_stream", "command": "describe", "target": null, "payload": null, "args": {}, "description": "현재 화면 상태 확인", "requires_confirm": false}]
```
"""

_SUMMARIZE_SYSTEM_PROMPT_FALLBACK = (
    "You are JARVIS. "
    "Summarize the search results concisely "
    "and answer the user's question directly. "
    "Respond in the same language as the user's request. "
    "Do NOT include action blocks."
)


def _get_planning_prompt() -> str:
    loaded = _load_prompt("deepthink_planning")
    return loaded if loaded else _PLANNING_SYSTEM_PROMPT_FALLBACK


def _get_execution_prompt() -> str:
    loaded = _load_prompt("deepthink_execution")
    return loaded if loaded else _EXECUTION_SYSTEM_PROMPT_FALLBACK


def _get_summarize_prompt() -> str:
    loaded = _load_prompt("deepthink_summarize")
    return loaded if loaded else _SUMMARIZE_SYSTEM_PROMPT_FALLBACK


def _parse_actions_from_content(
    content: str, step_id: str | None = None
) -> list[ClientActionInternal]:
    """AI 응답에서 action JSON 블록을 파싱한다.

    The prompt asks for ```actions, but some providers still emit ```json.
    Accept both when the JSON payload is an action object or action array.
    """
    pattern = r"```(?:actions|json)\s*\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    actions: list[ClientActionInternal] = []
    for match in matches:
        try:
            raw_list = json.loads(match.strip())
            if not isinstance(raw_list, list):
                raw_list = [raw_list]
            for item in raw_list:
                if not isinstance(item, dict) or "type" not in item:
                    continue
                normalized = normalize_action_payload(item)
                normalized.setdefault("step_id", step_id)
                actions.append(ClientActionInternal.model_validate(normalized))
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("failed to parse actions block: %s", exc)
    return actions


def _parse_plan_json(raw: str) -> dict[str, Any]:
    """AI의 플랜 응답에서 JSON을 추출한다.

    AI가 ```json 블록으로 감싸거나 순수 JSON으로 응답하는 경우 모두 처리.
    """
    # try extracting from code fence first
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1).strip())

    # try parsing raw as JSON
    stripped = raw.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)

    raise ValueError(f"cannot parse plan JSON from AI response: {raw[:200]}")


def _coerce_plan_steps(plan_data: Any) -> list[dict[str, Any]]:
    if isinstance(plan_data, dict):
        raw_steps = plan_data.get("steps", [])
    elif isinstance(plan_data, list):
        raw_steps = plan_data
    else:
        raw_steps = []

    steps: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_steps, start=1):
        if isinstance(item, dict):
            if "type" in item and "title" not in item:
                steps.append(
                    {
                        "id": item.get("step_id") or f"s{idx}",
                        "title": item.get("description") or item["type"],
                        "description": json.dumps(item, ensure_ascii=False),
                    }
                )
            else:
                steps.append(item)
        elif isinstance(item, str):
            steps.append({"id": f"s{idx}", "title": item, "description": item})
    return steps


def _fallback_actions_from_step(
    *,
    message: str,
    step: DeepThinkStepInput,
) -> list[ClientActionInternal]:
    text = f"{message}\n{step.title}\n{step.description}".lower()
    result_index = _extract_result_index(text)
    if result_index is not None and any(
        token in text
        for token in (
            "선택",
            "클릭",
            "들어가",
            "열어",
            "open",
            "click",
            "select",
        )
    ):
        return [
            ClientActionInternal(
                type="browser_control",
                command="select_result",
                target="active_tab",
                args={"index": result_index},
                description=f"현재 브라우저 검색 결과에서 {result_index}번째 항목 선택",
                requires_confirm=False,
                step_id=step.id,
            )
        ]

    if (
        "scroll" in text
        or "스크롤" in text
        or "내려" in text
        or "아래" in text
        or "page down" in text
    ):
        return [
            ClientActionInternal(
                type="browser_control",
                command="scroll",
                target="active_tab",
                args={"direction": "down", "amount": "page"},
                description="현재 브라우저 페이지를 아래로 스크롤",
                requires_confirm=False,
                step_id=step.id,
            )
        ]
    return []


def _extract_result_index(text: str) -> int | None:
    ordinal_words = {
        "첫번째": 1,
        "첫 번째": 1,
        "첫째": 1,
        "1번째": 1,
        "1번": 1,
        "두번째": 2,
        "두 번째": 2,
        "둘째": 2,
        "2번째": 2,
        "2번": 2,
        "세번째": 3,
        "세 번째": 3,
        "셋째": 3,
        "3번째": 3,
        "3번": 3,
        "네번째": 4,
        "네 번째": 4,
        "넷째": 4,
        "4번째": 4,
        "4번": 4,
        "다섯번째": 5,
        "다섯 번째": 5,
        "5번째": 5,
        "5번": 5,
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
    }
    for token, index in ordinal_words.items():
        if token in text:
            return index
    match = re.search(r"\b([1-9])\s*(?:st|nd|rd|th|번째|번)\b", text)
    if match:
        return int(match.group(1))
    return None


class DeepThinkService:
    def __init__(self, db: DBClient, ai_service: AIService) -> None:
        self.db = db
        self.ai_service = ai_service
        self.action_executor = ActionExecutor()

    def _select_deep_model(self, user_id: str) -> dict[str, Any]:
        """Use the user's deep selection first, then fall back to default."""
        selection = get_user_ai_selection(self.db, user_id=user_id)
        if selection is not None:
            deep_id = selection.get("deep_model_config_id")
            if deep_id:
                model = get_model_config_by_id_for_user(
                    self.db, user_id=user_id, model_config_id=deep_id
                )
                if model is not None and bool(model.get("is_active", True)):
                    logger.info(
                        "[deepthink] model selected via user deep_model_config_id "
                        "provider=%s/%s model=%s config_id=%s user=%s",
                        model["provider_mode"], model["provider_name"],
                        model["model_name"], model["id"], user_id,
                    )
                    return model

        config = get_active_model_for_user(self.db, user_id=user_id)
        if config is not None and bool(config.get("is_default", False)):
            logger.info(
                "[deepthink] model selected via default "
                "provider=%s/%s model=%s config_id=%s user=%s",
                config["provider_mode"], config["provider_name"],
                config["model_name"], config["id"], user_id,
            )
            return config

        if config is not None:
            logger.info(
                "[deepthink] model selected via active model "
                "provider=%s/%s model=%s config_id=%s user=%s",
                config["provider_mode"], config["provider_name"],
                config["model_name"], config["id"], user_id,
            )
            return config

        logger.warning(
            "[deepthink] no model configured, using env fallback user=%s",
            user_id,
        )
        return {
            "id": "default-deep",
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
            "supports_realtime": False,
            "transport": "http_sse",
            "input_modalities": "text",
            "output_modalities": "text",
        }

    def _build_ai_request(
        self,
        model: dict[str, Any],
        system_prompt: str,
        user_message: str,
        request_id: str,
        context_messages: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}]
        for message in context_messages or []:
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not content:
                continue
            if len(messages) == 1 and role == "assistant":
                continue
            if messages[-1]["role"] == role:
                messages[-1]["content"] += f"\n\n{content}"
            else:
                messages.append({"role": role, "content": content})

        if messages[-1]["role"] == "user":
            messages[-1]["content"] += f"\n\n{user_message}"
        else:
            messages.append({"role": "user", "content": user_message})

        return {
            "message": user_message,
            "route": "deep",
            "request_id": request_id,
            "provider_mode": model["provider_mode"],
            "provider_name": model["provider_name"],
            "model_name": model["model_name"],
            "api_key": model.get("api_key"),
            "endpoint": model.get("endpoint"),
            "system_prompt": system_prompt,
            "messages": messages,
        }

    def _get_chat_session_id(self, user_id: str) -> str:
        session = get_or_create_session_for_user(
            self.db,
            user_id=user_id,
            email="unknown@local.jarvis",
        )
        return str(session["id"])

    def _build_user_context(
        self,
        *,
        user_id: str,
        chat_id: str,
        route: str = "deep",
    ) -> tuple[str, list[dict[str, str]]]:
        settings = ensure_user_settings(self.db, user_id=user_id)
        persona = get_selected_persona_for_user(self.db, user_id=user_id)
        if persona is None:
            persona = ensure_default_persona_for_user(self.db, user_id=user_id)

        memories = list_memory_items(self.db, user_id=user_id, chat_id=chat_id, limit=5)
        summary = get_latest_chat_summary(self.db, chat_id=chat_id)
        recent_messages = list_recent_messages(self.db, chat_id, limit=12)
        metadata = settings.get("metadata", {}) if isinstance(settings, dict) else {}
        persona_hint = metadata.get("persona_hint") if isinstance(metadata, dict) else None
        custom_instructions = (
            metadata.get("custom_instructions") if isinstance(metadata, dict) else None
        )
        memory_lines = [
            f"- ({item['type']}/{item['importance']}) {item['content']}"
            for item in memories
        ]
        context_parts = [
            "## User Context",
            f"Persona: {persona['prompt_template']}",
            f"Tone: {persona['tone'] or 'balanced'}",
            f"Route mode: {route}",
            f"User locale: {settings['locale']}",
            f"User timezone: {settings['timezone']}",
            f"Preferred response style: {settings['response_style']}",
        ]
        if persona_hint:
            context_parts.append(f"Persona hint from user settings: {persona_hint}")
        if custom_instructions:
            context_parts.append(f"Custom user instructions: {custom_instructions}")
        if memory_lines:
            context_parts.append("Relevant memory:\n" + "\n".join(memory_lines))
        if summary is not None and summary.get("summary_text"):
            context_parts.append("Conversation summary:\n" + summary["summary_text"])
        return "\n\n".join(context_parts), recent_messages

    def _with_user_context(
        self,
        *,
        system_prompt: str,
        user_id: str,
        chat_id: str,
    ) -> tuple[str, list[dict[str, str]]]:
        user_context, recent_messages = self._build_user_context(
            user_id=user_id,
            chat_id=chat_id,
        )
        return f"{system_prompt}\n\n{user_context}", recent_messages

    # ── AI 기반 플래닝 ─────────────────────────────────────

    async def plan(
        self,
        body: DeepThinkPlanInternalRequest,
        user_id: str,
    ) -> DeepThinkPlanInternalResponse:
        """AI deep model을 사용해서 실행 플랜을 생성한다."""
        model = self._select_deep_model(user_id)
        chat_id = self._get_chat_session_id(user_id)
        system_prompt, context_messages = self._with_user_context(
            system_prompt=_get_planning_prompt(),
            user_id=user_id,
            chat_id=chat_id,
        )
        add_message(self.db, chat_id, "user", body.message)
        logger.info(
            "deepthink plan request_id=%s model=%s/%s",
            body.request_id,
            model["provider_name"],
            model["model_name"],
        )

        ai_result = await self.ai_service.respond_once(
            self._build_ai_request(
                model=model,
                system_prompt=system_prompt,
                user_message=body.message,
                request_id=body.request_id,
                context_messages=context_messages,
            )
        )

        try:
            plan_data = _parse_plan_json(ai_result["content"])
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("failed to parse AI plan: %s", exc)
            # fallback: 단일 스텝 플랜
            return DeepThinkPlanInternalResponse(
                request_id=body.request_id,
                goal=body.message,
                steps=[
                    DeepThinkStepInput(
                        id="s1",
                        title="요청 처리",
                        description=body.message,
                    )
                ],
                constraints=[],
            )

        steps = [
            DeepThinkStepInput(
                id=step.get("id", f"s{idx}"),
                title=step.get("title", f"Step {idx}"),
                description=step.get("description", ""),
            )
            for idx, step in enumerate(_coerce_plan_steps(plan_data), start=1)
        ]

        return DeepThinkPlanInternalResponse(
            request_id=body.request_id,
            goal=plan_data.get("goal", body.message) if isinstance(plan_data, dict) else body.message,
            steps=steps,
            constraints=plan_data.get("constraints", []) if isinstance(plan_data, dict) else [],
        )

    # ── AI 기반 실행 ───────────────────────────────────────

    # SUMMARIZE 프롬프트도 yaml에서 로드

    async def execute(
        self,
        body: DeepThinkInternalRequest,
        user_id: str,
    ) -> DeepThinkInternalResponse:
        """플래닝 단계를 순차적으로 AI에게 실행시키고, 각 단계에서 action을 추출한다.

        ActionExecutor가 서버 실행 가능한 액션(web_search 등)을 직접 처리하고,
        결과를 다음 단계 컨텍스트에 주입한다.
        클라이언트 실행 액션(terminal, mouse 등)은 response.actions로 반환한다.
        """
        model = self._select_deep_model(user_id)
        chat_id = self._get_chat_session_id(user_id)
        execution_prompt, context_messages = self._with_user_context(
            system_prompt=_get_execution_prompt(),
            user_id=user_id,
            chat_id=chat_id,
        )
        logger.info(
            "deepthink execute request_id=%s model=%s/%s steps=%d",
            body.request_id,
            model["provider_name"],
            model["model_name"],
            len(body.plan_steps),
        )

        step_results: list[DeepThinkStepOutput] = []
        all_client_actions: list[ClientActionInternal] = []
        accumulated_context: list[str] = list(body.execution_context)

        for step in body.plan_steps:
            step_prompt = (
                f"[현재 단계: {step.title}]\n"
                f"원래 요청: {body.message}\n"
                f"이 단계의 목표: {step.description}\n"
            )
            if accumulated_context:
                step_prompt += (
                    "\n이전 단계 결과:\n"
                    + "\n".join(accumulated_context)
                    + "\n"
                )

            try:
                # 1) AI가 이 단계에서 필요한 액션을 생성
                ai_result = await self.ai_service.respond_once(
                    self._build_ai_request(
                        model=model,
                        system_prompt=execution_prompt,
                        user_message=step_prompt,
                        request_id=body.request_id,
                        context_messages=context_messages,
                    )
                )
                step_content = ai_result["content"]
                step_actions = _parse_actions_from_content(
                    step_content, step_id=step.id
                )
                if not step_actions:
                    step_actions = _fallback_actions_from_step(
                        message=body.message,
                        step=step,
                    )
                    if step_actions:
                        step_content = (
                            f"{step_content}\n\n"
                            "[fallback] 기본 브라우저 제어 액션을 생성했습니다."
                        )

                # 2) ActionExecutor가 분류·실행
                exec_result: ExecutionResult = (
                    await self.action_executor.execute(step_actions)
                )

                # 3) 서버 실행 결과가 있으면 컨텍스트에 주입
                if exec_result.has_server_results:
                    accumulated_context.append(
                        exec_result.get_server_context()
                    )

                    # 검색 결과 → AI에게 요약 요청
                    search_text = exec_result.get_search_results_text()
                    if search_text:
                        step_content = await self._summarize_search(
                            model=model,
                            original_request=body.message,
                            search_results=search_text,
                            request_id=body.request_id,
                        )

                step_results.append(
                    DeepThinkStepOutput(
                        step_id=step.id,
                        title=step.title,
                        status="completed",
                        content=step_content,
                        actions=exec_result.client_actions,
                    )
                )
                all_client_actions.extend(exec_result.client_actions)
                accumulated_context.append(
                    f"- {step.title}: {step_content[:500]}"
                )

            except Exception as exc:
                logger.error(
                    "deepthink step failed step_id=%s error=%s",
                    step.id,
                    exc,
                )
                step_results.append(
                    DeepThinkStepOutput(
                        step_id=step.id,
                        title=step.title,
                        status="failed",
                        content=f"단계 실행 실패: {exc}",
                        actions=[
                            ClientActionInternal(
                                type="notify",
                                description=f"단계 '{step.title}' 실행 실패: {exc}",
                                requires_confirm=False,
                                step_id=step.id,
                            )
                        ],
                    )
                )

        completed = [s for s in step_results if s.status == "completed"]
        summary = (
            f"{len(completed)}/{len(step_results)} 단계 완료. "
            + "; ".join(s.title for s in completed)
        )
        content = "\n\n".join(
            f"### {s.title}\n{s.content}" for s in step_results
        )
        if content:
            add_message(self.db, chat_id, "assistant", content)

        return DeepThinkInternalResponse(
            request_id=body.request_id,
            steps=step_results,
            summary=summary,
            content=content,
            actions=all_client_actions,
        )

    async def _summarize_search(
        self,
        *,
        model: dict[str, Any],
        original_request: str,
        search_results: str,
        request_id: str,
    ) -> str:
        """검색 결과를 AI에게 요약시켜 사용자 답변을 생성한다."""
        summary_prompt = (
            f"원래 요청: {original_request}\n\n"
            f"아래는 웹 검색 결과입니다. "
            f"사용자의 요청에 맞게 검색 결과를 요약해서 답변해주세요.\n\n"
            f"검색 결과:\n{search_results}"
        )
        result = await self.ai_service.respond_once(
            self._build_ai_request(
                model=model,
                system_prompt=_get_summarize_prompt(),
                user_message=summary_prompt,
                request_id=request_id,
            )
        )
        return result["content"]
