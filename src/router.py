DEEP_KEYWORDS = {
    "plan",
    "analyze",
    "analysis",
    "design",
    "architecture",
    "tradeoff",
    "debug",
    "root cause",
    "계획",
    "분석",
    "설계",
    "아키텍처",
    "구조",
    "비교",
    "우선순위",
    "트레이드오프",
    "원인",
    "디버그",
}


def choose_route(message: str, task_type: str = "general") -> str:
    lowered = message.lower()
    if task_type == "analysis":
        return "deep"
    if len(message) >= 220:
        return "deep"
    if any(keyword in lowered for keyword in DEEP_KEYWORDS):
        return "deep"
    return "realtime"
