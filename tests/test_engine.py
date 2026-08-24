from core.config.engine import run_deep_thinking, run_realtime_conversation


def test_realtime_conversation_returns_low_latency_shape() -> None:
    result = run_realtime_conversation("배포 상태 알려줘")

    assert result.mode == "realtime"
    assert "실시간 응답" in result.content
    assert len(result.next_actions) == 2


def test_deep_thinking_detects_analysis_signals() -> None:
    result = run_deep_thinking(
        """
        Traceback: bad state
        이 로그를 보고 설계 관점에서 원인 분석해줘.
        """
    )

    assert result.mode == "deep"
    assert "error-log" in result.content
    assert "analysis-request" in result.content
