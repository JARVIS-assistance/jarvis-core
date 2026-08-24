# jarvis-core Runbook

## Local startup

1. `python3.12 -m pip install -r requirements.txt
python3.12 -m pip install -r requirements-dev.txt`
2. `python3.12 -m uvicorn app:app --app-dir src --reload --port 8000`

## Health check

- `curl http://127.0.0.1:8000/health`

## Common issues

- `No module named jarvis_contracts`: requirements 설치 누락
- DB 파일 권한 문제: `JARVIS_CORE_DB` 경로를 쓰기 가능한 위치로 변경

## Test/Lint

- `python3.12 -m pytest`
- `ruff check .`
